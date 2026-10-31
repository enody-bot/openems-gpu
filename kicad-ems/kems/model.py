"""Step 2: build the openEMS model of one study from the KiCad board, as plain JSON for the GPU runner (fdtd_run.py).

    python -m kems.model study.json [--out models/NAME.json]

study.json (paths relative to it):
  {
    "name": "SPI_SCLK",
    "board": "../board/my.kicad_pcb",
    "nets": ["/SPI_SCLK", "/SPI_MOSI"],                 nets modelled in detail (their routes get the fine mesh)
    "excite": "/SPI_SCLK",                              the net whose driver carries the source
    "drivers": {"U801.60": 66, "U801.61": 66},          driver pads -> output resistance (ohm): lumped ports
    "loads": {"default_C": 4e-12, "U301.5": 4e-12},     every other pad of the nets: a capacitor to its reference
    "open_refs": ["J", "TP"],                           pads of these parts are left open (probe only)
    "passives": {"R939": 33.0},                         optional: values for 2-pad parts on the nets (else Value)
    "self_c": {"L1": 3e-12},                            optional: winding C of inductors over L_SERIES_MAX (else 2 pF)
    "region": "auto",                                   or [x0, y0, x1, y1] (KiCad mm), or "board"; auto = the
                                                        nets + margin
    "margin_mm": 3.0,
    "fmax_hz": 3e9,
    "mesh": {"fine": 0.05, "region": 0.1, "board": 1.0, "air_mm": 40, "dz": 0.05},
    "outside": "planes",                                copper outside the region: one plane per layer of its
                                                        dominant net ("planes") or the real fills ("fills")
    "far_field": true,
    "excitation": {"kind": "gauss"},                    or {"kind": "edge", "rise_s": 0.8e-9, "tend_s": 8e-9}
    "td_planes": [{"name": "E_mid", "kind": "E", "between": ["In4.Cu", "In3.Cu"]},
                  {"name": "H_ref", "kind": "H", "face": "In3.Cu", "side": -1},
                  {"name": "E_air", "kind": "E", "above": "F.Cu", "h_mm": 0.25}]
  }

The model (units mm, KiCad page x / y; z from the bottom face of the board):
  sheets      zero-thickness PEC per copper layer: the outer layers on their inner face, the inner layers mid-copper
  dielectric  the stackup's dielectrics (each sub-layer) inside the board outline; loss as a conductivity fixed at
              1 GHz (sigma = 2 pi f e0 er tan d): right at 1 GHz, too lossy far below it
  region      every fill, track, pad and via of the board inside the region, as saved
  outside     each layer's dominant plane net as one plane, and the ground vias with an antipad in the planes of
              other nets (or the real fills with --outside fills; the check reports shorts the coarse cells make)
  decoupling  capacitors between a rail and ground as their ESL alone (a lumped L between the two planes at the
              part); never a series R-L-C: openEMS runs it as a float32 recursion that goes unstable
  drivers     lumped ports from the pad down to its reference layer; the excite net's driver carries the source
  loads       lumped capacitors pad to reference; series parts on the nets as lumped R / C / L bridged in the air
              above their pads, like the part's body
  probes      a voltage probe at every port, load and passive pad
  mesh        the thirds rule at every long straight edge of the detailed nets (a line h/3 inside the metal and
              2h/3 outside), a uniform lattice in the region offset a quarter cell from the board's grid, graded
              to the board and air cells; z lines at every sheet and a few per dielectric
Everything that openEMS resolves by edge centres is moved onto mesh nodes (on_mesh); `python -m kems.check` then
verifies the model on its mesh before any paid run."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

from . import geom
from .board import Board, is_ground, ring_area, short

C0 = 299792458.0
EPS0 = 8.8541878128e-12
F_LOSS = 1e9
DELTA = 1e-3                 # mm: half-width of a box that must hold exactly one mesh line
L_SERIES_MAX = 100e-9        # H: a bigger series inductor is modelled as its winding C (see METHOD.md)
SELF_C = 2e-12               # F: that C when the line gives no self_c for the part
ESL_NH = [('0201', 0.30), ('0402', 0.40), ('0603', 0.50), ('0805', 0.60), ('1206', 0.90), ('1210', 1.0),
          ('1812', 1.2), ('2220', 1.3)]
PRIO = dict(plane=5, antipad=8, fill=10, track=20, pad=20, via=30, post=30)


def kappa(er, tand):
    return 2 * math.pi * F_LOSS * EPS0 * er * tand


def parse_value(v, kind):
    """'33', '33R', '4k7', '100n', '1u', '0R', '10pF' -> a number in SI units (None if it cannot be read)."""
    s = v.strip().replace('Ω', '').replace('ohm', '').replace('Ohm', '')
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*([pnuµmkKMRrFfHh]?)([0-9]*)', s)
    if not m:
        return None
    num, unit, frac = m.groups()
    x = float(num + ('.' + frac if frac and unit in 'pnuµmkKMRr' and unit else ''))
    mult = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'm': 1e-3, 'k': 1e3, 'K': 1e3, 'M': 1e6, 'R': 1, 'r': 1}
    x *= mult.get(unit, 1.0)
    if kind == 'C' and unit not in ('p', 'n', 'u', 'µ', 'm'):
        x *= 1e-12 if x > 1 else 1.0            # a bare number on a capacitor is pF in most libraries
    return x


def esl_of(fp):
    for size, L in ESL_NH:
        if size in fp['lib'] or size in fp['value']:
            return L * 1e-9
    return 1.0e-9


# --------------------------------------------------------------------------------------------------- mesh helpers
def smooth(lines, max_res, ratio=1.3):
    """Fill between fixed lines so no cell exceeds max_res and neighbours grow by at most ~ratio: CSXCAD's
    SmoothMeshLines when openEMS' Python is installed, else a graded fill of each gap from both ends."""
    lines = np.unique(np.round(lines, 5))
    try:
        from CSXCAD.SmoothMeshLines import SmoothMeshLines
        return np.unique(np.round(SmoothMeshLines(lines, max_res, ratio), 5))
    except ImportError:
        pass
    out = [float(lines[0])]
    for k in range(len(lines) - 1):
        a, b = float(lines[k]), float(lines[k + 1])
        hl = out[-1] - out[-2] if len(out) > 1 else max_res
        hr = float(lines[k + 2] - lines[k + 1]) if k + 2 < len(lines) else max_res
        L, R = [a], [b]
        while True:
            nl, nr = min(hl * ratio, max_res), min(hr * ratio, max_res)
            if R[-1] - L[-1] <= max(nl, nr) * 1.5:
                break
            if nl <= nr:
                L.append(L[-1] + nl)
                hl = nl
            else:
                R.append(R[-1] - nr)
                hr = nr
        gap = R[-1] - L[-1]
        n = max(1, int(math.ceil(gap / max(nl, nr) - 0.25)))      # cells within 0.75-1.25 x the last
        mid = [L[-1] + gap * i / n for i in range(1, n)]
        out += L[1:] + mid + R[::-1]
    return np.unique(np.round(out, 5))


def thirds_lines(edges, lo, hi, h):
    """edges: [(coordinate, +1 metal above / -1 below, length)] of long straight metal edges. Lines h/3 inside the
    metal and 2h/3 outside, longest edges first, each kept only 2h/3 from every line already kept; the gaps then cut
    into cells of at most 1.3 h. On a uniform lattice an unresolved strip edge loads a line with capacitance (measured:
    7.95 ps/mm on 0.05 mm cells against the field solver's 7.05; with these lines 7.04)."""
    merged = {}
    for c, side, ln in sorted(edges):
        key = next((k for k in merged if k[1] == side and abs(k[0] - c) < 0.01), None)
        if key is None:
            merged[(c, side)] = ln
        else:
            merged[key] += ln
    cand = []
    for (c, side), ln in merged.items():
        cand += [(ln, c + side * h / 3), (ln, c - side * 2 * h / 3)]
    keep = []
    for ln, v in sorted(cand, key=lambda q: -q[0]):
        if lo < v < hi and all(abs(v - k) >= 2 * h / 3 - 1e-9 for k in keep):
            keep.append(v)
    return sorted(keep)


def thin(lines, keep, min_sep):
    """Drop lines closer than min_sep to a neighbour, never one of `keep`."""
    keep = set(np.round(keep, 5))
    out = []
    for v in np.round(sorted(lines), 5):
        if out and v - out[-1] < min_sep:
            if v in keep and out[-1] not in keep:
                out[-1] = v
            continue
        out.append(v)
    return np.array(out)


def axis(fixed_region, region, res, board, board_res, air, air_max, fine):
    """Region lattice (offset a quarter cell from the board's grid, so 0.05 mm pad and track edges fall on neither
    lines nor edge centres, where openEMS' inside test is a coin toss) + the thirds-rule lines, no cell under 2/3 of
    `fine` except between two thirds-rule lines; graded to `board_res` on the board and `air_max` in the air."""
    lo, hi = region
    o = lo + res / 4
    lat = np.arange(o, hi, res)
    fixed = np.array(sorted(fixed_region))
    if len(fixed):
        # around the thirds-rule lines, cells of about `fine` out to 2 fine from them, then the lattice
        near = np.concatenate([fixed + d for d in (-2 * fine, -fine, fine, 2 * fine)])
        lat = [v for v in lat if np.min(np.abs(fixed - v)) >= 2.5 * fine]
        cand = np.r_[lat, near]
    else:
        cand = lat
    lines = thin(np.r_[cand, fixed], fixed, 2 * fine / 3 - 1e-6)
    inner = smooth(lines, res)
    inner = thin(inner[(inner >= lo - 1e-9) & (inner <= hi + 1e-9)], fixed, 2 * fine / 3 - 1e-6)
    b_lo, b_hi = board
    on_board = smooth(np.r_[inner, b_lo, b_hi], board_res)
    return smooth(np.r_[on_board, b_lo - air, b_hi + air], air_max)


# --------------------------------------------------------------------------------------------------- model
class Builder:
    def __init__(self, spec, base):
        self.spec, self.base = spec, base
        self.B = Board(os.path.join(base, spec['board']))
        self.nets = list(spec['nets'])
        self.excite = spec.get('excite', self.nets[0])
        self.notes = []

    def note(self, s):
        self.notes.append(s)
        print('  note:', s)

    # z of each copper sheet and the dielectric slabs between them
    def stack(self):
        B = self.B
        Z = {}
        for i, L in enumerate(B.copper):
            z0, z1 = B.z[L]
            Z[L] = z0 if i == 0 else (z1 if i == len(B.copper) - 1 else (z0 + z1) / 2)   # F: bottom face, B: top
            # on the mesh's 1e-5 mm grid: openEMS places a zero-thickness sheet only on a z line equal to its
            # elevation, and a stackup sum such as 1.5740000000000003 misses the line 1.574 (the sheet is dropped,
            # with only an "Unused primitive" warning)
            Z[L] = round(Z[L], 5)
        slabs = []
        for a, b in zip(B.copper[:-1], B.copper[1:]):            # top to bottom: a above b
            zt, zb = Z[a], Z[b]
            ds = [d for d in B.dielectrics() if d['z0'] >= B.z[b][1] - 1e-6 and d['z1'] <= B.z[a][0] + 1e-6]
            if not ds:
                ds = [dict(t=zt - zb, er=4.5, tand=0.02, name='assumed')]
                self.note('no dielectric between %s and %s in the stackup: er 4.5 assumed' % (a, b))
            t = sum(d['t'] for d in ds)
            z = zb
            for d in sorted(ds, key=lambda d: d.get('z0', 0)):
                h = (zt - zb) * d['t'] / t                      # stretched over the copper's half-thickness
                slabs.append(dict(z0=round(z, 5), z1=round(z + h, 5), er=d['er'], tand=d['tand'], name=d['name']))
                z += h
            slabs[-1]['z1'] = zt
        self.Z, self.slabs = Z, slabs

    def region(self):
        sp, B = self.spec, self.B
        if sp.get('region') == 'board':                        # emission: the whole board radiates
            r = list(B.bbox)
        elif sp.get('region', 'auto') != 'auto':
            r = [float(v) for v in sp['region']]
        else:
            xs, ys = [], []
            for t in B.tracks:
                if t['net'] in self.nets:
                    xs += [t['a'][0], t['b'][0]]
                    ys += [t['a'][1], t['b'][1]]
            for p in B.pads:
                if p['net'] in self.nets:
                    xs.append(p['at'][0])
                    ys.append(p['at'][1])
            if not xs:
                raise SystemExit('none of the nets %s is on the board' % self.nets)
            m = sp.get('margin_mm', 3.0)
            r = [min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m]
        bx0, by0, bx1, by1 = B.bbox
        self.R = [max(r[0], bx0), max(r[1], by0), min(r[2], bx1), min(r[3], by1)]

    def build(self):
        sp, B = self.spec, self.B
        self.stack()
        self.region()
        x0, y0, x1, y1 = self.R
        Z = self.Z
        M = dict(format='kems-model/1', name=sp['name'], units='mm', z=Z, region=self.R, nets=self.nets,
                 excite=self.excite, metals=defaultdict(list), vias=[], holes=[], dielectrics=[], lumped=[], ports=[],
                 probes=[], boxes=[], wires=[])
        self.M = M

        def add(net, layer, ring, kind):
            if len(ring) >= 3 and abs(ring_area(ring)) > 1e-6:
                M['metals'][net or 'NONE'].append(dict(z=Z[layer], ring=[(round(x, 5), round(y, 5)) for x, y in ring],
                                                       prio=PRIO[kind], kind=kind, layer=layer))

        def inside(p, pad=0.0):
            return x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad
        # ---- the region: the board's copper as saved
        for f in B.fills:
            add(f['net'], f['layer'], geom.clip_rect(f['ring'], x0, y0, x1, y1), 'fill')
        for t in B.tracks:
            ring = geom.track_poly(t['a'], t['b'], t['w'])
            add(t['net'], t['layer'], geom.clip_rect(ring, x0, y0, x1, y1), 'track')
        for p in B.pads:
            if inside(p['at']):
                for L in p['layers']:
                    add(p['net'], L, p['poly'], 'pad')
        for v in B.vias:
            if not inside(v['at']):
                continue
            h = v['drill'] / 2
            zs = [Z[L] for L in v['layers']]
            M['vias'].append(dict(net=v['net'] or 'NONE', x0=v['at'][0] - h, x1=v['at'][0] + h, y0=v['at'][1] - h,
                                  y1=v['at'][1] + h, z0=min(zs), z1=max(zs), prio=PRIO['via'], at=list(v['at'])))
            for L in v['layers']:
                add(v['net'], L, geom.circle(v['at'], v['size'] / 2, 8), 'pad')
        # ---- outside the region
        strips = [(B.bbox[0] - 1, B.bbox[1] - 1, x0, B.bbox[3] + 1), (x1, B.bbox[1] - 1, B.bbox[2] + 1, B.bbox[3] + 1),
                  (x0, B.bbox[1] - 1, x1, y0), (x0, y1, x1, B.bbox[3] + 1)]
        strips = [s for s in strips if s[2] - s[0] > 1e-6 and s[3] - s[1] > 1e-6]
        board_area = abs(ring_area(B.outline))
        self.plane_net = {}
        outside = sp.get('outside', 'planes')
        for L in B.copper:
            area = defaultdict(float)
            for f in B.fills:
                if f['layer'] == L:
                    area[f['net']] += abs(ring_area(f['ring']))
            if area:
                net, a = max(area.items(), key=lambda kv: kv[1])
                if a > 0.25 * board_area:
                    self.plane_net[L] = net
            for s in strips:
                if outside == 'fills':
                    for f in B.fills:
                        if f['layer'] == L:
                            add(f['net'], L, geom.clip_rect(f['ring'], *s), 'plane')
                elif L in self.plane_net:
                    add(self.plane_net[L], L, geom.clip_rect(B.outline, *s), 'plane')
        self.outside_vias = []
        for v in B.vias:
            if inside(v['at']) or not is_ground(v['net']):
                continue
            span = [L for L in v['layers'] if L in self.plane_net]
            if len(span) < 2:
                continue
            h = v['drill'] / 2
            zs = [Z[L] for L in v['layers']]
            M['vias'].append(dict(net=v['net'], x0=v['at'][0] - h, x1=v['at'][0] + h, y0=v['at'][1] - h,
                                  y1=v['at'][1] + h, z0=min(zs), z1=max(zs), prio=PRIO['via'], at=list(v['at']),
                                  outside=True, antipad_layers=[L for L in v['layers'] if
                                                                self.plane_net.get(L) != v['net']]))
        # ---- dielectrics (inside the outline)
        for s in self.slabs:
            M['dielectrics'].append(dict(name='%s_%.3f' % (s['name'], s['z0']), z0=s['z0'], z1=s['z1'], er=s['er'],
                                         kappa=kappa(s['er'], s['tand']), ring=[list(p) for p in B.outline], prio=1))
        self.mesh()
        self.terminals()
        self.decaps()
        self.on_mesh()
        self.finish()
        return M

    # ---------------------------------------------------------------------------------------------- mesh
    def mesh(self):
        sp, B, M = self.spec, self.B, self.M
        ms = dict(fine=0.05, region=0.1, board=1.0, air_mm=40.0, dz=0.05, min_run=3.0)
        ms.update(sp.get('mesh', {}))
        fmax = sp.get('fmax_hz', 3e9)
        air_max = min(8.0, C0 / fmax * 1e3 / 20)
        x0, y0, x1, y1 = self.R
        ex, ey = [], []                     # (coordinate, side, length): edges along y give x lines and v.v.
        for t in B.tracks:
            if t['net'] not in self.nets:
                continue
            (ax, ay), (bx, by) = t['a'], t['b']
            w = t['w'] / 2
            if abs(ay - by) < 1e-3 and abs(ax - bx) >= ms['min_run']:
                ey += [(ay - w, +1, abs(ax - bx)), (ay + w, -1, abs(ax - bx))]
            if abs(ax - bx) < 1e-3 and abs(ay - by) >= ms['min_run']:
                ex += [(ax - w, +1, abs(ay - by)), (ax + w, -1, abs(ay - by))]
        h = ms['fine']
        fx = thirds_lines(ex, x0, x1, h)
        fy = thirds_lines(ey, y0, y1, h)
        bx0, by0, bx1, by1 = B.bbox
        xs = axis(fx, (x0, x1), ms['region'], (bx0, bx1), ms['board'], ms['air_mm'], air_max, h)
        ys = axis(fy, (y0, y1), ms['region'], (by0, by1), ms['board'], ms['air_mm'], air_max, h)
        zl = [self.Z[L] for L in B.copper]
        for s in self.slabs:
            n = int(np.clip(math.ceil((s['z1'] - s['z0']) / ms['dz'] - 1e-9), 2, 8))
            zl += list(np.linspace(s['z0'], s['z1'], n + 1))
        zl = np.unique(np.round(zl, 5))
        zs = smooth(np.r_[zl, zl.min() - ms['air_mm'], zl.max() + ms['air_mm']], air_max)
        M['mesh'] = dict(x=[float(v) for v in xs], y=[float(v) for v in ys], z=[float(v) for v in zs])
        M['mesh_spec'] = ms
        M['fmax_hz'] = fmax
        M['freqs'] = [float(f) for f in np.linspace(fmax / 600, fmax, 600)]
        exc = sp.get('excitation', {'kind': 'gauss'})
        M['excitation'] = exc
        M['f0'], M['fc'] = fmax / 2, fmax / 2
        M['nf2ff_center_m'] = [(bx0 + bx1) / 2e3, (by0 + by1) / 2e3, B.thickness / 2e3]
        M['far_field'] = bool(sp.get('far_field', False))

    # ---------------------------------------------------------------------------------------------- terminals
    def reference_layer(self, pad, L):
        """The copper layer next to `L` (nearest in z first) that holds plane copper under the pad centre."""
        B = self.B
        order = sorted([c for c in B.copper if c != L], key=lambda c: abs(self.Z[c] - self.Z[L]))
        shapes = self._layer_shapes()
        for c in order:
            nets = shapes[c].nets_at([pad['at']])[0]
            if nets and not (nets & set(self.nets)):
                return c, sorted(nets, key=lambda n: not is_ground(n))[0]
        raise SystemExit('no reference copper under %s.%s' % (pad['ref'], pad['num']))

    def _layer_shapes(self):
        if not hasattr(self, '_ls'):
            self._ls = {c: geom.Shapes() for c in self.B.copper}
            for f in self.B.fills:
                self._ls[f['layer']].add(f['net'], f['ring'])
            for L, net in self.plane_net.items():
                for p in self.M['metals'][net]:
                    if p['kind'] == 'plane' and p['layer'] == L:
                        self._ls[L].add(net, p['ring'])
        return self._ls

    def terminals(self):
        sp, B, M = self.spec, self.B, self.M
        drivers = {k: float(v) for k, v in sp.get('drivers', {}).items()}
        loads = dict(sp.get('loads', {}))
        default_C = float(loads.pop('default_C', 3e-12))
        open_refs = tuple(sp.get('open_refs', ['J', 'TP', 'H']))
        pval = sp.get('passives', {})
        x0, y0, x1, y1 = self.R
        done_parts, nr = set(), 0
        by_ref = {f['ref']: f for f in B.footprints}
        for p in B.pads:
            if p['net'] not in self.nets or not p['layers']:
                continue
            name = '%s.%s' % (p['ref'], p['num'])
            fp = by_ref.get(p['ref'])
            L = p['layers'][0] if p['layers'][0] in ('F.Cu', 'B.Cu') else p['layers'][0]
            if fp and fp['side'] == 'B' and 'B.Cu' in p['layers']:
                L = 'B.Cu'
            tag = '%s_%s' % (re.sub(r'[^A-Za-z0-9]+', '_', short(p['net'])), name.replace('.', '_'))
            fp_pads = [B.pads[i] for i in fp['pads'] if B.pads[i]['layers']] if fp else []
            two_pad = fp is not None and len(fp_pads) == 2 and re.match(r'(R|C|L|FB)\d', p['ref'])
            if name in drivers:
                ref_L, ref_net = self.reference_layer(p, L)
                nr += 1
                M['ports'].append(dict(nr=nr, name=tag, R=drivers[name], dir='z', excite=1 if p['net'] == self.excite else 0,
                                       start=[p['at'][0], p['at'][1], self.Z[ref_L]], stop=[p['at'][0], p['at'][1], self.Z[L]],
                                       net=p['net'], pad=name, net_hi=p['net'], net_lo=ref_net, layer=L, ref_layer=ref_L,
                                       region=p['poly']))
                M['probes'].append(dict(name='V_' + tag, start=[p['at'][0], p['at'][1], self.Z[L]],
                                        stop=[p['at'][0], p['at'][1], self.Z[ref_L]], net_hi=p['net'], net_lo=ref_net,
                                        pad=name, region=p['poly']))
            elif two_pad:
                if p['ref'] in done_parts:
                    continue
                done_parts.add(p['ref'])
                a, b = fp_pads
                kind = re.match(r'(R|C|L|FB)', p['ref']).group(1)
                v = pval.get(p['ref'])
                if v is None:
                    v = parse_value(fp['value'], kind if kind != 'FB' else 'R')
                if v is None:
                    self.note('%s: value %r not understood; left out' % (p['ref'], fp['value']))
                    continue
                el = dict(name='%s_%s' % (kind, p['ref']), R=None, C=None, L=None, series=False, ref=p['ref'],
                          value=v, kind=kind, pads=[a, b], net=p['net'])
                if kind in ('R', 'FB'):
                    el['R'] = max(v, 0.01)               # a 0 ohm link as 10 mohm
                elif kind == 'C':
                    if is_ground(a['net']) or is_ground(b['net']):
                        el['C'] = v                      # to ground on the net: a plain C (no series ESL)
                    else:
                        el['C'] = v
                elif v > L_SERIES_MAX:
                    # a power inductor is far above its self-resonance in the solver's band, so it is its winding
                    # capacitance; a lumped L this big also takes openEMS to NaN within a few thousand steps
                    el['C'] = sp.get('self_c', {}).get(p['ref'], SELF_C)
                    el['as'] = 'winding C'
                    self.note('%s: %.3g H above %.0f nH, modelled as its winding capacitance %.3g F%s' % (
                        p['ref'], v, L_SERIES_MAX * 1e9, el['C'], '' if p['ref'] in sp.get('self_c', {}) else
                        ' (assumed: give self_c from the data sheet SRF, C = 1 / ((2 pi SRF)^2 L))'))
                else:
                    el['L'] = v
                M['lumped'].append(el)
                for q in (a, b):
                    if q['net'] in self.nets or True:
                        try:
                            ref_L, ref_net = self.reference_layer(q, L)
                        except SystemExit:
                            continue
                        qt = '%s_%s_%s' % (re.sub(r'[^A-Za-z0-9]+', '_', short(q['net'])), q['ref'], q['num'])
                        M['probes'].append(dict(name='V_' + qt, start=[q['at'][0], q['at'][1], self.Z[L]],
                                                stop=[q['at'][0], q['at'][1], self.Z[ref_L]], net_hi=q['net'] or 'NONE',
                                                net_lo=ref_net, pad='%s.%s' % (q['ref'], q['num']), region=q['poly']))
            else:
                ref_L, ref_net = self.reference_layer(p, L)
                probe = dict(name='V_' + tag, start=[p['at'][0], p['at'][1], self.Z[L]],
                             stop=[p['at'][0], p['at'][1], self.Z[ref_L]], net_hi=p['net'], net_lo=ref_net, pad=name,
                             region=p['poly'])
                M['probes'].append(probe)
                if p['ref'] in open_refs or re.match(r'^(%s)\d' % '|'.join(map(re.escape, open_refs)), p['ref']):
                    continue                                    # a prefix (J, TP) or a whole ref (U801): open
                C = float(loads.get(name, default_C))
                if C <= 0:
                    continue                                    # "default_C": 0 leaves the pads open (probe only)
                M['lumped'].append(dict(name='C_' + tag, dir='z', start=[p['at'][0], p['at'][1], self.Z[ref_L]],
                                        stop=[p['at'][0], p['at'][1], self.Z[L]], R=None, C=C, L=None, series=False,
                                        net_hi=p['net'], net_lo=ref_net, pad=name, region=p['poly']))
        if not any(p['excite'] for p in M['ports']):
            raise SystemExit('no driver on the excite net %s: add its pad to "drivers"' % self.excite)

    def decaps(self):
        """Rail-to-ground capacitors (not on the detailed nets) as their ESL between the two planes at the part."""
        B, M = self.B, self.M
        shapes = self._layer_shapes()
        n = 0
        for fp in B.footprints:
            if not re.match(r'C\d', fp['ref']):
                continue
            pads = [B.pads[i] for i in fp['pads'] if B.pads[i]['layers']]
            nets = {p['net'] for p in pads}
            if len(pads) != 2 or len(nets) != 2 or nets & set(self.nets) or not any(is_ground(x) for x in nets):
                continue
            gnd = [x for x in nets if is_ground(x)][0]
            rail = (nets - {gnd}).pop()
            at = fp['at']
            have = {c: shapes[c].nets_at([at])[0] for c in B.copper}
            Lr = [c for c in B.copper if rail in have[c]]
            Lg = [c for c in B.copper if gnd in have[c]]
            if not Lr or not Lg:
                continue
            a, b = min(((r, g) for r in Lr for g in Lg), key=lambda q: abs(self.Z[q[0]] - self.Z[q[1]]))
            v = parse_value(fp['value'], 'C') or 0.0
            el = dict(name='Cdec_%s' % fp['ref'], dir='z', R=None, series=False, ref=fp['ref'], radius=3.0,
                      start=[at[0], at[1], min(self.Z[a], self.Z[b])], stop=[at[0], at[1], max(self.Z[a], self.Z[b])],
                      net_hi=rail if self.Z[a] > self.Z[b] else gnd, net_lo=gnd if self.Z[a] > self.Z[b] else rail)
            if v >= 1e-9:
                el.update(C=None, L=esl_of(fp))
            else:
                el.update(C=v, L=None)
            M['lumped'].append(el)
            n += 1
        M['n_decaps'] = n

    # ---------------------------------------------------------------------------------------------- placement
    def on_mesh(self):
        from .check import Grid
        M = self.M
        xs, ys, zs = (np.array(M['mesh'][a]) for a in 'xyz')

        def nearest(lines, v):
            return float(lines[int(np.argmin(np.abs(lines - v)))])
        # vias: at least one x and one y line inside the barrel (a z-directed edge)
        for v in M['vias']:
            for a0, a1, lines in (('x0', 'x1', xs), ('y0', 'y1', ys)):
                if not np.any((lines >= v[a0] - 1e-9) & (lines <= v[a1] + 1e-9)):
                    c = nearest(lines, (v[a0] + v[a1]) / 2)
                    v[a0], v[a1] = min(v[a0], c - DELTA), max(v[a1], c + DELTA)
            for L in v.pop('antipad_layers', []):        # outside: the via through a plane of another net
                ix = np.where((xs >= v['x0'] - 1e-9) & (xs <= v['x1'] + 1e-9))[0]
                iy = np.where((ys >= v['y0'] - 1e-9) & (ys <= v['y1'] + 1e-9))[0]
                i0, i1, j0, j1 = ix[0], ix[-1], iy[0], iy[-1]
                if i0 == 0 or j0 == 0 or i1 >= len(xs) - 1 or j1 >= len(ys) - 1:
                    continue
                M['holes'].append(dict(start=[float(xs[i0] - 0.75 * (xs[i0] - xs[i0 - 1])),
                                              float(ys[j0] - 0.75 * (ys[j0] - ys[j0 - 1])), self.Z[L]],
                                       stop=[float(xs[i1] + 0.75 * (xs[i1 + 1] - xs[i1])),
                                             float(ys[j1] + 0.75 * (ys[j1 + 1] - ys[j1])), self.Z[L]],
                                       prio=PRIO['antipad'], er=4.0, kappa=kappa(4.0, 0.02)))
        G = Grid(M)

        def place(e):
            region = e.pop('region', None)
            radius = e.pop('radius', 0.6)
            x0, y0 = e['start'][0], e['start'][1]
            zhi, zlo = max(e['start'][2], e['stop'][2]), min(e['start'][2], e['stop'][2])
            cx, cy = xs[np.abs(xs - x0) <= radius], ys[np.abs(ys - y0) <= radius]
            cand = sorted(((x, y) for x in cx for y in cy), key=lambda q: (q[0] - x0) ** 2 + (q[1] - y0) ** 2)
            from matplotlib.path import Path
            path = Path(np.asarray(region)) if region is not None else None
            for x, y in cand:
                if path is not None and not path.contains_point((x, y)):
                    continue
                if G.node_net(x, y, zhi) == e['net_hi'] and G.node_net(x, y, zlo) == e['net_lo']:
                    return float(x), float(y)
            return None
        dropped = []
        for coll in ('ports', 'lumped', 'probes'):
            keep = []
            for e in M[coll]:
                if coll == 'lumped' and 'pads' in e:
                    keep.append(e)
                    continue
                q = place(e)
                if q is None:
                    if coll == 'ports' or (coll == 'lumped' and e.get('C') is not None and 'Cdec' not in e['name']):
                        raise SystemExit('%s: no mesh node on its pad with %s above and %s below; refine the mesh '
                                         'there or move the element' % (e['name'], e['net_hi'], e['net_lo']))
                    dropped.append(e['name'])
                    continue
                x, y = q
                if coll == 'probes':
                    e['start'], e['stop'] = [x, y, e['start'][2]], [x, y, e['stop'][2]]
                else:
                    e['start'] = [x - DELTA, y - DELTA, e['start'][2]]
                    e['stop'] = [x + DELTA, y + DELTA, e['stop'][2]]
                e['node'] = [x, y]
                keep.append(e)
            M[coll] = keep
        if dropped:
            self.note('%d elements without a mesh node of the right nets were left out: %s' % (
                len(dropped), ', '.join(dropped[:12]) + (' ...' if len(dropped) > 12 else '')))
        # two-pad parts: a PEC post up from each pad's inner edge to the first z line above the sheet, the lumped
        # element between the posts up there (the part's body); the gap between the pads on the board stays as drawn
        top, bot = self.B.copper[0], self.B.copper[-1]
        for el in [e for e in M['lumped'] if 'pads' in e]:
            a, b = el.pop('pads')
            layer = top if top in a['layers'] else bot
            zs_sheet = self.Z[layer]
            k = int(np.argmin(np.abs(zs - zs_sheet)))
            z_air = float(zs[k + 1] if layer == top else zs[k - 1])
            ax_ = 0 if abs(a['at'][0] - b['at'][0]) >= abs(a['at'][1] - b['at'][1]) else 1
            tr = 1 - ax_
            lines, tlines = (xs, ys) if ax_ == 0 else (ys, xs)
            lo, hi = sorted((a, b), key=lambda p: p['at'][ax_])
            plo, phi = np.asarray(lo['poly']), np.asarray(hi['poly'])
            g0 = lines[(lines <= plo[:, ax_].max() + 1e-9) & (lines >= plo[:, ax_].min() - 1e-9)]
            g1 = lines[(lines >= phi[:, ax_].min() - 1e-9) & (lines <= phi[:, ax_].max() + 1e-9)]
            if not len(g0) or not len(g1):
                raise SystemExit('%s: no mesh line on a pad (pads %.2f mm wide); refine the mesh' % (
                    el['name'], min(np.ptp(plo[:, ax_]), np.ptp(phi[:, ax_]))))
            g0, g1 = float(g0.max()), float(g1.min())
            c = (lo['at'][tr] + hi['at'][tr]) / 2
            w = 0.4 * min(np.ptp(plo[:, tr]), np.ptp(phi[:, tr]))

            def on(t, g, net):
                x, y = (g, t) if ax_ == 0 else (t, g)
                return G.node_net(x, y, zs_sheet) == (net or 'NONE')
            use = [float(t) for t in tlines[(tlines >= c - w) & (tlines <= c + w)] if on(t, g0, lo['net']) and on(t, g1, hi['net'])]
            if not use:
                raise SystemExit('%s: no mesh line with both ends on pad copper; refine the mesh' % el['name'])
            t0, t1 = min(use) - DELTA, max(use) + DELTA
            el['dir'] = 'xy'[ax_]
            el['start'], el['stop'] = ([g0, t0, z_air], [g1, t1, z_air]) if ax_ == 0 else ([t0, g0, z_air], [t1, g1, z_air])
            el['gap'], el['gap_nets'] = [g0, g1], [lo['net'] or 'NONE', hi['net'] or 'NONE']
            for g, net in ((g0, lo['net']), (g1, hi['net'])):
                box = dict(x0=g - DELTA, x1=g + DELTA, y0=t0, y1=t1) if ax_ == 0 else dict(x0=t0, x1=t1, y0=g - DELTA, y1=g + DELTA)
                M['vias'].append(dict(net=net or 'NONE', z0=min(zs_sheet, z_air), z1=max(zs_sheet, z_air),
                                      prio=PRIO['post'], post=el['name'], **box))

    def finish(self):
        M, sp = self.M, self.spec
        xs, ys, zs = (np.array(M['mesh'][a]) for a in 'xyz')
        dmin = [float(np.min(np.diff(a))) for a in (xs, ys, zs)]
        dt = 1.0 / (C0 * 1e3 * math.sqrt(sum(1.0 / d ** 2 for d in dmin)))
        M['td_planes'] = []
        for P in sp.get('td_planes', []):
            M['td_planes'].append(dict(P, z=self.plane_z(P)))
        M['metals'] = dict(M['metals'])
        M['stats'] = dict(nx=len(xs), ny=len(ys), nz=len(zs), cells=int((len(xs) - 1) * (len(ys) - 1) * (len(zs) - 1)),
                          dmin=dmin, dt_s=dt, n_metal_polys=sum(len(v) for v in M['metals'].values()),
                          n_vias=len(M['vias']), n_ports=len(M['ports']), n_lumped=len(M['lumped']),
                          n_probes=len(M['probes']))
        h = hashlib.sha256(open(self.B.path, 'rb').read()).hexdigest()
        M['meta'] = dict(board=os.path.abspath(self.B.path), board_sha256=h, spec=sp, notes=self.notes,
                         plane_net=self.plane_net, copper=self.B.copper,
                         stack=[{k: v for k, v in s.items()} for s in self.B.stack])
        M.pop('boxes', None)

    def plane_z(self, P):
        zs = sorted(self.M['mesh']['z'])
        Z = self.Z
        if 'between' in P:
            a, b = Z[P['between'][0]], Z[P['between'][1]]
            want = a + P.get('frac', 0.5) * (b - a)
            inside = [z for z in zs if min(a, b) + 1e-6 < z < max(a, b) - 1e-6]
            return min(inside, key=lambda z: abs(z - want))
        if 'face' in P:
            z0 = Z[P['face']]
            return max(z for z in zs if z < z0 - 1e-6) if P.get('side', -1) < 0 else min(z for z in zs if z > z0 + 1e-6)
        if 'above' in P:
            return min(zs, key=lambda z: abs(z - (Z[P['above']] + P.get('h_mm', 0.25))))
        raise SystemExit('td plane %s: give between, face or above' % P.get('name'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('study')
    ap.add_argument('--out')
    a = ap.parse_args()
    spec = json.load(open(a.study))
    base = os.path.dirname(os.path.abspath(a.study))
    out = a.out or os.path.join(base, 'models', spec['name'] + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    M = Builder(spec, base).build()
    json.dump(M, open(out, 'w'))
    s = M['stats']
    print('%s: mesh %d x %d x %d = %.1f M cells, dmin %s mm, dt %.1f fs; %d polygons, %d vias, %d ports, %d lumped '
          '(%d decaps as ESL), %d probes -> %s (%.1f MB)' % (
              spec['name'], s['nx'], s['ny'], s['nz'], s['cells'] / 1e6, [round(v, 4) for v in s['dmin']],
              s['dt_s'] * 1e15, s['n_metal_polys'], s['n_vias'], s['n_ports'], s['n_lumped'], M['n_decaps'],
              s['n_probes'], out, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
