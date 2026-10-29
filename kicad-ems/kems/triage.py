"""Step 1: read a KiCad board and list the nets worth a full-wave look, the power rails, and copper that is not there.

    python -m kems.triage BOARD.kicad_pcb --out study/triage [--assume tr.json]

Writes study/triage/TRIAGE.md (the ranked list to show the engineer), triage.json (every number), and figures:
overview.png, refplane_<net>.png for the ranked nets, opens.png when copper opens were found.

Per signal net:
  class       from the net name and what it connects (clock, high-speed pair, fast bus, switch node, I2C / UART,
              analog sense, GPIO), with an assumed 10-90 % edge per class (override with --assume)
  length      routed length per layer, vias, the one-way delay with each layer's effective permittivity, and
              delay / edge: above 0.25 the net behaves as a transmission line
  reference   the track sampled every 0.25 mm; at each sample the zone fills on the copper layers directly above and
              below: GND, another net (power), or nothing. Reported as mm over no copper, mm over another net only,
              and the number of reference changes (split crossings)
  return via  at each signal via whose two layers have no reference layer in common, the distance to the nearest GND
              via that joins both references (the return current's detour)
  edge        the closest approach to the board outline
Copper connectivity: every net's tracks, vias, pads and fills joined by geometry alone. A net in several pieces is an
open; when only a footprint's jumper_pad_groups join the pieces, KiCad's DRC cannot see it (it treats the grouped pads
as joined inside the part). Check the part's datasheet: LP01 had ten open lines through flow-through ESD arrays.
Power rails: nets with zone fills or wide tracks, their layers, copper area and decoupling capacitors.

The classification is a heuristic from net names. Read it against the schematic and correct it before ranking."""
import argparse
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

from . import geom
from .board import CONTROL, Board, is_ground, is_power, short

C0_MM_PER_NS = 299.792458

# class: (pattern on the net name, assumed 10-90 % edge in ns, weight in the ranking, what to simulate)
CLASSES = [
    ('hs-pair', r'(USB.*D[PNM+-]?$|USB.*_(P|N)$|^D[PM]$|D[+-]$|DP$|DM$|_DP$|_DN$|_DM$|TX[PN]|RX[PN]|TD[PN]|RD[PN]|'
                r'MDI\w*|MIPI|CSI|DSI|HDMI|LVDS|PCIE|SATA|_P$|_N$)', 0.3, 3.0, 'SI full-wave: impedance, eye; EMI'),
    ('clock', r'(CLK|SCK|SCLK|XTAL|XIN|XOUT|OSC|MCLK|BCLK|LRCLK|WS$|RCLK|TCK)', 0.8, 3.0,
     'SI full-wave (ringing, edges at each load); EMI source'),
    ('switch-node', r'(^|[_/])(SW\d*|LX\d*|PH(ASE)?\d*|SW_NODE|SWN|BUCK_SW|BOOST_SW)$', 3.0, 3.0,
     'EMI source: radiated emission of the hot loop'),
    ('fast-bus', r'(MOSI|MISO|SDI|SDO|SIO\d|QSPI|SPI|SDIO|SD_|DQ\d*|DQS|DDR|ADDR|A\d+$|D\d+$|IO\d+|DATA|CMD|CS_?N?$|'
                 r'SS_?N?$|RMII|RGMII|MII|TXD|RXD|VSYNC|HSYNC|PCLK)', 1.0, 2.0, 'SI full-wave, crosstalk'),
    ('i2c-uart', r'(SCL|SDA|I2C|UART|_TX$|_RX$|TXD0|RXD0|SWDIO|SWCLK|JTAG|TMS|TDI|TDO)', 1.5, 1.5,
     'SI circuit / full-wave: undershoot at the MCU drive strength'),
    ('analog-sense', r'(SENSE|SNS|ISNS|IMON|VMON|ADC|AIN|FB$|_FB|VREF|REF$|NTC|TEMP_A|CS[+-]|CSP|CSN|KELVIN)', None, 2.0,
     'victim: coupling from fast nets nearby'),
]
DEFAULT_TR = 2.0            # ns, any other digital net (a microcontroller GPIO)
SAMPLE = 0.25               # mm between reference samples
RETURN_VIA_MM = 2.0         # a return via further than this is flagged
WIDE_TRACK = 0.5            # mm: a track this wide marks a power path


SW_PIN = re.compile(r'^(SW|LX|PH|PHASE|SW_?NODE)([_\d]\w*)?$', re.I)


def classify(net, B, pads_by_net, all_nets):
    name = short(net).upper()
    pads = pads_by_net.get(net, [])
    for cls, pat, tr, w, study in CLASSES:
        if re.search(pat, name):
            if cls == 'hs-pair' and re.search(r'(_P|_N|P|N|\+|-)$', name) and not re.search(r'((^|_)D[PMN+-]$|USB)', name):
                # a _P / _N name is a pair only when its partner exists (an active-low _N alone is not)
                stem, end = net[:-1], net[-1]
                partner = stem + {'P': 'N', 'N': 'P', '+': '-', '-': '+'}.get(end.upper(), '?')
                if partner not in all_nets and partner.lower() not in all_nets:
                    continue
            return cls, tr, w, study
    # a pad whose pin function is a switch output (SW, LX, PH) marks the switch node of a converter
    if any(SW_PIN.match(p.get('pinfunction', '')) for p in pads):
        return 'switch-node', 3.0, 3.0, 'EMI source: radiated emission of the hot loop'
    return 'gpio', DEFAULT_TR, 1.0, 'SI circuit model; full-wave if long'


def short_runs_dropped(states, ds, min_run=0.5):
    """States with runs of 'none' / 'other' / 'mixed' shorter than min_run mm turned into the state around them:
    a fill's antipad round the net's own via, or a pad's clearance, is not a reference problem."""
    st = list(states)
    i, n = 0, len(st)
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        if st[i] != 'gnd' and (j - i) * ds < min_run:
            fill = st[i - 1] if i > 0 else (st[j] if j < n else st[i])
            for k in range(i, j):
                st[k] = fill
        i = j
    return st


def er_eff(B, layer, w):
    """Effective permittivity of a track on `layer`: microstrip on the outer layers (Hammerstad), the mean of the
    dielectrics either side inside."""
    up, dn = B.neighbours(layer)
    diel = B.dielectrics()

    def er_between(a, b):
        za, zb = B.z[a], B.z[b]
        lo, hi = min(za[1], zb[1]), max(za[0], zb[0])
        ds = [d for d in diel if d['z0'] >= lo - 1e-6 and d['z1'] <= hi + 1e-6]
        t = sum(d['t'] for d in ds)
        return sum(d['er'] * d['t'] for d in ds) / t if t else 4.3, t
    if up is None or dn is None:
        er, h = er_between(layer, dn or up)
        u = w / max(h, 1e-3)
        return (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 / u)
    e1, _ = er_between(layer, up)
    e2, _ = er_between(layer, dn)
    return (e1 + e2) / 2


def connectivity(B):
    """Copper islands per net from geometry alone. Returns {net: [island, ...]} for nets with more than one island
    that holds a pad, each island a list of pad names, and which of them jumper_pad_groups would join."""
    by_net = defaultdict(lambda: dict(t=[], v=[], p=[], f=[]))
    for i, t in enumerate(B.tracks):
        by_net[t['net']]['t'].append(i)
    for i, v in enumerate(B.vias):
        by_net[v['net']]['v'].append(i)
    for i, p in enumerate(B.pads):
        if p['layers']:
            by_net[p['net']]['p'].append(i)
    for i, f in enumerate(B.fills):
        by_net[f['net']]['f'].append(i)
    out = {}
    for net, it in by_net.items():
        if not net or net.startswith('unconnected-') or len(it['p']) < 2:
            continue
        nodes = [('t', i) for i in it['t']] + [('v', i) for i in it['v']] + [('p', i) for i in it['p']] + \
                [('f', i) for i in it['f']]
        ix = {n: k for k, n in enumerate(nodes)}
        parent = list(range(len(nodes)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def join(a, b):
            ra, rb = find(ix[a]), find(ix[b])
            if ra != rb:
                parent[ra] = rb
        layer_items = defaultdict(list)        # layer -> [(node, outline array)]
        for i in it['t']:
            t = B.tracks[i]
            layer_items[t['layer']].append((('t', i), np.asarray(geom.track_poly(t['a'], t['b'], t['w']))))
        for i in it['p']:
            p = B.pads[i]
            for L in p['layers']:
                layer_items[L].append((('p', i), np.asarray(p['poly'])))
        for i in it['v']:
            v = B.vias[i]
            for L in v['layers']:
                layer_items[L].append((('v', i), np.asarray(geom.circle(v['at'], v['size'] / 2))))
        for i in it['f']:
            f = B.fills[i]
            layer_items[f['layer']].append((('f', i), np.asarray(f['ring'])))
        # a through-hole pad or via joins its layers (the barrel): every layer entry of one node is the same node
        for L, items in layer_items.items():
            boxes = np.array([np.r_[r.min(0), r.max(0)] for _, r in items])
            for a in range(len(items)):
                na, ra = items[a]
                cand = np.where((boxes[a + 1:, 0] <= boxes[a, 2] + 1e-3) & (boxes[a + 1:, 2] >= boxes[a, 0] - 1e-3) &
                                (boxes[a + 1:, 1] <= boxes[a, 3] + 1e-3) & (boxes[a + 1:, 3] >= boxes[a, 1] - 1e-3))[0]
                for c in cand + a + 1:
                    nb, rb = items[c]
                    if find(ix[na]) != find(ix[nb]) and geom.touch(ra, rb):
                        join(na, nb)
        groups = defaultdict(list)
        for n in nodes:
            groups[find(ix[n])].append(n)
        islands = [[B.pads[i]['ref'] + '.' + B.pads[i]['num'] for k, i in g if k == 'p'] for g in groups.values()]
        islands = [s for s in islands if s]
        if len(islands) > 1:
            # would the footprints' jumper groups join them?
            jg = []
            for f in B.footprints:
                for g in f['jumper_groups']:
                    jg.append({f['ref'] + '.' + n for n in g})
            uf = list(range(len(islands)))

            def f2(a):
                while uf[a] != a:
                    a = uf[a]
                return a
            for g in jg:
                hit = [k for k, s in enumerate(islands) if g & set(s)]
                for k in hit[1:]:
                    uf[f2(k)] = f2(hit[0])
            hidden = len({f2(k) for k in range(len(islands))}) == 1
            out[net] = dict(islands=islands, joined_only_by_jumper_groups=hidden)
    return out


def scan(B, assume=None):
    assume = assume or {}
    fills = {L: geom.Shapes() for L in B.copper}
    for f in B.fills:
        fills[f['layer']].add(f['net'], f['ring'])
    pads_by_net = defaultdict(list)
    for p in B.pads:
        pads_by_net[p['net']].append(p)
    fill_area = defaultdict(float)
    fill_layers = defaultdict(set)
    from .board import ring_area
    for f in B.fills:
        fill_area[f['net']] += abs(ring_area(f['ring']))
        fill_layers[f['net']].add(f['layer'])
    tracks_by_net = defaultdict(list)
    for t in B.tracks:
        tracks_by_net[t['net']].append(t)
    vias_by_net = defaultdict(list)
    for v in B.vias:
        vias_by_net[v['net']].append(v)
    gnd_vias = [v for v in B.vias if is_ground(v['net'])]
    cap_to_gnd = defaultdict(list)          # net -> capacitors between it and a ground
    for f in B.footprints:
        if re.match(r'C\d', f['ref']):
            nets = {B.pads[i]['net'] for i in f['pads']}
            if len(nets) == 2 and any(is_ground(n) for n in nets):
                cap_to_gnd[(nets - {n for n in nets if is_ground(n)}).pop()].append(f)
    all_nets = set(B.nets)
    outline = np.asarray(B.outline + [B.outline[0]], float)

    def edge_dist(pts):
        a, b = outline[:-1], outline[1:]
        d = b - a
        L2 = np.maximum((d ** 2).sum(1), 1e-12)
        best = np.full(len(pts), 1e9)
        for k in range(len(a)):
            t = np.clip(((pts - a[k]) @ d[k]) / L2[k], 0, 1)
            q = a[k] + t[:, None] * d[k]
            best = np.minimum(best, np.hypot(*(pts - q).T))
        return best

    power, signals = [], []
    for net in B.nets:
        if not net or net.startswith('unconnected-') or is_ground(net):
            continue
        ts = tracks_by_net.get(net, [])
        wide = [t for t in ts if t['w'] >= WIDE_TRACK]
        decaps = [f['ref'] + ' ' + f['value'] for f in cap_to_gnd.get(net, [])]
        is_sw = any(SW_PIN.match(p.get('pinfunction', '')) for p in pads_by_net.get(net, [])) or \
            bool(re.search(CLASSES[2][1], short(net).upper()))
        control = bool(CONTROL.search(short(net)))
        if not is_sw and (is_power(net) or (not control and (fill_area.get(net, 0) > 4.0 or len(decaps) >= 2 or
                          (wide and sum(math.dist(t['a'], t['b']) for t in wide) > 5)))):
            srcs = sorted({p['ref'] for p in pads_by_net.get(net, []) if re.match(r'(U|L|Q|J|F|D)\d', p['ref'])})
            power.append(dict(net=net, fill_mm2=round(fill_area.get(net, 0), 1), layers=sorted(fill_layers.get(net, set()),
                              key=B.copper.index), narrowest_track_mm=min([t['w'] for t in ts], default=None),
                              decaps=decaps, n_decaps=len(decaps), parts=srcs))
            continue
        if (not ts and not is_sw) or len(pads_by_net.get(net, [])) < 2:
            continue
        cls, tr, weight, study = classify(net, B, pads_by_net, all_nets)
        tr = assume.get(net, assume.get(cls, tr))
        per_layer = defaultdict(float)
        delay = 0.0
        samples, states = [], []
        gap_mm = other_mm = mixed_mm = 0.0
        changes = 0
        own_holes = [(v['at'], v['size'] / 2 + 0.35) for v in vias_by_net.get(net, [])] + \
                    [(p['at'], max(p['size']) / 2 + 0.35) for p in pads_by_net.get(net, []) if p['drill'] > 0]
        for t in ts:
            L = math.dist(t['a'], t['b'])
            per_layer[t['layer']] += L
            delay += L * math.sqrt(er_eff(B, t['layer'], t['w'])) / C0_MM_PER_NS
            pts, ds = geom.samples_along(t['a'], t['b'], SAMPLE)
            up, dn = B.neighbours(t['layer'])
            refs = [fills[n].nets_at(pts) if n else [set()] * len(pts) for n in (up, dn)]
            seg_states, seg_refs = [], []
            for k in range(len(pts)):
                ru, rd = refs[0][k], refs[1][k]
                gu, gd = any(is_ground(n) for n in ru), any(is_ground(n) for n in rd)
                o = bool({n for n in ru | rd if not is_ground(n)})
                if any(math.dist(pts[k], c) < r for c, r in own_holes):
                    st = 'gnd'                   # inside the antipad of the net's own via or hole: not a gap
                elif (gu or gd) and o and up and dn and not (gu and gd):
                    st = 'mixed'
                elif gu or gd:
                    st = 'gnd'
                elif o:
                    st = 'other'
                else:
                    st = 'none'
                seg_states.append(st)
                seg_refs.append((tuple(sorted(ru)), tuple(sorted(rd))))
            seg_states = short_runs_dropped(seg_states, ds)
            gap_mm += ds * seg_states.count('none')
            other_mm += ds * seg_states.count('other')
            mixed_mm += ds * seg_states.count('mixed')
            # a reference change: the set of reference nets differs between two runs of at least 1 mm
            runs, k = [], 0
            while k < len(seg_refs):
                j = k
                while j < len(seg_refs) and seg_refs[j] == seg_refs[k]:
                    j += 1
                if (j - k) * ds >= 1.0 and any(seg_refs[k]):
                    runs.append(seg_refs[k])
                k = j
            changes += sum(1 for a, b in zip(runs[:-1], runs[1:]) if a != b)
            samples += [(float(p[0]), float(p[1])) for p in pts]
            states += seg_states
        # return vias
        worst_ret, n_far = 0.0, 0
        for v in vias_by_net.get(net, []):
            used = {t['layer'] for t in ts if min(math.dist(t['a'], v['at']), math.dist(t['b'], v['at'])) <= v['size'] / 2 + 0.01}
            used |= {p_ for p in pads_by_net.get(net, []) if math.dist(p['at'], v['at']) < 0.5 for p_ in p['layers']}
            if len(used) < 2:
                continue
            ref_sets = []
            for L in used:
                rs = set()
                for n in B.neighbours(L):
                    if n and any(is_ground(x) for x in fills[n].nets_at([v['at']])[0]):
                        rs.add(n)
                ref_sets.append(rs)
            common = set.intersection(*ref_sets) if ref_sets else set()
            if common or not all(ref_sets):
                continue
            need = set().union(*ref_sets)
            cand = [math.dist(g['at'], v['at']) for g in gnd_vias if need <= set(g['layers']) or len(set(g['layers']) & need) >= 2]
            d = min(cand) if cand else float('inf')
            worst_ret = max(worst_ret, d)
            if d > RETURN_VIA_MM:
                n_far += 1
        pts_all = np.array(samples) if samples else np.zeros((0, 2))
        edge = float(edge_dist(pts_all).min()) if len(pts_all) else None
        total = sum(per_layer.values())
        ratio = delay / tr if tr else 0.0
        # ranking: the class weight times what makes a net worth a 3-D model
        score = weight * (min(ratio, 3.0) + 0.5 * min(gap_mm, 10) + 0.2 * min(other_mm, 20) + 0.1 * min(mixed_mm, 20)
                          + 0.5 * min(changes, 10) + (1.0 if n_far else 0.0) + (0.5 if edge is not None and edge < 1.0 else 0))
        if cls == 'switch-node':
            score += 12.0                                  # an emission source regardless of length
        reasons = []
        if ratio > 0.25:
            reasons.append('electrically long: delay %.2f ns = %.1f x the %.2g ns edge' % (delay, ratio, tr))
        if gap_mm > 0.25:
            reasons.append('%.1f mm with no copper on the adjacent layers' % gap_mm)
        if other_mm > 0.25:
            reasons.append('%.1f mm referenced to a non-ground net only' % other_mm)
        if changes:
            reasons.append('%d reference changes along the route' % changes)
        if n_far:
            reasons.append('%d layer change(s) with the nearest return via %.1f mm away' % (n_far, worst_ret))
        if edge is not None and edge < 1.0:
            reasons.append('%.2f mm from the board edge' % edge)
        if cls == 'switch-node':
            reasons.append('switching node: an emission source')
        signals.append(dict(net=net, cls=cls, tr_ns=tr, study=study, length_mm=round(total, 2),
                            per_layer={k: round(v, 2) for k, v in per_layer.items()}, vias=len(vias_by_net.get(net, [])),
                            pads=sorted(p['ref'] + '.' + p['num'] for p in pads_by_net.get(net, [])),
                            delay_ns=round(delay, 3), delay_over_edge=round(ratio, 2), knee_mhz=round(350 / tr) if tr else None,
                            gap_mm=round(gap_mm, 2), other_ref_mm=round(other_mm, 2), mixed_ref_mm=round(mixed_mm, 2),
                            ref_changes=changes, worst_return_via_mm=round(worst_ret, 2) if worst_ret != float('inf') else None,
                            far_return_vias=n_far, edge_mm=round(edge, 2) if edge is not None else None,
                            score=round(score, 2), reasons=reasons, _samples=samples, _states=states))
    signals.sort(key=lambda s: -s['score'])
    power.sort(key=lambda p: -p['fill_mm2'])
    return signals, power


# ---------------------------------------------------------------------------------------------- figures
def _board_axes(B, ax):
    o = np.asarray(B.outline + [B.outline[0]])
    ax.plot(o[:, 0], o[:, 1], color='0.3', lw=0.8)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('x (mm, KiCad page)')
    ax.set_ylabel('y (mm)')


COLORS = {'hs-pair': '#d62728', 'clock': '#ff7f0e', 'switch-node': '#9467bd', 'fast-bus': '#1f77b4',
          'i2c-uart': '#2ca02c', 'analog-sense': '#8c564b', 'gpio': '#7f7f7f'}


def figures(B, signals, opens, out, top):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    W, H = B.bbox[2] - B.bbox[0], B.bbox[3] - B.bbox[1]
    fw = 14
    fig, ax = plt.subplots(figsize=(fw, max(3.0, fw * H / W * 0.9 + 1.2)))
    _board_axes(B, ax)
    for f in B.fills:
        if is_ground(f['net']):
            r = np.asarray(f['ring'])
            ax.fill(r[:, 0], r[:, 1], color='0.93', lw=0, zorder=0)
    shown = set()
    for s in signals[:top]:
        c = COLORS.get(s['cls'], 'k')
        for t in B.tracks:
            if t['net'] == s['net']:
                ax.plot([t['a'][0], t['b'][0]], [t['a'][1], t['b'][1]], color=c, lw=1.0,
                        label=s['cls'] if s['cls'] not in shown else None)
                shown.add(s['cls'])
        bad = [p for p, st in zip(s['_samples'], s['_states']) if st == 'none']
        if bad:
            b = np.asarray(bad)
            ax.plot(b[:, 0], b[:, 1], 'x', color='red', ms=3)
    ax.set_title('Top %d nets by risk, coloured by class; red x: no reference copper on the adjacent layers; grey: GND fills' % top)
    ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'overview.png'), dpi=150)
    plt.close(fig)
    for s in signals[:top]:
        pts = np.asarray(s['_samples'])
        if not len(pts):
            continue
        x0, y0 = pts.min(0) - 2
        x1, y1 = pts.max(0) + 2
        fig, ax = plt.subplots(figsize=(10, max(2.5, 10 * (y1 - y0) / max(x1 - x0, 1) + 1)))
        for t in B.tracks:
            if t['net'] == s['net']:
                ax.plot([t['a'][0], t['b'][0]], [t['a'][1], t['b'][1]], color='0.6', lw=2.5, zorder=1)
        col = {'gnd': '#2ca02c', 'mixed': '#bcbd22', 'other': '#ff7f0e', 'none': '#d62728'}
        st = np.array(s['_states'])
        for k, c in col.items():
            m = st == k
            if m.any():
                ax.plot(pts[m, 0], pts[m, 1], '.', ms=3, color=c, label=k, zorder=2)
        for v in B.vias:
            if v['net'] == s['net']:
                ax.plot(*v['at'], 'o', mfc='none', mec='k', ms=5, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect('equal')
        ax.set_title('%s (%s): reference under the route - green GND, yellow GND one side only, orange another net only, '
                     'red none; o: vias' % (short(s['net']), s['cls']), fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'refplane_%s.png' % re.sub(r'[^A-Za-z0-9_+-]+', '_', short(s['net']))), dpi=130)
        plt.close(fig)
    if opens:
        fig, ax = plt.subplots(figsize=(fw, max(3.0, fw * H / W * 0.9 + 1.2)))
        _board_axes(B, ax)
        for net, o in opens.items():
            pads = {p['ref'] + '.' + p['num']: p for p in B.pads if p['net'] == net}
            for k, isl in enumerate(o['islands']):
                xy = np.asarray([pads[n]['at'] for n in isl if n in pads])
                ax.plot(xy[:, 0], xy[:, 1], 'o', ms=4, label='%s piece %d' % (short(net), k + 1))
        ax.set_title('Nets in more than one copper piece (geometry only, jumper_pad_groups ignored)')
        if len(opens) <= 8:
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'opens.png'), dpi=150)
        plt.close(fig)


def write_md(B, signals, power, opens, out, top):
    L = []
    L.append('# Triage: %s\n' % os.path.basename(B.path))
    L.append('%d copper layers (%s), %.3f mm; %d nets, %d tracks, %d vias, %d zone fill polygons.%s\n' % (
        len(B.copper), ', '.join(B.copper), B.thickness, len(B.nets), len(B.tracks), len(B.vias), len(B.fills),
        ' **No stackup in the file: layers spread evenly, er 4.5 assumed.**' if B.stack_assumed else ''))
    L.append('Stackup (top to bottom): ' + '; '.join(
        ('%s %.3f' % (s['name'], s['t'])) if s['kind'] != 'dielectric' else ('%.3f er %.2f' % (s['t'], s['er']))
        for s in B.stack if s['kind'] != 'mask') + '\n')
    if opens:
        L.append('## Copper opens\n')
        L.append('Nets whose pads are not joined by copper (geometry only). "Hidden" = only a footprint\'s '
                 '`jumper_pad_groups` join the pieces, so KiCad DRC reports them as connected.\n')
        L.append('| Net | Pieces (pads) | Hidden from DRC |\n|---|---|---|')
        for net, o in sorted(opens.items(), key=lambda kv: not kv[1]['joined_only_by_jumper_groups']):
            L.append('| %s | %s | %s |' % (short(net), ' / '.join(', '.join(i) for i in o['islands']),
                                          '**yes**' if o['joined_only_by_jumper_groups'] else 'no (DRC shows it)'))
        L.append('')
    L.append('## Signal nets ranked by risk (top %d of %d)\n' % (min(top, len(signals)), len(signals)))
    L.append('Edges are assumed per class (the `tr` column); correct them from the drivers\' data sheets and drive '
             'settings. Delay / edge above 0.25 means the net is a transmission line.\n')
    L.append('| # | Net | Class | tr (ns) | Length (mm) | Vias | Delay / edge | No ref (mm) | Other-net ref (mm) | '
             'Ref changes | Return via (mm) | Why | Suggested study |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, s in enumerate(signals[:top]):
        L.append('| %d | %s | %s | %s | %.1f | %d | %.2f | %.1f | %.1f | %d | %s | %s | %s |' % (
            i + 1, short(s['net']), s['cls'], s['tr_ns'], s['length_mm'], s['vias'], s['delay_over_edge'], s['gap_mm'],
            s['other_ref_mm'], s['ref_changes'], s['worst_return_via_mm'] if s['worst_return_via_mm'] else '-',
            '; '.join(s['reasons']) or '-', s['study']))
    L.append('\n![overview](overview.png)\n')
    L.append('## Power rails\n')
    L.append('Candidates for the DC power-plane study (IR drop, current density, via currents) and the PDN impedance '
             'study (plane resonances, decoupling).\n')
    L.append('| Rail | Fill area (mm2) | Fill layers | Narrowest track (mm) | Decoupling caps | Parts on the rail |')
    L.append('|---|---|---|---|---|---|')
    for p in power:
        L.append('| %s | %.0f | %s | %s | %d | %s |' % (short(p['net']), p['fill_mm2'], ', '.join(p['layers']) or '-',
                                                       p['narrowest_track_mm'], p['n_decaps'], ', '.join(p['parts'][:8])))
    L.append('')
    open(os.path.join(out, 'TRIAGE.md'), 'w').write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('board')
    ap.add_argument('--out', default='triage')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--assume', help='JSON {net or class: edge in ns} to override the assumed edges')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = Board(a.board)
    assume = json.load(open(a.assume)) if a.assume else {}
    signals, power = scan(B, assume)
    opens = connectivity(B)
    figures(B, signals, opens, a.out, a.top)
    write_md(B, signals, power, opens, a.out, a.top)
    json.dump(dict(board=os.path.abspath(a.board), copper=B.copper, stack=B.stack, z=B.z, thickness=B.thickness,
                   stack_assumed=B.stack_assumed, signals=[{k: v for k, v in s.items() if not k.startswith('_')}
                                                           for s in signals],
                   power=power, opens=opens), open(os.path.join(a.out, 'triage.json'), 'w'), indent=1)
    print('%d signal nets ranked, %d power rails, %d nets with copper opens (%d hidden from DRC) -> %s' % (
        len(signals), len(power), len(opens), sum(o['joined_only_by_jumper_groups'] for o in opens.values()),
        os.path.join(a.out, 'TRIAGE.md')))


if __name__ == '__main__':
    main()
