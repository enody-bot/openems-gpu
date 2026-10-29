"""Render step 1: one run's field planes and the board's layers as textures for the Blender scene (scene.py).

    python -m kems.render.textures MODEL.json RUN_OUT --dest SCENE_DIR [--swing 3.3] [--ppmm 20] [--range-db 50]

RUN_OUT is a run's out/<name> folder from fdtd_run.py --td-planes (fields_<plane>.npz, traces.npz, run.json); the
model's study must have asked for `td_planes`, and an edge excitation gives the clearest picture (one edge running the
line, then its reflections). Writes into SCENE_DIR:
  layer_<Cu>.png      each copper layer inside the model's region, RGBA: copper where there is metal, the studied nets
                      brighter, the excited net brightest; rows = KiCad y downward
  frames/<plane>/f_NNNN.png   every dump frame, 8-bit log code on one scale per field kind (E, H): code c stands for
                      |field| = lo (hi / lo) ** (c / 255), hi the 99.9th percentile of the strongest frame, lo
                      --range-db below it; 0 is "below lo"
  scene.json          format kems-scene/1: box (KiCad mm), copper layers with z (mm above the board's bottom face),
                      dielectric slabs, planes with z, frame count, time step, scale and unit, the nets.
The fields are scaled to a source swinging 0 -> --swing V (openEMS' soft source depends on the port's R, so its EMF
differs from run to run; the scale is the swing over the settled EMF u + R i of the driven port). The board is only
read (model meta.board)."""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from ..board import Board
from ..geom import track_poly

COPPER, NET_CU, EXC_CU = (150, 96, 52), (230, 160, 90), (255, 214, 150)


def weights(src, dst):
    i = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
    w = np.clip((dst - src[i]) / (src[i + 1] - src[i]), 0, 1)
    return i, w


def resample(a, ix, wx, iy, wy):
    a = a.astype(np.float32)
    top = a[iy][:, ix] * (1 - wx) + a[iy][:, ix + 1] * wx
    bot = a[iy + 1][:, ix] * (1 - wx) + a[iy + 1][:, ix + 1] * wx
    return top * (1 - wy)[:, None] + bot * wy[:, None]


def code(v, lo, hi):
    c = np.log(np.maximum(v, 1e-30) / lo) / math.log(hi / lo)
    return ((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8) * (v >= lo)).astype(np.uint8)


def swing_scale(M, out, swing):
    tr = np.load(os.path.join(out, 'traces.npz'))
    p = [p for p in M['ports'] if p['excite']][0]
    k = p['nr']
    u = tr['port_ut_%d' % k] if 'port_ut_%d' % k in tr else tr['port_ut%d' % k]
    i = tr['port_it_%d' % k] if 'port_it_%d' % k in tr else tr['port_it%d' % k]
    emf = u + p['R'] * i
    end = float(np.median(emf[-max(5, len(emf) // 20):]))
    return swing / end if abs(end) > 1e-3 * np.abs(emf).max() else swing / float(np.abs(emf).max())


def layers(B, box, ppmm, nets, excite, dest):
    x0, y0, x1, y1 = box
    W, H = int(round((x1 - x0) * ppmm)), int(round((y1 - y0) * ppmm))

    def px(pts):
        return [((x - x0) * ppmm, (y - y0) * ppmm) for x, y in pts]
    out = []
    for L in B.copper:
        img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for pass_nets, col in ((None, COPPER), (set(nets) - {excite}, NET_CU), ({excite}, EXC_CU)):
            def want(n):
                return (pass_nets is None) or (n in pass_nets)
            for f in B.fills:
                if f['layer'] == L and want(f['net']):
                    d.polygon(px(f['ring']), fill=col + (255,))
            for t in B.tracks:
                if t['layer'] == L and want(t['net']):
                    d.polygon(px(track_poly(t['a'], t['b'], t['w'])), fill=col + (255,))
                    for c in (t['a'], t['b']):
                        (cx, cy), r = px([c])[0], t['w'] / 2 * ppmm
                        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (255,))
            for p in B.pads:
                if L in p['layers'] and want(p['net']):
                    d.polygon(px(p['poly']), fill=col + (255,))
            for v in B.vias:
                if L in v['layers'] and want(v['net']):
                    (cx, cy), r = px([v['at']])[0], v['size'] / 2 * ppmm
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (255,))
        for v in B.vias:                                                 # drills show through every layer
            (cx, cy), r = px([v['at']])[0], v['drill'] / 2 * ppmm
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 0))
        name = 'layer_%s.png' % L.replace('.', '_')
        img.save(os.path.join(dest, name), optimize=True)
        z0, z1 = B.z[L]
        out.append(dict(name=L, png=name, z_mm=round((z0 + z1) / 2, 4)))
    return out, [W, H]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('run')
    ap.add_argument('--dest', required=True)
    ap.add_argument('--board', help='default: the model\'s meta.board')
    ap.add_argument('--swing', type=float, default=3.3)
    ap.add_argument('--ppmm', type=float, default=20.0, help='texture pixels per mm')
    ap.add_argument('--range-db', type=float, default=50.0)
    a = ap.parse_args()
    M = json.load(open(a.model))
    run = json.load(open(os.path.join(a.run, 'run.json')))
    os.makedirs(a.dest, exist_ok=True)
    B = Board(a.board or M['meta']['board'])
    box = [round(float(v), 4) for v in M['region']]
    k = swing_scale(M, a.run, a.swing)
    copper, size = layers(B, box, a.ppmm, M.get('nets', []), M['excite'], a.dest)
    x0, y0, x1, y1 = box
    W, H = size
    planes, mags = [], {}
    for P in M.get('td_planes', []):
        f = os.path.join(a.run, 'fields_%s.npz' % P['name'])
        if os.path.exists(f):
            d = dict(np.load(f))
            d['k'] = float(d.get('peak', 1.0))            # fdtd_run.py stores the field / its peak
            mags[P['name']] = d
    hi = {}
    for name, d in mags.items():
        kind = [P for P in M['td_planes'] if P['name'] == name][0]['kind']
        strongest = int(np.argmax(d['fmax'])) if 'fmax' in d else int(np.argmax(d['mag'].reshape(len(d['mag']), -1).max(1)))
        hi[kind] = max(hi.get(kind, 0.0), k * d['k'] * float(np.percentile(d['mag'][strongest].astype(np.float32), 99.9)))
    hi = {kk: (v if v > 0 else 1.0) for kk, v in hi.items()}
    lo = {kk: v * 10 ** (-a.range_db / 20) for kk, v in hi.items()}
    t = None
    for P in M.get('td_planes', []):
        if P['name'] not in mags:
            continue
        d = mags[P['name']]
        xs, ys = d['x'] * 1e3, d['y'] * 1e3                    # openEMS writes the dump mesh in metres
        ix, wx = weights(xs, np.linspace(x0, x1, W))
        iy, wy = weights(ys, np.linspace(y0, y1, H))
        fdir = os.path.join(a.dest, 'frames', P['name'])
        os.makedirs(fdir, exist_ok=True)
        for n in range(len(d['mag'])):
            Image.fromarray(code(k * d['k'] * resample(d['mag'][n], ix, wx, iy, wy), lo[P['kind']], hi[P['kind']])).save(
                os.path.join(fdir, 'f_%04d.png' % (n + 1)), optimize=True)
        t = d['t']
        planes.append(dict(name=P['name'], kind=P['kind'], z_mm=P['z'], frames='frames/%s/f_####.png' % P['name'],
                           n=len(d['mag']), unit='V/m' if P['kind'] == 'E' else 'A/m',
                           scale=dict(lo=lo[P['kind']], hi=hi[P['kind']], map='log'),
                           between=P.get('between') or [P.get('face') or P.get('above')]))
        print("%-16s %4d frames %dx%d, %s scale %.3g..%.3g" % (P["name"], len(d["mag"]), W, H,
                                                                P['kind'], lo[P['kind']], hi[P['kind']]))
    diel = []
    zc = {L: B.z[L] for L in B.copper}
    for a_, b_ in zip(B.copper[:-1], B.copper[1:]):
        diel.append(dict(between=[a_, b_], z0=round(zc[b_][1], 4), z1=round(zc[a_][0], 4)))
    scene = dict(format='kems-scene/1', model=M['name'], board=os.path.basename(B.path), box=box, outline=B.outline,
                 thickness_mm=B.thickness, copper=copper, dielectrics=diel, planes=planes, texture_px=size,
                 t_ns=[float(t[0] * 1e9), float(np.median(np.diff(t)) * 1e9)] if t is not None else None,
                 nets=M.get('nets', []), excite=M['excite'], swing_v=a.swing, scale_to_swing=k,
                 engine=run.get('engine'), excitation=run.get('excitation'))
    json.dump(scene, open(os.path.join(a.dest, 'scene.json'), 'w'), indent=1)
    print('scene.json: %d copper layers, %d field planes; fields x%.4g to a %.2f V swing' % (
        len(copper), len(planes), k, a.swing))


if __name__ == '__main__':
    main()
