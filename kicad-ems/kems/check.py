"""Step 3: check a model on its own mesh before any paid run, and estimate what the run costs.

    python -m kems.check models/NAME.json [--tend 20e-9] [--png]

openEMS decides metal edge by edge: an edge is PEC when its centre is inside a metal primitive (a point on the
primitive's boundary counts as inside), the highest priority wins, and a zero-size box between two mesh lines holds no
edge at all. Every check below works on that rule, on the mesh the run will use:
  ports      each port, load and probe on a mesh node, with its net's copper on the upper node and its reference's on
             the lower one (a port between two lines excites nothing; one on the wrong copper excites the wrong net)
  shorts     in the region, a node touched by PEC edges of two nets: the mesh has joined them. A short that involves
             a detailed net fails the check; others are listed (they change the ground system, not the line)
  pieces     each detailed net, walked over its PEC edges and via barrels, must be one piece holding all its
             terminals; a net in pieces on the mesh is open in the model even when the board is fine
  stability  no series R-L-C lumped element (openEMS' float32 recursion for it diverges at decoupling values), and
             no lumped L over 100 nH (a 3.3 uH power inductor took the solver to NaN within 3,000 steps)
  size       cells, time step, steps for the run length, GPU memory, and time and cost on the measured GPUs
Exit status 1 when a check fails."""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from matplotlib.path import Path

# measured on 37-43 M-cell board models: GCells/s and $/h (RunPod Secure Cloud / Modal, October 2026)
GPUS = [('RTX 3080 (local)', 5.1, 0.0), ('RTX A5000, RunPod', 5.0, 0.27), ('RTX 5090, RunPod', 13.8, 1.19),
        ('B200, Modal', 20.5, 6.9)]
BOOT_MIN = 3.0                 # pod start, image pull and copy-back, billed


class Grid:
    """Which net owns a point of a copper sheet in openEMS' terms (highest priority, boundary inside)."""

    def __init__(self, M):
        self.M = M
        self.by_z = defaultdict(list)                  # z -> [(prio, net, Path, bbox)]
        for net, polys in M['metals'].items():
            for p in polys:
                r = np.asarray(p['ring'], float)
                self.by_z[round(p['z'], 6)].append((p['prio'], net, Path(r), (r[:, 0].min(), r[:, 1].min(),
                                                                              r[:, 0].max(), r[:, 1].max())))
        self.vias = [(v['prio'], v['net'], v['x0'], v['y0'], v['x1'], v['y1'], v['z0'], v['z1']) for v in M['vias']]
        self.holes = [(h['prio'], h['start'][0], h['start'][1], h['stop'][0], h['stop'][1], h['start'][2])
                      for h in M.get('holes', [])]

    def owners(self, pts, z):
        """Owner net of each point (N, 2) on the plane z ('' = dielectric), and the winning priority."""
        pts = np.asarray(pts, float)
        best = np.full(len(pts), -1.0)
        net = np.array([''] * len(pts), dtype=object)
        zk = round(z, 6)
        for prio, n, path, (x0, y0, x1, y1) in self.by_z.get(zk, []):
            m = (pts[:, 0] >= x0 - 1e-7) & (pts[:, 0] <= x1 + 1e-7) & (pts[:, 1] >= y0 - 1e-7) & (pts[:, 1] <= y1 + 1e-7)
            m &= best < prio
            if not m.any():
                continue
            idx = np.where(m)[0]
            hit = idx[path.contains_points(pts[idx], radius=1e-7) | path.contains_points(pts[idx], radius=-1e-7)]
            best[hit], net[hit] = prio, n
        for prio, n, x0, y0, x1, y1, z0, z1 in self.vias:
            if not (z0 - 1e-7 <= z <= z1 + 1e-7):
                continue
            m = (pts[:, 0] >= x0 - 1e-7) & (pts[:, 0] <= x1 + 1e-7) & (pts[:, 1] >= y0 - 1e-7) & (pts[:, 1] <= y1 + 1e-7)
            m &= best < prio
            best[m], net[m] = prio, n
        for prio, x0, y0, x1, y1, hz in self.holes:
            if abs(hz - z) > 1e-6:
                continue
            m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & (best < prio)
            best[m], net[m] = prio, ''
        return net, best

    def node_net(self, x, y, z):
        """The net a node is joined to: the one owning the PEC edges round it ('' if none or several)."""
        h = 1e-4
        pts = [(x + h, y), (x - h, y), (x, y + h), (x, y - h)]
        nets, _ = self.owners(pts, z)
        s = {n for n in nets if n}
        return s.pop() if len(s) == 1 else ''


def check(M, tend=None, png=None):
    G = Grid(M)
    xs, ys, zs = (np.array(M['mesh'][a]) for a in 'xyz')
    x0, y0, x1, y1 = M['region']
    ix = np.where((xs >= x0 - 1e-6) & (xs <= x1 + 1e-6))[0]
    iy = np.where((ys >= y0 - 1e-6) & (ys <= y1 + 1e-6))[0]
    X, Y = xs[ix], ys[iy]
    nets = set(M['nets'])
    fails, warns, info = [], [], []
    # ---- sheets exactly on z lines: openEMS drops a zero-thickness polygon off its line (1.5740000000000003 is not
    # 1.574) with no more than an "Unused primitive" warning, and a port onto it then drives nothing
    zset = set(float(z) for z in zs)
    off = sorted({p['z'] for polys in M['metals'].values() for p in polys if float(p['z']) not in zset})
    if off:
        fails.append('copper sheets off the z mesh lines (exact match needed): z = %s' % ', '.join(repr(z) for z in off[:6]))
    for coll in ('ports', 'lumped'):
        for e in M[coll]:
            for z in (e['start'][2], e['stop'][2]):
                if float(z) not in zset:
                    fails.append('%s %s ends at z = %r, off the z mesh lines' % ({'ports': 'port', 'lumped': 'lumped element'}[coll], e['name'], z))
                    break
    # ---- terminals
    for coll in ('ports', 'lumped', 'probes'):
        for e in M[coll]:
            if 'node' not in e:
                continue
            x, y = e['node']
            if not (np.any(np.abs(xs - x) < 1e-6) and np.any(np.abs(ys - y) < 1e-6)):
                fails.append('%s %s is not on a mesh node' % (coll[:-1], e['name']))
                continue
            zhi, zlo = max(e['start'][2], e['stop'][2]), min(e['start'][2], e['stop'][2])
            hi, lo = G.node_net(x, y, zhi), G.node_net(x, y, zlo)
            if hi != e['net_hi'] or lo != e['net_lo']:
                fails.append('%s %s: upper node on %r (wants %r), lower on %r (wants %r)' % (
                    coll[:-1], e['name'], hi, e['net_hi'], lo, e['net_lo']))
    for e in M['lumped']:
        if e.get('series'):
            fails.append('%s is a series R-L-C: openEMS (float32) runs it unstable; use its ESL or a plain C' % e['name'])
        if e.get('L') is not None and e['L'] > 100e-9:
            fails.append('%s is a lumped L of %.3g H: above ~100 nH openEMS goes to NaN in a few thousand steps; '
                         'model it as its winding C (model.py does)' % (e['name'], e['L']))
        if sum(e.get(k) is not None for k in ('R', 'L', 'C')) > 1 and e.get('L') is not None and e.get('C') is not None:
            warns.append('%s has both L and C (a parallel LC): check its resonance is intended' % e['name'])
    # ---- edge owners per sheet in the region, shorts and pieces
    sheets = sorted(set(round(z, 6) for z in M['z'].values()))
    zname = {round(v, 6): k for k, v in M['z'].items()}
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    shorts = defaultdict(int)
    short_at = defaultdict(list)
    owner_maps = {}
    for z in sheets:
        kz = int(np.argmin(np.abs(zs - z)))
        cx = np.stack(np.meshgrid((X[:-1] + X[1:]) / 2, Y, indexing='ij'), -1).reshape(-1, 2)
        cy = np.stack(np.meshgrid(X, (Y[:-1] + Y[1:]) / 2, indexing='ij'), -1).reshape(-1, 2)
        ox = G.owners(cx, z)[0].reshape(len(X) - 1, len(Y))
        oy = G.owners(cy, z)[0].reshape(len(X), len(Y) - 1)
        owner_maps[z] = (ox, oy)
        inc = defaultdict(set)                     # node (i, j) -> nets of its PEC edges
        for (i, j) in zip(*np.nonzero(ox != '')):
            n = ox[i, j]
            inc[(i, j)].add(n)
            inc[(i + 1, j)].add(n)
            if n in nets:
                join((kz, ix[i], iy[j], n), (kz, ix[i + 1], iy[j], n))
        for (i, j) in zip(*np.nonzero(oy != '')):
            n = oy[i, j]
            inc[(i, j)].add(n)
            inc[(i, j + 1)].add(n)
            if n in nets:
                join((kz, ix[i], iy[j], n), (kz, ix[i], iy[j + 1], n))
        for (i, j), s in inc.items():
            if len(s) > 1:
                key = tuple(sorted(s))
                shorts[(zname.get(z, z), key)] += 1
                if len(short_at[(zname.get(z, z), key)]) < 3:
                    short_at[(zname.get(z, z), key)].append((round(float(X[i]), 3), round(float(Y[j]), 3)))
    # via barrels: z edges inside each detailed-net via box join its sheets
    for v in M['vias']:
        if v['net'] not in nets:
            continue
        vx = np.where((xs >= v['x0'] - 1e-9) & (xs <= v['x1'] + 1e-9))[0]
        vy = np.where((ys >= v['y0'] - 1e-9) & (ys <= v['y1'] + 1e-9))[0]
        kk = [k for k, z in enumerate(zs) if v['z0'] - 1e-6 <= z <= v['z1'] + 1e-6]
        ks = [k for k in kk if any(abs(zs[k] - s) < 1e-6 for s in sheets)]
        for i in vx:
            for j in vy:
                for a, b in zip(ks[:-1], ks[1:]):
                    join((a, i, j, v['net']), (b, i, j, v['net']))
                if v.get('post'):                  # a part's post: joins the sheet to the z line above it
                    join((kk[0], i, j, v['net']), (kk[-1], i, j, v['net']))
    # bridges: a lumped part's two posts are joined through the part (a different net on each side: fine)
    for (L, key), n in sorted(shorts.items(), key=lambda kv: -kv[1]):
        msg = '%s: %d nodes join %s, e.g. at %s' % (L, n, ' + '.join(key), short_at[(L, key)])
        (fails if set(key) & nets else warns).append('short on the mesh, ' + msg)
    # pieces: every terminal node of a detailed net in one piece
    for net in nets:
        terms = []
        for coll in ('ports', 'lumped', 'probes'):
            for e in M[coll]:
                if 'node' in e and e.get('net_hi') == net:
                    zhi = max(e['start'][2], e['stop'][2])
                    kz = int(np.argmin(np.abs(zs - zhi)))
                    i = int(np.argmin(np.abs(xs - e['node'][0])))
                    j = int(np.argmin(np.abs(ys - e['node'][1])))
                    terms.append(((kz, i, j, net), e['name']))
        for v in M['vias']:                       # a part's post on this net: its foot on the sheet is a terminal
            if v.get('post') and v['net'] == net:
                k = int(np.argmin(np.abs(zs - v['z0']))) if any(abs(v['z0'] - q) < 1e-6 for q in sheets) else \
                    int(np.argmin(np.abs(zs - v['z1'])))
                i = int(np.where((xs >= v['x0'] - 1e-9) & (xs <= v['x1'] + 1e-9))[0][0])
                j = int(np.where((ys >= v['y0'] - 1e-9) & (ys <= v['y1'] + 1e-9))[0][0])
                terms.append(((k, i, j, net), v['post']))
        groups = defaultdict(list)
        for node, name in terms:
            parent.setdefault(node, node)
            groups[find(node)].append(name)
        if len(groups) > 1:
            fails.append('%s is in %d pieces on the mesh: %s' % (net, len(groups), ' | '.join(
                ', '.join(sorted(set(g))[:6]) for g in groups.values())))
        elif terms:
            info.append('%s: one piece on the mesh, %d terminals' % (net, len(terms)))
    # ---- size and cost
    s = M['stats']
    cells, dt = s['cells'], s['dt_s']
    exc = M.get('excitation', {})
    tend = tend or exc.get('tend_s') or (60e-9 if M.get('far_field') else 20e-9)
    steps = int(math.ceil(tend / dt))
    mem_gb = cells * 120 / 1e9               # E, H, coefficients and PML in float32: about 100-130 bytes a cell
    est = []
    for name, gcs, rate in GPUS:
        sec = cells * steps / (gcs * 1e9)
        est.append(dict(gpu=name, minutes=round(sec / 60, 1), usd=round(rate * (sec / 3600 + BOOT_MIN / 60), 2)))
    info.append('%.1f M cells, dt %.1f fs, %d steps for %.1f ns, about %.1f GB of GPU memory' % (
        cells / 1e6, dt * 1e15, steps, tend * 1e9, mem_gb))
    for e in est:
        info.append('  %-20s %6.1f min  $%.2f' % (e['gpu'], e['minutes'], e['usd']))
    if png:
        plot(M, owner_maps, X, Y, png)
    return dict(fails=fails, warnings=warns, info=info, steps=steps, tend_s=tend, cells=cells, mem_gb=mem_gb,
                estimates=est)


def plot(M, owner_maps, X, Y, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {round(v, 6): k for k, v in M['z'].items()}
    zs = sorted(owner_maps, reverse=True)
    fig, axs = plt.subplots(len(zs), 1, figsize=(14, 2.2 * len(zs) * max(0.4, (Y[-1] - Y[0]) / (X[-1] - X[0]) * 5)))
    axs = np.atleast_1d(axs)
    nets = sorted({n for z in zs for n in np.unique(owner_maps[z][0]) if n})
    detail = set(M['nets'])
    cmap = {n: (0.85, 0.85, 0.85) for n in nets}
    pal = plt.get_cmap('tab10')
    for k, n in enumerate(sorted(detail)):
        cmap[n] = pal(k % 10)[:3]
    for ax, z in zip(axs, zs):
        ox = owner_maps[z][0]
        img = np.ones(ox.shape + (3,))
        for n in nets:
            img[ox == n] = cmap[n]
        ax.imshow(img.transpose(1, 0, 2), origin='upper', extent=[X[0], X[-1], Y[-1], Y[0]], aspect='equal',
                  interpolation='nearest')
        ax.set_title('%s on the mesh (x-edge owners): detailed nets in colour, other copper grey' % names.get(z, z),
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--tend', type=float, help='simulated time (s) for the estimate')
    ap.add_argument('--png', action='store_true', help='write <model>_mesh.png: each sheet as the mesh sees it')
    a = ap.parse_args()
    M = json.load(open(a.model))
    png = os.path.splitext(a.model)[0] + '_mesh.png' if a.png else None
    r = check(M, a.tend, png)
    out = os.path.splitext(a.model)[0] + '_check.json'
    json.dump(r, open(out, 'w'), indent=1)
    for k in ('info', 'warnings', 'fails'):
        for line in r[k]:
            print('%-5s %s' % ({'info': '', 'warnings': 'WARN', 'fails': 'FAIL'}[k], line))
    print('check %s -> %s' % ('FAILED' if r['fails'] else 'passed', out))
    sys.exit(1 if r['fails'] else 0)


if __name__ == '__main__':
    main()
