"""Polygon helpers shared by the triage and the model builder (numpy + matplotlib.path, which the image also has).

Point tests use plain Path.contains_points. Do not use its `radius=` padding on KiCad fills: a fractured fill keeps its
holes as zero-width slits, the padding overlaps itself along a slit and points beside it read as outside."""
import math

import numpy as np
from matplotlib.path import Path


def track_poly(a, b, w):
    """Rectangle of a track segment with square caps of w / 2 (KiCad's caps are round: a few um^2 of copper)."""
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    h = w / 2
    if L < 1e-9:
        return [(ax - h, ay - h), (ax + h, ay - h), (ax + h, ay + h), (ax - h, ay + h)]
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny, ex, ey = -uy * h, ux * h, ux * h, uy * h
    return [(ax - ex + nx, ay - ey + ny), (bx + ex + nx, by + ey + ny), (bx + ex - nx, by + ey - ny),
            (ax - ex - nx, ay - ey - ny)]


def square(at, size):
    h = size / 2
    return [(at[0] - h, at[1] - h), (at[0] + h, at[1] - h), (at[0] + h, at[1] + h), (at[0] - h, at[1] + h)]


def clip_rect(ring, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a (possibly fractured, non-convex) ring against an axis-aligned rectangle."""
    def clip(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            P, Q = pts[i - 1], pts[i]
            ip, iq = inside(P), inside(Q)
            if iq:
                if not ip:
                    out.append(inter(P, Q))
                out.append(Q)
            elif ip:
                out.append(inter(P, Q))
        return out

    def ix(xc):
        return lambda P, Q: (xc, P[1] + (Q[1] - P[1]) * (xc - P[0]) / (Q[0] - P[0]))

    def iy(yc):
        return lambda P, Q: (P[0] + (Q[0] - P[0]) * (yc - P[1]) / (Q[1] - P[1]), yc)
    pts = list(ring)
    for inside, inter in ((lambda p: p[0] >= x0, ix(x0)), (lambda p: p[0] <= x1, ix(x1)),
                          (lambda p: p[1] >= y0, iy(y0)), (lambda p: p[1] <= y1, iy(y1))):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return pts


def bbox(ring):
    a = np.asarray(ring, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


class Shapes:
    """Polygons with a net and a bounding box, for vectorised 'which net is here' queries."""

    def __init__(self):
        self.items = []          # (net, Path, bbox, prio)

    def add(self, net, ring, prio=0):
        if len(ring) >= 3:
            self.items.append((net, Path(np.asarray(ring, float)), bbox(ring), prio))

    def nets_at(self, pts):
        """For an (N, 2) array: a list of N sets of the nets whose polygons hold each point."""
        pts = np.asarray(pts, float)
        out = [set() for _ in range(len(pts))]
        if not len(pts):
            return out
        for net, path, (x0, y0, x1, y1), _ in self.items:
            m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            if not m.any():
                continue
            idx = np.where(m)[0]
            hit = path.contains_points(pts[idx])
            for i in idx[hit]:
                out[i].add(net)
        return out

    def top_at(self, pts):
        """For an (N, 2) array: the net of the highest-priority polygon holding each point ('' if none), and whether
        two different nets tie at that priority."""
        pts = np.asarray(pts, float)
        best = np.full(len(pts), -1e9)
        net = np.array([''] * len(pts), dtype=object)
        tie = np.zeros(len(pts), bool)
        for n, path, (x0, y0, x1, y1), prio in self.items:
            m = (pts[:, 0] >= x0 - 1e-9) & (pts[:, 0] <= x1 + 1e-9) & (pts[:, 1] >= y0 - 1e-9) & (pts[:, 1] <= y1 + 1e-9)
            if not m.any():
                continue
            idx = np.where(m)[0]
            hit = idx[path.contains_points(pts[idx], radius=1e-9) | path.contains_points(pts[idx], radius=-1e-9)]
            if not len(hit):
                continue
            hi = hit[prio > best[hit]]
            eq = hit[(prio == best[hit]) & (net[hit] != n)]
            best[hi], net[hi], tie[hi] = prio, n, False
            tie[eq] = True
        return net, tie


def touch(A, B, tol=1e-4):
    """Do two copper outlines (rings, (N, 2) arrays) touch or overlap? A vertex of either inside (or on) the other,
    which is how KiCad joins copper: a thermal spoke's end vertices lie in the pad, a track's end cap in the fill or
    pad it ends on. Two outlines that only cross edge-to-edge without a vertex inside the other are missed."""
    A, B = np.asarray(A, float), np.asarray(B, float)

    def probes(R):
        """A small outline's vertices, edge midpoints and centre (a round-rect pad's corners can sit outside a fill
        that holds the pad); a large one's vertices."""
        if len(R) > 64:
            return R
        return np.vstack([R, (R + np.roll(R, -1, 0)) / 2, R.mean(0, keepdims=True)])
    ax0, ay0 = A.min(0)
    ax1, ay1 = A.max(0)
    bx0, by0 = B.min(0)
    bx1, by1 = B.max(0)
    if ax1 < bx0 - tol or bx1 < ax0 - tol or ay1 < by0 - tol or by1 < ay0 - tol:
        return False
    for P, Q, (x0, y0, x1, y1) in ((probes(A), B, (bx0, by0, bx1, by1)), (probes(B), A, (ax0, ay0, ax1, ay1))):
        m = (P[:, 0] >= x0 - tol) & (P[:, 0] <= x1 + tol) & (P[:, 1] >= y0 - tol) & (P[:, 1] <= y1 + tol)
        if not m.any():
            continue
        path = Path(Q)
        pts = P[m]
        if path.contains_points(pts, radius=tol).any() or path.contains_points(pts, radius=-tol).any():
            return True
    return False


def circle(at, r, n=8):
    return [(at[0] + r * math.cos(2 * math.pi * k / n), at[1] + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def samples_along(a, b, step):
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    n = max(1, int(math.ceil(L / step)))
    t = (np.arange(n) + 0.5) / n
    return np.stack([ax + t * (bx - ax), ay + t * (by - ay)], 1), L / n
