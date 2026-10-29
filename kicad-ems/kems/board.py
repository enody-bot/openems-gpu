"""Reads a KiCad board (.kicad_pcb, KiCad 8 / 9 / 10 / 10.99) directly, without pcbnew, into plain Python data.

    from kems.board import Board
    B = Board('my.kicad_pcb')

Units are KiCad page millimetres (x right, y down). What it reads:
  stack       the physical stackup top to bottom: copper, dielectric (each sub-layer of a dielectric on its own: KiCad
              writes extra sub-layers after `addsublayer`, which a reader that takes the first thickness drops) and mask
  z           per copper layer (z_bottom, z_top) in mm above the bottom face of the board
  outline     the board outline ring from Edge.Cuts (the largest closed ring), and `cutouts` (the other rings)
  tracks      straight segments {net, layer, a, b, w}; arc tracks are split into short segments
  vias        {net, at, size, drill, layers (every copper layer the barrel spans)}
  pads        {ref, num, net, at, rot, size, shape, kind, layers, drill, poly}; poly is the copper outline
  fills       zone fills as saved {net, layer, ring}; KiCad stores holes as zero-width slits in the ring
  footprints  {ref, value, at, rot, side, lib, pads (indices), jumper_groups}
Zone fills are read as saved: refill the board in KiCad (or `kicad-cli pcb drc --refill-zones` on a copy) first.
"""
import math
import re

from .sexpr import QStr, kv, kvs, parse

GND_RE = re.compile(r'(^|[/_\-.])(A|D|P|S|C)?GND\w*$|(^|[/_\-.])(VSS\w*|0V|GROUND)$', re.I)
POWER_RE = re.compile(r'^(\+\S+|-\d\S*|V?\d+V\d*\w{0,4}|(A|D|IO|P)?V(CC|DD|EE|PP|SS)\w{0,8}|VBUS\w{0,4}|VIN\w{0,4}|'
                      r'VOUT\w{0,4}|VBAT\w{0,4}|VSYS|VLED|VCORE|VREG\w{0,4}|V_?\d\w{0,6})$', re.I)
CONTROL = re.compile(r'(^|_)(EN|ENABLE|SENSE|SNS|FB|PG|PGOOD|CTRL|HO|LO|BOOT|BST|COMP|RT|SNUB|TRIM|CNFG|CFG|ISNS|IMON|'
                     r'SS|ILIM|UVLO|OVP|PWM|FILT|DRV|GATE|SHDN|FAULT|INT|DET|ID|SW|LX|ADC|MON|SET|SEL)(_|$)', re.I)


def is_ground(net):
    return bool(net) and bool(GND_RE.search(net))


def is_power(net):
    """A rail by its name: the last path component is a rail name (+3V3, VCC_IO, VBUS) with no control word in it
    (VLED_BUCK_EN, VIN_SENSE are signals)."""
    n = short(net)
    return bool(net) and not is_ground(net) and bool(POWER_RE.match(n)) and not CONTROL.search(n)


def short(net):
    return (net or '').split('/')[-1] or '(none)'


def rot(lx, ly, deg):
    """KiCad's rotation of a local point (y down, positive angle counter-clockwise on screen)."""
    t = math.radians(deg)
    return lx * math.cos(t) + ly * math.sin(t), -lx * math.sin(t) + ly * math.cos(t)


def ring_area(ring):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        a += x1 * y2 - x2 * y1
    return a / 2


def arc_points(s, m, e, step_deg=10.0):
    """Points along the circular arc start -> mid -> end (inclusive)."""
    (x1, y1), (x2, y2), (x3, y3) = s, m, e
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [s, e]
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(p[1] - uy, p[0] - ux) for p in (s, m, e))

    def ccw(a, b):
        return (b - a) % (2 * math.pi)
    span = ccw(a1, a3)
    if ccw(a1, a2) > span:          # the mid point is on the other way round
        span = span - 2 * math.pi
    n = max(2, int(math.ceil(abs(math.degrees(span)) / step_deg)))
    return [(ux + r * math.cos(a1 + span * k / n), uy + r * math.sin(a1 + span * k / n)) for k in range(n + 1)]


def _xy(node):
    return float(node[1]), float(node[2])


def _pts(node):
    """(pts (xy ..) (arc (start ..) (mid ..) (end ..)) ..) -> [(x, y)]."""
    out = []
    for p in node[1:]:
        if not isinstance(p, list):
            continue
        if p[0] == 'xy':
            out.append(_xy(p))
        elif p[0] == 'arc':
            pa = arc_points(_xy(kv(p, 'start')), _xy(kv(p, 'mid')), _xy(kv(p, 'end')))
            out += pa if not out else pa[1:]
    return out


class Board:
    def __init__(self, path):
        self.path = path
        root = parse(open(path, encoding='utf-8').read())[0]
        self.root = root
        self.version = int(kv(root, 'version')[1]) if kv(root, 'version') else 0
        self._net_table = {}
        for n in kvs(root, 'net'):
            if len(n) >= 3:
                self._net_table[str(n[1])] = str(n[2])
        self._read_layers()
        self._read_stack()
        self.tracks, self.vias, self.pads, self.fills, self.footprints = [], [], [], [], []
        edges = []
        for item in root[1:]:
            if not isinstance(item, list):
                continue
            tag = item[0]
            if tag == 'segment':
                self._segment(item)
            elif tag == 'arc':
                self._arc(item)
            elif tag == 'via':
                self._via(item)
            elif tag == 'footprint':
                self._footprint(item, edges)
            elif tag == 'zone':
                self._zone(item)
            elif tag.startswith('gr_') and self._layer(item) == 'Edge.Cuts':
                edges += self._edge_item(item, lambda p: p)
        self._outline(edges)
        self.nets = sorted({t['net'] for t in self.tracks} | {p['net'] for p in self.pads} |
                           {v['net'] for v in self.vias} | {f['net'] for f in self.fills} - {''})
        self.nets = [n for n in self.nets if n]

    # ------------------------------------------------------------------ layers and stackup
    def _read_layers(self):
        L = kv(self.root, 'layers')
        names = []
        for e in (L[1:] if L else []):
            if isinstance(e, list) and len(e) >= 3 and str(e[1]).endswith('.Cu'):
                names.append(str(e[1]))
        inner = sorted([n for n in names if n.startswith('In')], key=lambda n: int(re.findall(r'\d+', n)[0]))
        self.copper = [n for n in ['F.Cu'] if n in names] + inner + [n for n in ['B.Cu'] if n in names]

    def _read_stack(self):
        setup = kv(self.root, 'setup')
        st = kv(setup, 'stackup') if setup else None
        thick = float(kv(kv(self.root, 'general'), 'thickness')[1]) if kv(self.root, 'general') and \
            kv(kv(self.root, 'general'), 'thickness') else 1.6
        stack = []
        if st is not None:
            for lay in kvs(st, 'layer'):
                name = str(lay[1])
                typ = str(kv(lay, 'type')[1]) if kv(lay, 'type') else ''
                if name.endswith('.Cu'):
                    stack.append(dict(name=name, kind='copper', t=self._num(lay, 'thickness', 0.035)))
                elif 'Mask' in name:
                    stack.append(dict(name=name, kind='mask', t=self._num(lay, 'thickness', 0.01),
                                      er=self._num(lay, 'epsilon_r', 3.5), tand=self._num(lay, 'loss_tangent', 0.02)))
                elif name.startswith('dielectric') or typ in ('core', 'prepreg'):
                    # a dielectric with sub-layers repeats thickness / material / epsilon_r / loss_tangent after each
                    # `addsublayer` atom: walk the children in order and start a new sub-layer at every thickness
                    subs, cur = [], None
                    for c in lay[2:]:
                        if isinstance(c, list) and c and c[0] == 'thickness':
                            cur = dict(name='%s.%d' % (name, len(subs) + 1), kind='dielectric', t=float(c[1]),
                                       er=4.5, tand=0.02, material='', role=typ)
                            subs.append(cur)
                        elif isinstance(c, list) and c and cur is not None:
                            if c[0] == 'epsilon_r':
                                cur['er'] = float(c[1])
                            elif c[0] == 'loss_tangent':
                                cur['tand'] = float(c[1])
                            elif c[0] == 'material':
                                cur['material'] = str(c[1])
                    stack += subs
        if not any(s['kind'] == 'copper' for s in stack):           # no stackup: spread the copper evenly
            n = len(self.copper)
            d = (thick - 0.035 * n) / max(n - 1, 1)
            for i, c in enumerate(self.copper):
                stack.append(dict(name=c, kind='copper', t=0.035))
                if i < n - 1:
                    stack.append(dict(name='dielectric %d.1' % (i + 1), kind='dielectric', t=d, er=4.5, tand=0.02,
                                      material='assumed', role='assumed'))
            self.stack_assumed = True
        else:
            self.stack_assumed = False
        self.stack = stack
        # heights above the bottom face, skipping the masks (they are left out of the models)
        z, self.z = 0.0, {}
        for s in reversed([s for s in stack if s['kind'] != 'mask']):
            s['z0'], s['z1'] = z, z + s['t']
            if s['kind'] == 'copper':
                self.z[s['name']] = (z, z + s['t'])
            z += s['t']
        self.thickness = z

    @staticmethod
    def _num(node, key, default):
        c = kv(node, key)
        try:
            return float(c[1]) if c else default
        except ValueError:
            return default

    def dielectrics(self):
        return [s for s in self.stack if s['kind'] == 'dielectric']

    def neighbours(self, layer):
        """The copper layers directly above and below `layer` in the stack (None at the outside)."""
        i = self.copper.index(layer)
        return (self.copper[i - 1] if i > 0 else None, self.copper[i + 1] if i + 1 < len(self.copper) else None)

    def gap(self, a, b):
        """Dielectric height (mm) between two copper layers."""
        za, zb = self.z[a], self.z[b]
        return max(za[0], zb[0]) - min(za[1], zb[1])

    # ------------------------------------------------------------------ items
    def _net(self, node):
        n = kv(node, 'net_name')
        if n is not None and len(n) > 1:
            return str(n[1])
        n = kv(node, 'net')
        if n is None or len(n) < 2:
            return ''
        if len(n) >= 3:
            return str(n[2])
        if isinstance(n[1], QStr):
            return str(n[1])
        return self._net_table.get(str(n[1]), '')

    @staticmethod
    def _layer(node):
        L = kv(node, 'layer')
        return str(L[1]) if L else None

    def _cu_layers(self, node):
        L = kv(node, 'layers') or kv(node, 'layer')
        out = []
        for name in (L[1:] if L else []):
            name = str(name)
            if name == '*.Cu':
                out += self.copper
            elif name == 'F&B.Cu':
                out += [c for c in ('F.Cu', 'B.Cu') if c in self.copper]
            elif name in self.copper:
                out.append(name)
        return out

    def _segment(self, item):
        s, e = kv(item, 'start'), kv(item, 'end')
        lay = self._layer(item)
        if lay not in self.copper:
            return
        self.tracks.append(dict(net=self._net(item), layer=lay, a=_xy(s), b=_xy(e), w=float(kv(item, 'width')[1])))

    def _arc(self, item):
        lay = self._layer(item)
        if lay not in self.copper:
            return
        pts = arc_points(_xy(kv(item, 'start')), _xy(kv(item, 'mid')), _xy(kv(item, 'end')))
        w, net = float(kv(item, 'width')[1]), self._net(item)
        for a, b in zip(pts[:-1], pts[1:]):
            self.tracks.append(dict(net=net, layer=lay, a=a, b=b, w=w, arc=True))

    def _via(self, item):
        L = [str(x) for x in kv(item, 'layers')[1:]]
        ends = [l for l in L if l in self.copper]
        if len(ends) < 2:
            ends = [self.copper[0], self.copper[-1]]
        i, j = sorted(self.copper.index(l) for l in ends[:2])
        self.vias.append(dict(net=self._net(item), at=_xy(kv(item, 'at')), size=float(kv(item, 'size')[1]),
                              drill=float(kv(item, 'drill')[1]), layers=self.copper[i:j + 1]))

    def _zone(self, item):
        net = self._net(item)
        for f in kvs(item, 'filled_polygon'):
            lay = self._layer(f)
            if lay not in self.copper:
                continue
            ring = _pts(kv(f, 'pts'))
            if len(ring) >= 3:
                self.fills.append(dict(net=net, layer=lay, ring=ring))

    def _footprint(self, item, edges):
        props = {str(p[1]): str(p[2]) for p in kvs(item, 'property') if len(p) > 2}
        ref = props.get('Reference', '')
        for t in kvs(item, 'fp_text'):                    # KiCad 6 / 7
            if len(t) > 2 and str(t[1]) == 'reference':
                ref = str(t[2])
            if len(t) > 2 and str(t[1]) == 'value':
                props.setdefault('Value', str(t[2]))
        at = kv(item, 'at')
        if at is not None:
            fx, fy = _xy(at)
            fr = float(at[3]) if len(at) > 3 else 0.0
        else:                                             # KiCad 10.99: (transform (translate x y) (rotate a))
            tf = kv(item, 'transform')
            tr, ro = kv(tf, 'translate'), kv(tf, 'rotate')
            fx, fy = _xy(tr)
            fr = float(ro[1]) if ro else 0.0
        side = 'B' if self._layer(item) == 'B.Cu' else 'F'

        def place(p):
            dx, dy = rot(p[0], p[1], fr)
            return (fx + dx, fy + dy)
        idx = []
        for pd in kvs(item, 'pad'):
            pa = kv(pd, 'at')
            lx, ly = float(pa[1]), float(pa[2])
            prot = float(pa[3]) if len(pa) > 3 else 0.0      # absolute: it includes the footprint's rotation
            sz = kv(pd, 'size')
            size = (float(sz[1]), float(sz[2]))
            dr = kv(pd, 'drill')
            drill = 0.0
            if dr is not None:
                nums = [x for x in dr[1:] if not isinstance(x, list) and x != 'oval']
                drill = float(nums[0]) if nums else 0.0
            shape = str(pd[3]) if len(pd) > 3 else 'rect'
            pf = kv(pd, 'pinfunction')
            pad = dict(ref=ref, num=str(pd[1]), net=self._net(pd), at=place((lx, ly)), rot=prot, size=size,
                       shape=shape, kind=str(pd[2]), layers=self._cu_layers(pd), drill=drill,
                       pinfunction=str(pf[1]) if pf else '')
            pad['poly'] = pad_poly(pad, kv(pd, 'primitives'))
            idx.append(len(self.pads))
            self.pads.append(pad)
        groups = []
        jg = kv(item, 'jumper_pad_groups')
        if jg is not None:
            for g in jg[1:]:
                if isinstance(g, list):
                    groups.append([str(x) for x in g])
        lib = str(item[1]) if len(item) > 1 else ''
        self.footprints.append(dict(ref=ref, value=props.get('Value', ''), at=(fx, fy), rot=fr, side=side, lib=lib,
                                    pads=idx, jumper_groups=groups))
        for g in item[1:]:
            if isinstance(g, list) and g and g[0].startswith('fp_') and self._layer(g) == 'Edge.Cuts':
                edges += self._edge_item(g, place)

    # ------------------------------------------------------------------ outline
    def _edge_item(self, g, place):
        """Edge.Cuts item -> [('ring', pts) | ('open', pts)] in page coordinates."""
        tag = g[0][3:]
        if tag == 'line':
            return [('open', [place(_xy(kv(g, 'start'))), place(_xy(kv(g, 'end')))])]
        if tag == 'arc':
            if kv(g, 'mid') is not None:
                pts = arc_points(_xy(kv(g, 'start')), _xy(kv(g, 'mid')), _xy(kv(g, 'end')), 5.0)
            else:                                         # KiCad 5: start = centre, end = start point, angle
                c, s0, ang = _xy(kv(g, 'start')), _xy(kv(g, 'end')), float(kv(g, 'angle')[1])
                r = math.hypot(s0[0] - c[0], s0[1] - c[1])
                a0 = math.atan2(s0[1] - c[1], s0[0] - c[0])
                n = max(2, int(abs(ang) / 5))
                pts = [(c[0] + r * math.cos(a0 + math.radians(ang) * k / n),
                        c[1] + r * math.sin(a0 + math.radians(ang) * k / n)) for k in range(n + 1)]
            return [('open', [place(p) for p in pts])]
        if tag == 'rect':
            (x0, y0), (x1, y1) = _xy(kv(g, 'start')), _xy(kv(g, 'end'))
            return [('ring', [place(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])]
        if tag == 'circle':
            c, e = _xy(kv(g, 'center')), _xy(kv(g, 'end'))
            r = math.hypot(e[0] - c[0], e[1] - c[1])
            return [('ring', [place((c[0] + r * math.cos(2 * math.pi * k / 72), c[1] + r * math.sin(2 * math.pi * k / 72)))
                              for k in range(72)])]
        if tag == 'poly':
            return [('ring', [place(p) for p in _pts(kv(g, 'pts'))])]
        return []

    def _outline(self, edges):
        rings = [pts for kind, pts in edges if kind == 'ring' and len(pts) >= 3]
        opens = [list(pts) for kind, pts in edges if kind == 'open']
        tol = 0.01

        def close(a, b):
            return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
        while opens:
            chain = opens.pop(0)
            grown = True
            while grown and not close(chain[0], chain[-1]):
                grown = False
                for i, o in enumerate(opens):
                    if close(chain[-1], o[0]):
                        chain += o[1:]
                    elif close(chain[-1], o[-1]):
                        chain += o[::-1][1:]
                    elif close(chain[0], o[-1]):
                        chain = o[:-1] + chain
                    elif close(chain[0], o[0]):
                        chain = o[::-1][:-1] + chain
                    else:
                        continue
                    opens.pop(i)
                    grown = True
                    break
            if len(chain) >= 4 and close(chain[0], chain[-1]):
                rings.append(chain[:-1])
        if not rings:
            xs = [p for t in self.tracks for p in (t['a'][0], t['b'][0])] + [p['at'][0] for p in self.pads]
            ys = [p for t in self.tracks for p in (t['a'][1], t['b'][1])] + [p['at'][1] for p in self.pads]
            rings = [[(min(xs) - 1, min(ys) - 1), (max(xs) + 1, min(ys) - 1), (max(xs) + 1, max(ys) + 1),
                      (min(xs) - 1, max(ys) + 1)]]
            self.outline_assumed = True
        else:
            self.outline_assumed = False
        rings.sort(key=lambda r: -abs(ring_area(r)))
        self.outline, self.cutouts = rings[0], rings[1:]
        xs, ys = [p[0] for p in self.outline], [p[1] for p in self.outline]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))

    # ------------------------------------------------------------------ helpers
    def footprint(self, ref):
        for f in self.footprints:
            if f['ref'] == ref:
                return f
        raise KeyError(ref)

    def pad(self, ref, num):
        for i in self.footprint(ref)['pads']:
            if self.pads[i]['num'] == str(num):
                return self.pads[i]
        raise KeyError('%s.%s' % (ref, num))

    def net_pads(self, net):
        return [p for p in self.pads if p['net'] == net]


def pad_poly(p, primitives=None, n_round=16):
    """Copper outline of a pad in page coordinates (round-rect corners as square: a few um of copper)."""
    hx, hy = p['size'][0] / 2, p['size'][1] / 2
    shape = p['shape']
    if shape == 'circle':
        loc = [(hx * math.cos(2 * math.pi * k / n_round), hx * math.sin(2 * math.pi * k / n_round))
               for k in range(n_round)]
    elif shape == 'oval':
        r = min(hx, hy)
        cx, cy = hx - r, hy - r
        loc = []
        for k in range(n_round):
            a = 2 * math.pi * k / n_round
            ox = cx if math.cos(a) >= 0 else -cx
            oy = cy if math.sin(a) >= 0 else -cy
            loc.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    else:
        loc = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    if shape == 'custom' and primitives is not None:
        for g in kvs(primitives, 'gr_poly'):
            pts = _pts(kv(g, 'pts'))
            if len(pts) >= 3 and abs(ring_area(pts)) > abs(ring_area(loc)):
                loc = pts                     # the largest primitive stands for the pad
    out = []
    for lx, ly in loc:
        dx, dy = rot(lx, ly, p['rot'])
        out.append((p['at'][0] + dx, p['at'][1] + dy))
    return out
