"""Step 4 (on the GPU): run one model with openEMS and write its results. Needs only numpy and h5py besides openEMS,
so it runs in the openems-gpu image as is.

    python fdtd_run.py --model models/NAME.json --sim /scratch/sim --out out/NAME [--engine gpu]
                       [--nrts N | --tend 8e-9] [--edge 0.8e-9] [--td-planes] [--no-nf2ff] [--csx-only]

Excitation: the model's (`excitation`): a Gaussian pulse over 0..fmax (spectra, far field), or a raised-cosine step
with a given 10-90 % rise (--edge, for waveforms and the field animation). The run ends at --tend / --nrts, or at
-40 dB energy on openEMS' exact schedule (once per excitation Nyquist period), so a run stops at the same step on every
GPU. For emissions use a fixed length: the energy swings +-5 dB and the -40 dB test can fire on a dip.

Outputs in --out:
  run.json        engine line (check it says "GPU, backend: HIP": openEMS falls back to the CPU silently), cells,
                  steps, speed, final energy, wall times, the energy history; "unstable": true when it grew
  traces.npz      t (s) and every probe voltage / port voltage and current, as openEMS wrote them
  spectra.npz     freqs; Vsrc = U + R I of the excited port (the Thevenin source); every probe's V(f) and V / Vsrc
  farfield.npz    with a far-field box: E_theta / E_phi at 3 m (complex [f, theta, phi]), also per volt of source
  fields_<p>.npz  with --td-planes: |E| or |H tangential| on each plane as float16 [frame, y, x] / peak, peak, times, grid lines
UNSTABLE.txt appears when the watchdog stopped a run whose energy grew past 30 x its peak during the pulse."""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

GROWTH_KILL = 30.0


def safe(s):
    return re.sub(r'[^A-Za-z0-9_]+', '_', s).strip('_') or 'NONE'


def build(M, FDTD=None, td_planes=False):
    from CSXCAD import ContinuousStructure
    CSX = ContinuousStructure()
    if FDTD is not None:
        FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)
    for a in 'xyz':
        grid.AddLine(a, M['mesh'][a])
    for d in M['dielectrics']:
        mat = CSX.AddMaterial(d['name'], epsilon=d['er'], kappa=d['kappa'])
        pts = np.array(d['ring'], float).T
        mat.AddLinPoly([pts[0], pts[1]], 'z', d['z0'], d['z1'] - d['z0'], priority=d['prio'])
    props = {}

    def metal(net):
        k = safe(net)
        if k not in props:
            props[k] = CSX.AddMetal('PEC_' + k)
        return props[k]
    n_poly = 0
    for net, polys in M['metals'].items():
        m = metal(net)
        for p in polys:
            pts = np.array(p['ring'], float).T
            m.AddPolygon([pts[0], pts[1]], 'z', p['z'], priority=p['prio'])
            n_poly += 1
    for v in M['vias']:
        metal(v['net']).AddBox([v['x0'], v['y0'], v['z0']], [v['x1'], v['y1'], v['z1']], priority=v['prio'])
    if M.get('holes'):
        mat = CSX.AddMaterial('antipad', epsilon=M['holes'][0]['er'], kappa=M['holes'][0]['kappa'])
        for h in M['holes']:
            mat.AddBox(h['start'], h['stop'], priority=h['prio'])
    for L in M['lumped']:
        kw = dict(ny=L['dir'], caps=True)
        for k in ('R', 'C', 'L'):
            if L.get(k) is not None:
                kw[k] = L[k]
        CSX.AddLumpedElement(L['name'], **kw).AddBox(L['start'], L['stop'], priority=50)
    ports = []
    if FDTD is not None:
        for p in M['ports']:
            ports.append(FDTD.AddLumpedPort(p['nr'], p['R'], p['start'], p['stop'], p['dir'], excite=float(p['excite']),
                                            priority=60))
    for pr in M['probes']:
        CSX.AddProbe(pr['name'], p_type=0).AddBox(pr['start'], pr['stop'])
    if td_planes:
        x0, y0, x1, y1 = M['region']
        for P in M.get('td_planes', []):
            d = CSX.AddDump('td_' + P['name'], dump_type=0 if P['kind'] == 'E' else 1, dump_mode=1, file_type=1,
                            opt_resolution=[P.get('res_mm', 0.1), P.get('res_mm', 0.1), 0.02])
            d.AddBox([x0, y0, P['z']], [x1, y1, P['z']])
    return CSX, ports, n_poly


def convert_td_planes(sim, out, meta):
    """openEMS' HDF5 time series ([component, x, y, z], NXYZ) -> fields_<name>.npz: |E| (or |H tangential|) as
    float16 [frame, y, x], times (s), the dump's x / y lines (m), the per-frame maxima."""
    import h5py
    meta['td_planes'] = []
    for f in sorted(glob.glob(os.path.join(sim, 'td_*.h5'))):
        name = os.path.basename(f)[3:-3]
        with h5py.File(f, 'r') as h:
            x, y = np.array(h['/Mesh/x']), np.array(h['/Mesh/y'])
            td = h['/FieldData/TD']
            keys = sorted(td.keys(), key=lambda k: float(td[k].attrs['time'][0]) if 'time' in td[k].attrs else int(k))
            times, frames = [], []
            for k in keys:
                a = np.array(td[k], dtype=np.float32)[:, :, :, 0]
                tang = a[:2] if name.startswith('H') else a
                frames.append(np.sqrt((tang ** 2).sum(0)).T)
                times.append(float(td[k].attrs['time'][0]) if 'time' in td[k].attrs else float('nan'))
        arr = np.stack(frames)
        peak = float(arr.max()) or 1.0                 # float16 of raw A/m underflows (6e-8): store field / peak
        np.savez_compressed(os.path.join(out, 'fields_%s.npz' % name), mag=(arr / peak).astype(np.float16), peak=peak,
                            t=np.array(times), x=x, y=y, fmax=arr.reshape(len(arr), -1).max(1).astype(np.float32))
        meta['td_planes'].append(dict(name=name, frames=len(arr), shape=list(arr.shape), peak=peak))


def read_probe(path):
    d = np.loadtxt(path, comments='%')
    d = np.atleast_2d(d)
    return d[:, 0], d[:, 1]


def dft(t, v, freqs):
    """openEMS' convention (utilities.DFT_time2freq): 2 dt sum v e^-jwt."""
    dt = t[1] - t[0]
    out = np.empty(len(freqs), complex)
    for k in range(0, len(freqs), 64):
        f = np.asarray(freqs[k:k + 64])[:, None]
        out[k:k + 64] = 2 * dt * (v[None, :] * np.exp(-2j * np.pi * f * t[None, :])).sum(1)
    return out


def energy_history(txt):
    out = []
    for st, e in re.findall(r'Timestep: *(\d+) \|\|.*?Energy: *~ *([0-9.e+-]+|inf|nan|-nan)', txt):
        out.append((int(st), float('inf') if 'n' in e else float(e)))
    return out


def watchdog(logfile, note, stop):
    """Kills the run when the field energy grows after the excitation: a stable run only decays once the pulse is in,
    so energy above GROWTH_KILL x its peak during the pulse is a numerical instability."""
    import signal
    while not stop.wait(20):
        try:
            txt = open(logfile, errors='replace').read()
        except OSError:
            continue
        hist = energy_history(txt)
        bad = [(st, e) for st, e in hist if e == float('inf')]
        if bad:                                    # NaN or inf: already diverged, during the pulse or not
            with open(note, 'w') as f:
                f.write('numerically unstable: energy not finite at step %d\n' % bad[0][0])
            os.kill(os.getpid(), signal.SIGKILL)
        m = re.search(r'Excitation signal length is: (\d+) timesteps', txt)
        if not m:
            continue
        settle = 2 * int(m.group(1))
        peak = max([e for st, e in hist if st <= settle] or [0.0])
        late = [(st, e) for st, e in hist if st > settle]
        if peak > 0 and late and late[-1][1] > GROWTH_KILL * peak:
            with open(note, 'w') as f:
                f.write('numerically unstable: energy %.3g at step %d against %.3g during the pulse\n' % (
                    late[-1][1], late[-1][0], peak))
            os.kill(os.getpid(), signal.SIGKILL)


def run_fdtd(FDTD, sim, engine, logfile):
    import threading
    os.makedirs(sim, exist_ok=True)
    fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    s1, s2 = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    t0 = time.time()
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(logfile, os.path.join(os.path.dirname(logfile), 'UNSTABLE.txt'), stop),
                     daemon=True).start()
    try:
        kw = dict(cleanup=True, engine=engine)
        try:
            FDTD.Run(sim, exact_endcriteria=True, **kw)
        except TypeError:                          # an openEMS without the exact end-criteria schedule
            FDTD.Run(sim, **kw)
    finally:
        stop.set()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            import ctypes
            import ctypes.util
            ctypes.CDLL(ctypes.util.find_library('c')).fflush(None)
        except Exception:
            pass
        os.dup2(s1, 1)
        os.dup2(s2, 2)
        os.close(fd)
    return time.time() - t0


def postprocess(M, sim, out, nf2ff, meta):
    freqs = np.array(M['freqs'])
    tr = {}
    for f in sorted(glob.glob(os.path.join(sim, 'V_*'))) + sorted(glob.glob(os.path.join(sim, 'port_*t*'))):
        name = os.path.basename(f)
        if '.' in name:
            continue
        t, v = read_probe(f)
        tr[name] = v
        tr['t'] = t
    np.savez_compressed(os.path.join(out, 'traces.npz'), **tr)
    exc = [p for p in M['ports'] if p['excite']]
    spec = dict(freqs=freqs)
    if exc and 't' in tr:
        k = exc[0]['nr']
        ut, it = ('port_ut_%d' % k, 'port_it_%d' % k) if 'port_ut_%d' % k in tr else ('port_ut%d' % k, 'port_it%d' % k)
        t = tr['t']
        U, I = dft(t, tr[ut], freqs), dft(t, tr[it], freqs)
        spec['U1'], spec['I1'], spec['Vsrc'] = U, I, U + exc[0]['R'] * I
        for name, v in tr.items():
            if name.startswith('V_'):
                spec[name] = dft(t, v, freqs)
                spec['H_' + name] = spec[name] / spec['Vsrc']
        meta['source_port'] = dict(nr=k, R=exc[0]['R'], u=ut, i=it)
    np.savez_compressed(os.path.join(out, 'spectra.npz'), **spec)
    if nf2ff is not None:
        theta, phi = np.arange(0, 181, 10.0), np.arange(0, 360, 10.0)
        t0 = time.time()
        res = nf2ff.CalcNF2FF(sim, freqs, theta, phi, radius=3.0, center=M['nf2ff_center_m'], read_cached=False)
        meta['nf2ff_s'] = time.time() - t0
        ff = dict(freqs=freqs, theta=theta, phi=phi, r=3.0, E_theta=np.array(res.E_theta).astype(np.complex64),
                  E_phi=np.array(res.E_phi).astype(np.complex64), Prad=np.array(res.Prad))
        if 'Vsrc' in spec:
            ff['E_theta_per_V'] = (np.array(res.E_theta) / spec['Vsrc'][:, None, None]).astype(np.complex64)
            ff['E_phi_per_V'] = (np.array(res.E_phi) / spec['Vsrc'][:, None, None]).astype(np.complex64)
        np.savez_compressed(os.path.join(out, 'farfield.npz'), **ff)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--engine', default='gpu')
    ap.add_argument('--nrts', type=int, default=0)
    ap.add_argument('--tend', type=float, default=0.0, help='simulated time (s); overrides the model')
    ap.add_argument('--end', type=float, default=1e-4, help='energy end criterion (1e-4 = -40 dB)')
    ap.add_argument('--edge', type=float, default=None, help='raised-cosine step with this 10-90 %% rise (s)')
    ap.add_argument('--oversample', type=int, default=6, help='dumps and probe samples per Nyquist interval')
    ap.add_argument('--td-planes', action='store_true', help="the model's time-domain field planes (for rendering)")
    ap.add_argument('--no-nf2ff', action='store_true')
    ap.add_argument('--csx-only', action='store_true', help='build the CSX structure and write its XML; no run')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sim, out = os.path.abspath(a.sim), os.path.abspath(a.out)
    meta = dict(argv=sys.argv, start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    M = json.load(open(a.model))
    if a.csx_only:
        CSX, _, n = build(M, td_planes=a.td_planes)
        CSX.Write2XML(os.path.join(out, 'csx_only.xml'))
        print('CSX built: %d polygons, %d vias, %d lumped, %d probes' % (n, len(M['vias']), len(M['lumped']),
                                                                       len(M['probes'])))
        return
    from openEMS import openEMS
    exc = dict(M.get('excitation', {'kind': 'gauss'}))
    if a.edge is not None:
        exc = dict(kind='edge', rise_s=a.edge, tend_s=exc.get('tend_s', 8e-9))
    tend = a.tend or exc.get('tend_s', 0.0)
    nrts = a.nrts or (int(math.ceil(tend / M['stats']['dt_s'])) if tend else 1000000)
    FDTD = openEMS(NrTS=nrts, EndCriteria=a.end if not tend else 0.0)
    fmax = M.get('fmax_hz', 3e9)
    if exc['kind'] == 'edge':
        T, t0 = exc['rise_s'] / 0.5903, 0.1e-9            # raised cosine: 10-90 % = 0.5903 of its length
        FDTD.SetCustomExcite('if(t<%g,0,if(t<%g,0.5*(1-cos(pi*(t-%g)/%g)),1))' % (t0, t0 + T, t0, T), fmax, fmax)
        meta['excitation'] = dict(kind='raised-cosine step', rise_10_90_s=exc['rise_s'], length_s=T, delay_s=t0)
    else:
        FDTD.SetGaussExcite(M['f0'], M['fc'])
        meta['excitation'] = dict(kind='gauss', f0=M['f0'], fc=M['fc'])
    FDTD.SetOverSampling(a.oversample)
    FDTD.SetBoundaryCond(['PML_8'] * 6)
    CSX, ports, n_poly = build(M, FDTD, td_planes=a.td_planes)
    nf2ff = None
    if M.get('far_field') and not a.no_nf2ff:
        nf2ff = FDTD.CreateNF2FFBox(frequency=list(M['freqs']), opt_resolution=[M.get('nf2ff_res', 2.0)] * 3)
    log = os.path.join(out, 'openems.log')
    meta.update(cells=M['stats']['cells'], nrts=nrts, tend_s=tend, nf2ff=nf2ff is not None)
    meta['wall_run_s'] = run_fdtd(FDTD, sim, a.engine, log)
    txt = open(log, errors='replace').read()
    m = re.findall(r'Create FDTD engine \(([^)]*\)?)\)', txt)
    meta['engine'] = m[-1] if m else None
    m = re.findall(r'Time for (\d+) iterations with ([\d.e+]+) cells : ([\d.]+) sec', txt)
    if m:
        meta['steps'], meta['solver_s'] = int(m[-1][0]), float(m[-1][2])
    m = re.findall(r'Speed: *([\d.]+) MCells/s', txt)
    if m:
        meta['mcells_per_s'] = float(m[-1])
    hist = energy_history(txt)
    meta['energy_history'] = hist[::max(1, len(hist) // 60)]
    meta['log_tail'] = txt.splitlines()[-30:]
    if any(e == float('inf') for st, e in hist) or os.path.exists(os.path.join(out, 'UNSTABLE.txt')):
        meta['unstable'] = True
        nf2ff = None
    postprocess(M, sim, out, nf2ff, meta)
    if a.td_planes:
        try:
            convert_td_planes(sim, out, meta)
        except Exception as e:                      # keep the raw dumps rather than lose the run
            meta['td_convert_error'] = repr(e)
            os.system('cp "%s"/td_*.h5 "%s"' % (sim, out))
    meta['end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(meta, open(os.path.join(out, 'run.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('energy_history', 'log_tail')}))


if __name__ == '__main__':
    main()
