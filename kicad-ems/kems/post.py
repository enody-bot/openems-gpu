"""Step 5: turn a run's results into numbers, figures and a findings section for the report.

    python -m kems.post si  OUT_DIR MODEL.json [--swing 3.3] [--vil 0.3] [--vih 0.7] [--edge 0.8e-9]
    python -m kems.post emi OUT_DIR MODEL.json --source clock:12e6:3.3:0.5:0.8e-9 [--source switcher:400e3:24:0.2:3e-9]
    python -m kems.post pdn OUT_DIR MODEL.json [--target 0.05] [--fmin 1e6]

si   waveforms at every probe for a source swinging 0 -> --swing. An edge-excited run (fdtd_run.py --edge) is scaled
     by its source EMF (u + R i of the driven port: openEMS' lumped source depends on R, so the scale is per run); a
     Gaussian run is turned into the response to a raised-cosine edge of --edge through V(f) / Vsrc(f) inside the
     excited band. Per probe on the driven net: 50 % delay after the source, peak, minimum, the deepest dip back
     while the edge is inside the input band (vil..vih of the swing: a dip there can double-clock an edge-triggered
     input), the end value. Per probe on other nets: the peak crosstalk.
     Note: the FR-4 loss is a conductivity fixed at 1 GHz, which leaks at DC (receivers settle a little under
     the swing). Judge levels against the settled value, and edges by their shape.
emi  each source (kind:f:amplitude:duty:edge) as a trapezoid: at every harmonic n f, |c_n| = 2 A d |sinc(n d)| x
     min(1, 1 / (pi n tr f)) (the edge term as its envelope; 'switcher' takes the duty term as its envelope as well,
     its duty moves with load). The field is |c_n| |E / V|(f_n) in the worst direction (theta 0-180, phi 0-350 at
     10 deg; |E| from both polarisations, interpolated in dB between solver frequencies), as RMS (/ sqrt 2) at 3 m,
     +6 dB for the ground reflection of an OATS / semi-anechoic site. Limits: FCC 15.109 Class B 3 m (40 / 43.5 / 46
     dBuV/m from 30 / 88 / 216 MHz, 54 above 960 MHz) and CISPR 32 Class B scaled to 3 m (40.5 / 47.5 below / above
     230 MHz). Spread spectrum, the detector and the measurement antenna's height scan are not modelled: a margin
     under ~6 dB means "measure it", not "passes".
pdn  a rail's impedance seen from a load: a Gaussian run whose excited port sits at the load's power pad (study
     "excite": the rail, "drivers": {"U1.5": 1.0}); Z(f) = U / I of that port over the band, the decoupling as its ESL
     (no ESR: the resonances read high, which flags them). Reports the peaks above --target (ohm) and the band where
     |Z| stays under it. Below ~20 MHz the board's plane-pair capacitance and the capacitors' C dominate, which a
     Gaussian run to fmax resolves poorly: read the low end with a circuit model of the capacitors, not from here. DC
     drop is not this tool's job (FR-4 loss leaks at DC); use a DC grid solver.
Each writes <OUT_DIR>/<kind>.json, <kind>.png and FINDINGS_<kind>.md (a section the report collects)."""
import argparse
import json
import os
import re

import numpy as np

GROUND_DB = 6.0


def fcc_3m(f):
    return np.where(f < 88e6, 40.0, np.where(f < 216e6, 43.5, np.where(f < 960e6, 46.0, 54.0)))


def cispr_3m(f):
    return np.where(f < 230e6, 30.0, 37.0) + 10.5


def short(n):
    return n.split('/')[-1]


def raised_cosine(t, rise, t0=0.1e-9):
    T = rise / 0.5903
    x = np.clip((t - t0) / T, 0, 1)
    return 0.5 * (1 - np.cos(np.pi * x))


def crossing(t, v, level):
    i = np.where((v[:-1] < level) & (v[1:] >= level))[0]
    if not len(i):
        return None
    i = i[0]
    return float(t[i] + (level - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i]))


def si(out, model, swing=3.3, vil=0.3, vih=0.7, edge=0.8e-9):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    M = json.load(open(model))
    run = json.load(open(os.path.join(out, 'run.json')))
    tr = np.load(os.path.join(out, 'traces.npz'))
    exc = [p for p in M['ports'] if p['excite']][0]
    k = exc['nr']
    u = tr['port_ut_%d' % k] if 'port_ut_%d' % k in tr else tr['port_ut%d' % k]
    i = tr['port_it_%d' % k] if 'port_it_%d' % k in tr else tr['port_it%d' % k]
    t = tr['t']
    emf = u + exc['R'] * i
    if run.get('excitation', {}).get('kind') == 'raised-cosine step':
        scale = swing / float(np.median(emf[-max(5, len(emf) // 20):]) or emf.max())
        wave = {n: tr[n] * scale for n in tr.files if n.startswith('V_')}
        src = emf * scale
        how = 'edge-excited run scaled by its source EMF (x%.4g)' % scale
    else:                       # Gaussian: V(f) / Vsrc(f) times the spectrum of the wanted edge, in the excited band
        n = 1 << int(np.ceil(np.log2(len(t) * 2)))
        dt = float(np.median(np.diff(t)))
        f = np.fft.rfftfreq(n, dt)
        E = np.fft.rfft(emf, n)
        band = np.abs(E) > 1e-3 * np.abs(E).max()
        tt = np.arange(n) * dt
        X = np.fft.rfft(swing * raised_cosine(tt, edge), n)
        wave = {}
        for name in tr.files:
            if name.startswith('V_'):
                H = np.where(band, np.fft.rfft(tr[name], n) / np.where(band, E, 1), 0)
                wave[name] = np.fft.irfft(H * X, n)[:len(t)]
        src = swing * raised_cosine(t - t[0], edge)
        how = 'Gaussian run, response to a %.2g ns raised-cosine edge synthesised from V/Vsrc' % (edge * 1e9)
    t50 = crossing(t, src, swing / 2) or 0.0
    res = dict(how=how, swing=swing, vil=vil * swing, vih=vih * swing, driven={}, crosstalk_mv={}, run=dict(
        engine=run.get('engine'), steps=run.get('steps'), mcells_per_s=run.get('mcells_per_s')))
    lo, hi = vil * swing, vih * swing
    for p in M['probes']:
        v = wave.get(p['name'])
        if v is None:
            continue
        if p['net_hi'] == M['excite']:
            tc = crossing(t, v, swing / 2)
            run_max = np.maximum.accumulate(v)
            m = (run_max >= lo) & (v <= hi) & (run_max <= hi + 0.05 * swing)
            dip = float((run_max - v)[m].max()) if m.any() else 0.0
            top = np.nonzero(v >= 0.9 * swing)[0]          # any fall on the way up to 90 %, in or out of the band
            end = top[0] if len(top) else len(v)
            rise_dip = float((run_max[:end] - v[:end]).max()) if end else 0.0
            res['driven'][p['pad']] = dict(delay_ns=round((tc - t50) * 1e9, 3) if tc else None,
                                           peak_v=round(float(v.max()), 3), min_v=round(float(v.min()), 3),
                                           end_v=round(float(np.median(v[-max(5, len(v) // 20):])), 3),
                                           dip_in_band_v=round(dip, 3), dip_rising_v=round(rise_dip, 3))
        else:
            res['crosstalk_mv']['%s %s' % (short(p['net_hi']), p['pad'])] = round(float(np.abs(v).max()) * 1e3, 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t * 1e9, src, 'k:', lw=1, label='source EMF')
    for p in M['probes']:
        if p['name'] in wave and p['net_hi'] == M['excite']:
            axs[0].plot(t * 1e9, wave[p['name']], lw=1.2, label=p['pad'])
    axs[0].axhspan(lo, hi, color='0.92', zorder=0)
    axs[0].set_ylabel('V')
    axs[0].set_title('%s: %s at every pad (input band %.2g-%.2g V shaded)' % (M['name'], short(M['excite']), lo, hi),
                     fontsize=9)
    axs[0].legend(fontsize=7, ncol=3)
    for p in M['probes']:
        if p['name'] in wave and p['net_hi'] != M['excite']:
            axs[1].plot(t * 1e9, wave[p['name']] * 1e3, lw=1, label='%s %s' % (short(p['net_hi']), p['pad']))
    axs[1].set_ylabel('mV')
    axs[1].set_xlabel('ns')
    axs[1].set_title('crosstalk on the other nets', fontsize=9)
    axs[1].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'si.png'), dpi=140)
    plt.close(fig)
    json.dump(res, open(os.path.join(out, 'si.json'), 'w'), indent=1)
    L = ['## %s: signal integrity of %s\n' % (M['name'], short(M['excite'])),
         'openEMS on %s, %s steps. %s; swing %.2f V.\n' % (res['run']['engine'], res['run']['steps'], how, swing),
         '| Pad | 50 %% delay (ns) | Peak (V) | Min (V) | Dip in %.2g-%.2g V | Dip below 90 %% | End (V) |' % (lo, hi),
         '|---|---|---|---|---|---|---|']
    for pad, d in res['driven'].items():
        L.append('| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |' % (pad, d['delay_ns'], d['peak_v'], d['min_v'],
                                                                    d['dip_in_band_v'], d['dip_rising_v'], d['end_v']))
    if res['crosstalk_mv']:
        L.append('\nCrosstalk peaks: ' + ', '.join('%s %.1f mV' % kv for kv in sorted(
            res['crosstalk_mv'].items(), key=lambda kv: -kv[1])[:10]) + '.\n')
    L.append('\n![%s waveforms](si.png)\n' % M['name'])
    open(os.path.join(out, 'FINDINGS_si.md'), 'w').write('\n'.join(L))
    return res


def harmonics(kind, f0, A, duty, tr, fmax, fmin=30e6):
    n = np.arange(max(1, int(fmin // f0)), int(fmax // f0) + 1)
    fn = n * f0
    edge = np.minimum(1.0, 1.0 / (np.pi * n * tr * f0))
    if kind == 'switcher':
        body = np.minimum(1.0, 1.0 / (np.pi * n * duty))
        cn = 2 * A * duty * body * edge
    else:
        cn = np.maximum(2 * A * duty * np.abs(np.sinc(n * duty)), 2 * A * (duty - 0.02) * np.abs(np.sinc(n * (duty - 0.02))))
        cn = cn * edge
    keep = fn >= fmin
    return fn[keep], cn[keep]


def emi(out, model, sources):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    M = json.load(open(model))
    ff = np.load(os.path.join(out, 'farfield.npz'))
    f = ff['freqs']
    Epv = np.sqrt(np.abs(ff['E_theta_per_V']) ** 2 + np.abs(ff['E_phi_per_V']) ** 2)      # [f, theta, phi]
    worst_db = 20 * np.log10(np.maximum(Epv, 1e-30)).reshape(len(f), -1)
    res = dict(convention='|E| = |c_n| |E/V| worst direction, RMS, +%.0f dB ground, 3 m' % GROUND_DB, sources={})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fl = np.linspace(30e6, f[-1], 2000)
    for spec in sources:
        kind, f0, A, duty, tr = spec.split(':')
        f0, A, duty, tr = float(f0), float(A), float(duty), float(tr)
        fn, cn = harmonics(kind, f0, A, duty, tr, f[-1])
        if not len(fn):
            continue
        # interpolate each direction in dB on the solver grid, then take the worst direction per harmonic
        if len(fn) > 4000:
            grid = np.linspace(fn[0], fn[-1], 4000)
            env = np.max(np.stack([np.interp(grid, f, worst_db[:, j]) for j in range(worst_db.shape[1])]), 0)
            e_db = np.interp(fn, grid, env)
        else:
            e_db = np.max(np.stack([np.interp(fn, f, worst_db[:, j]) for j in range(worst_db.shape[1])]), 0)
        dbuv = e_db + 20 * np.log10(cn / np.sqrt(2) / 1e-6) + GROUND_DB
        m_fcc = fcc_3m(fn) - dbuv
        m_cis = cispr_3m(fn) - dbuv
        i, j = int(np.argmin(m_fcc)), int(np.argmin(m_cis))
        label = '%s %.4g MHz %.3g V %.2g ns' % (kind, f0 / 1e6, A, tr * 1e9)
        res['sources'][label] = dict(fcc_margin_db=round(float(m_fcc[i]), 1), fcc_at_mhz=round(float(fn[i] / 1e6), 2),
                                     level_dbuv=round(float(dbuv[i]), 1), cispr_margin_db=round(float(m_cis[j]), 1),
                                     cispr_at_mhz=round(float(fn[j] / 1e6), 2))
        if len(fn) > 400:
            ax.plot(fn / 1e6, dbuv, lw=0.8, label='%s: FCC margin %+.1f dB at %.1f MHz' % (label, m_fcc[i], fn[i] / 1e6))
        else:
            ax.plot(fn / 1e6, dbuv, '.-', ms=3, lw=0.6, label='%s: FCC margin %+.1f dB at %.1f MHz' % (label, m_fcc[i], fn[i] / 1e6))
    ax.plot(fl / 1e6, fcc_3m(fl), 'k-', lw=2, label='FCC 15.109 Class B, 3 m')
    ax.plot(fl / 1e6, cispr_3m(fl), 'k:', lw=1.5, label='CISPR 32 Class B at 3 m')
    ax.set_xscale('log')
    ax.set_ylim(-20, 80)
    ax.set_xlabel('MHz')
    ax.set_ylabel('dBuV/m at 3 m (+6 dB ground)')
    ax.set_title('%s: radiated emission, worst direction' % M['name'], fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'emi.png'), dpi=140)
    plt.close(fig)
    json.dump(res, open(os.path.join(out, 'emi.json'), 'w'), indent=1)
    L = ['## %s: radiated emission\n' % M['name'], res['convention'] + '.\n',
         '| Source | FCC 15.109 B margin | at (MHz) | Level (dBuV/m) | CISPR 32 B margin | at (MHz) |',
         '|---|---|---|---|---|---|']
    for k, d in res['sources'].items():
        L.append('| %s | %+.1f dB | %.2f | %.1f | %+.1f dB | %.2f |' % (k, d['fcc_margin_db'], d['fcc_at_mhz'],
                                                                       d['level_dbuv'], d['cispr_margin_db'], d['cispr_at_mhz']))
    L.append('\n![%s emission](emi.png)\n' % M['name'])
    open(os.path.join(out, 'FINDINGS_emi.md'), 'w').write('\n'.join(L))
    return res


def pdn(out, model, target=0.05, fmin=1e6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    M = json.load(open(model))
    sp = np.load(os.path.join(out, 'spectra.npz'))
    f, Z = sp['freqs'], sp['U1'] / np.where(np.abs(sp['I1']) > 0, sp['I1'], 1e-30)
    m = f >= fmin
    f, Z = f[m], np.abs(Z[m])
    pk = [i for i in range(1, len(Z) - 1) if Z[i] >= Z[i - 1] and Z[i] >= Z[i + 1] and Z[i] > target]
    pk = sorted(pk, key=lambda i: -Z[i])[:8]
    over = f[Z > target]
    res = dict(target_ohm=target, peaks=[dict(mhz=round(float(f[i] / 1e6), 1), ohm=round(float(Z[i]), 4)) for i in pk],
               first_over_mhz=round(float(over[0] / 1e6), 1) if len(over) else None, port=[p['pad'] for p in M['ports']
                                                                                       if p['excite']][0])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.loglog(f / 1e6, Z, lw=1.2, label='|Z| at %s' % res['port'])
    ax.axhline(target, color='k', ls=':', label='target %.3g ohm' % target)
    for i in pk:
        ax.annotate('%.0f MHz\n%.3g ohm' % (f[i] / 1e6, Z[i]), (f[i] / 1e6, Z[i]), fontsize=7, ha='center',
                    xytext=(0, 8), textcoords='offset points')
    ax.set_xlabel('MHz')
    ax.set_ylabel('ohm')
    ax.set_title('%s: %s impedance at the load' % (M['name'], short(M['excite'])), fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'pdn.png'), dpi=140)
    plt.close(fig)
    json.dump(res, open(os.path.join(out, 'pdn.json'), 'w'), indent=1)
    L = ['## %s: %s impedance at %s\n' % (M['name'], short(M['excite']), res['port']),
         'Z = U / I of a port at the load pad, decoupling as its ESL (no ESR, so the peaks read high). Target %.3g ohm; '
         '|Z| first exceeds it at %s MHz.\n' % (target, res['first_over_mhz']),
         '| Resonance (MHz) | abs(Z) (ohm) |', '|---|---|'] + ['| %.1f | %.4g |' % (p['mhz'], p['ohm']) for p in res['peaks']]
    L.append('\n![%s PDN](pdn.png)\n' % M['name'])
    open(os.path.join(out, 'FINDINGS_pdn.md'), 'w').write('\n'.join(L))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['si', 'emi', 'pdn'])
    ap.add_argument('out')
    ap.add_argument('model')
    ap.add_argument('--swing', type=float, default=3.3)
    ap.add_argument('--vil', type=float, default=0.3)
    ap.add_argument('--vih', type=float, default=0.7)
    ap.add_argument('--edge', type=float, default=0.8e-9)
    ap.add_argument('--source', action='append', default=[], help='kind:f:amplitude:duty:edge (kind clock|switcher)')
    ap.add_argument('--target', type=float, default=0.05, help='pdn: target impedance (ohm)')
    ap.add_argument('--fmin', type=float, default=1e6)
    a = ap.parse_args()
    if a.kind == 'si':
        r = si(a.out, a.model, a.swing, a.vil, a.vih, a.edge)
    elif a.kind == 'emi':
        r = emi(a.out, a.model, a.source)
    else:
        r = pdn(a.out, a.model, a.target, a.fmin)
    print(json.dumps(r, indent=1)[:3000])


if __name__ == '__main__':
    main()
