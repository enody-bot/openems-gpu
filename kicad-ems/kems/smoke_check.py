"""Smoke check of a short run (numpy only, runs on the pod): did the source drive its net, and is the run sane?

    python smoke_check.py out/NAME models/NAME.json

Judged by ratios, not volts: a lumped port's source is 1 V/m times the port length, so absolute levels mean nothing.
  drive     at the excited port |u| / R|i| between 0.05 and 20 (a port on the wrong copper reads ~0 or ~inf)
  follows   at least one other probe on the excited net reaches 30 % of the port's peak voltage (the edge got there;
            a switch node driving a resonant loop can read lower, check its spectrum instead)
  quiet     no probe on another net exceeds the excited net's peak (a short in the model would)
  energy    the energy did not grow past 30 x its peak during the pulse
Exit 1 with the reasons when a test fails; writes out/NAME/smoke.json."""
import json
import os
import sys

import numpy as np


def main():
    out, model = sys.argv[1], sys.argv[2]
    M = json.load(open(model))
    tr = np.load(os.path.join(out, 'traces.npz'))
    run = json.load(open(os.path.join(out, 'run.json')))
    exc = [p for p in M['ports'] if p['excite']][0]
    k = exc['nr']
    u = tr['port_ut_%d' % k] if 'port_ut_%d' % k in tr else tr['port_ut%d' % k]
    i = tr['port_it_%d' % k] if 'port_it_%d' % k in tr else tr['port_it%d' % k]
    fails = []
    ratio = float(np.abs(u).max() / max(exc['R'] * np.abs(i).max(), 1e-30))
    if not 0.05 < ratio < 20:
        fails.append('drive: |u| / R|i| = %.3g at the source port' % ratio)
    peak = float(np.abs(u).max())
    own, other = {}, {}
    for p in M['probes']:
        if p['name'] in tr:
            (own if p['net_hi'] == M['excite'] else other)[p['name']] = float(np.abs(tr[p['name']]).max())
    far = {n: v / peak for n, v in own.items() if n != 'V_' + exc['name']}
    if far and max(far.values()) < 0.3:
        fails.append('follows: no probe on %s reaches 30 %% of the port peak (best %.2f)' % (M['excite'],
                                                                                          max(far.values())))
    if other and own and max(other.values()) > max(own.values()):
        n = max(other, key=other.get)
        fails.append('quiet: %s on another net peaks above the excited net (%.3g > %.3g)' % (n, other[n],
                                                                                            max(own.values())))
    if run.get('unstable'):
        fails.append('energy: grew (unstable)')
    r = dict(drive_ratio=ratio, follow=far, others_peak=max(other.values()) if other else None,
             engine=run.get('engine'), steps=run.get('steps'), mcells_per_s=run.get('mcells_per_s'), fails=fails)
    json.dump(r, open(os.path.join(out, 'smoke.json'), 'w'), indent=1)
    print(json.dumps(r, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
