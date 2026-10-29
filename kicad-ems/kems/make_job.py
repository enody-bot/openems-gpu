"""Step 4: package one model as a gpu-run job directory (see ../README.md and the repo's README).

    python -m kems.make_job models/NAME.json --dest jobs/NAME [--smoke] [-- <fdtd_run.py arguments>]

The job holds fdtd_run.py, the model, SHA256SUMS and a run.sh that fails (non-zero exit, so gpu-run reports it) when:
  2  a file arrived damaged (SHA256SUMS);
  3  openEMS did not create the GPU engine: it falls back to the CPU silently when it finds no usable GPU;
  4  the energy grew (the runner's watchdog stopped an unstable run);
  5  --smoke: the smoke check failed (the source did not drive its net, or the energy did not fall).
A smoke job runs 6 % of the steps (at least 20,000) and checks the start; run it first on every new model, then the
full job. Then:
    ../gpu-run jobs/NAME --gpu smoke --max-min 20                      # one job, cheapest cards first
    ../gpu-run sweep jobs/*/ --budget 10 --max-min 60                  # many at once under a hard budget
Results come back to jobs/NAME/results/<UTC time>-<gpu>/out/NAME/."""
import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

RUN_SH = """#!/usr/bin/env bash
# {name}: {model} through fdtd_run.py {args}; packaged by kems/make_job.py on {date}.
set -uo pipefail
sha256sum -c SHA256SUMS || exit 2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > gpu_telemetry.csv 2>&1 &
TEL=$!
python fdtd_run.py --model models/{model} --sim /scratch/sim/{name} --out out/{name} {args} 2>&1 | tee {name}.log
rc=${{PIPESTATUS[0]}}
kill $TEL 2>/dev/null
rm -rf /scratch/sim
engine=$(grep -h 'Create FDTD engine' out/{name}/openems.log 2>/dev/null | tail -1)
echo "=== {name} rc=$rc engine: $engine"
[ "$rc" -eq 0 ] || exit "$rc"
case "$engine" in *"backend: HIP"*) ;; *) echo "FAIL: openEMS did not run on the GPU"; exit 3 ;; esac
if [ -e out/{name}/UNSTABLE.txt ] || grep -q '"unstable": true' out/{name}/run.json; then
  echo "FAIL: the energy grew; see out/{name}/run.json"; exit 4
fi
{smoke}exit 0
"""

SMOKE = """python smoke_check.py out/{name} models/{model} || exit 5
"""


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        extra = argv[argv.index('--') + 1:]
        argv = argv[:argv.index('--')]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--dest', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--name')
    a = ap.parse_args(argv)
    M = json.load(open(a.model))
    name = a.name or os.path.basename(a.dest.rstrip('/'))
    if a.smoke:
        steps = M['stats'].get('steps') or int((M.get('excitation', {}).get('tend_s') or 20e-9) / M['stats']['dt_s'])
        extra = ['--nrts', str(max(20000, int(0.06 * steps)))] + [x for x in extra if x != '--td-planes']
    if os.path.exists(a.dest):
        shutil.rmtree(a.dest)
    os.makedirs(os.path.join(a.dest, 'models'))
    mname = os.path.basename(a.model)
    shutil.copy(a.model, os.path.join(a.dest, 'models', mname))
    shutil.copy(os.path.join(HERE, 'fdtd_run.py'), a.dest)
    shutil.copy(os.path.join(HERE, 'smoke_check.py'), a.dest)
    open(os.path.join(a.dest, 'run.sh'), 'w').write(RUN_SH.format(
        name=name, model=mname, args=' '.join(shlex.quote(x) for x in extra), date=datetime.date.today().isoformat(),
        smoke=SMOKE.format(name=name, model=mname) if a.smoke else ''))
    os.chmod(os.path.join(a.dest, 'run.sh'), 0o755)
    sums = []
    for f in ('fdtd_run.py', 'smoke_check.py', 'models/' + mname):
        sums.append('%s  %s' % (hashlib.sha256(open(os.path.join(a.dest, f), 'rb').read()).hexdigest(), f))
    open(os.path.join(a.dest, 'SHA256SUMS'), 'w').write('\n'.join(sums) + '\n')
    print('job %s -> %s (%s)' % (name, a.dest, ' '.join(extra) or 'model defaults'))


if __name__ == '__main__':
    main()
