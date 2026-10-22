#!/usr/bin/env bash
# Smoke job: the 3080 box's T0002 test (smoke_gpu.py, unchanged): a 13.8 M-cell free-space box with
# PML and a lumped port, NRTS timesteps (default 3000) on the GPU engine. openEMS falls back to the CPU
# silently when it finds no usable GPU, so the job fails unless the HIP backend was created.
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python smoke_gpu.py gpu "${NRTS:-3000}" sim_gpu | tee smoke_gpu.json
python - <<'EOF'
import json
r = json.load(open("smoke_gpu.json"))
engine = r["engine_created"] or ""
assert "backend: HIP" in engine, f"openEMS did not create the GPU engine: {engine!r}"
print(f"OK: {engine}, {r['cells']} cells, {r['mcells_per_s']} MCells/s, {r['wall_s_total']} s")
EOF
