#!/usr/bin/env bash
# openEMS's own GPU engine test (python/Tests/GPU_Engine.py at ee45c6d, in the image): 19 cases run
# on the basic, gpu-reference and gpu engines and compared. Every case must have reached the HIP
# backend; without a GPU the test skips the device comparison and still prints PASS.
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python /opt/openEMS/share/openEMS/tests/GPU_Engine.py 2>&1 | tee gpu_engine.log
n=$(grep -c 'HIP backend: max. deviation' gpu_engine.log || true)
[ "$n" -eq 19 ] || { echo "FAIL: $n of 19 cases ran on the HIP backend"; exit 1; }
echo "OK: 19 of 19 cases on the HIP backend, PASS"
