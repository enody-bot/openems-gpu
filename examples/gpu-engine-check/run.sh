#!/usr/bin/env bash
# openEMS's own GPU engine test (python/Tests/GPU_Engine.py, in the image): every case runs on the
# basic, gpu-reference and gpu engines and is compared. Every case must have reached the HIP
# backend; without a GPU the test skips the device comparison and still prints PASS.
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python /opt/openEMS/share/openEMS/tests/GPU_Engine.py 2>&1 | tee gpu_engine.log
n=$(grep -c 'HIP backend: max. deviation' gpu_engine.log || true)
cases=$(grep -c '^PASS \[' gpu_engine.log || true)
[ "$cases" -gt 0 ] && [ "$n" -eq "$cases" ] || { echo "FAIL: $n of $cases cases ran on the HIP backend"; exit 1; }
echo "OK: $n of $cases cases on the HIP backend, PASS"
