# openems-gpu

Run an openEMS simulation on a rented cloud GPU with one command. `gpu-run` starts a RunPod pod from
this repo's image, waits for it to join the tailnet as an ephemeral, locked-down node, copies the
job over Tailscale SSH, streams its output, copies the results back and terminates the pod.

```
Mac ── gpu-run ──> RunPod API: create pod (Secure Cloud, A100 / H100, this image, TS_OAUTH secret)
                    pod: tailscaled (userspace) joins as gpu-<pod id>, tag:gpu-pod, Tailscale SSH
Mac ── rsync/ssh over the tailnet ──> /scratch/job on the pod's NVMe, bash run.sh, output streamed
Mac <── rsync ── files the job wrote  ->  <job>/results/<UTC time>-<gpu>/
Mac ── RunPod API: terminate pod (finally: also on errors and Ctrl-C)
```

The solver is the GPU openEMS the 3080 box runs, unchanged: SeanMollet/openEMS `GPU_merge` at
ee45c6d (= thliebig/openEMS PR #235), its HIP backend compiled for NVIDIA by nvcc, in openEMS-Project
v0.37.0 (CSXCAD c46e320, fparser 4b9c845). The image builds it the way the box's
`openems-bench/BUILD.md` does, with GPU code for sm_61 (GTX 1070), sm_75 (2080 Ti), sm_80 (A100),
sm_86 (3080, A-series) and sm_90 (H100).

| File | What it is |
|---|---|
| `gpu-run` | the launcher (single-file Python, run by `uv`; no dependencies) |
| `Dockerfile` | two-stage build: CUDA 12.6 devel to build, CUDA 12.6 base + runtime libraries, tailscale, rsync, ssh client |
| `docker/entrypoint.sh` | pod start: watchdog, `tailscaled`, `tailscale up`, `tailscale logout` on exit |
| `docker/gpu-job` | runs the job on the pod, detached from the SSH session, and streams its log |
| `docker/hip-nvidia.sh`, `docker/localConfig.cmake` | the box's HIP-on-NVIDIA setup (AMD's signed headers, CMake settings) |
| `.github/workflows/image.yml` | builds and pushes `ghcr.io/enody-bot/openems-gpu:<git sha>` and `:latest` |
| `docs/tailnet-policy.md` | the tailnet policy, the OAuth client and the RunPod secret |
| `examples/smoke/` | 13.8 M-cell GPU smoke job (the box's T0002 test); fails on a silent CPU fallback |
| `examples/gpu-engine-check/` | openEMS's own `GPU_Engine.py`: 19 cases, GPU against CPU engines |

## One-time setup

1. **Tailnet policy.** Apply [docs/tailnet-policy.md](docs/tailnet-policy.md), sections 1 and 2.
2. **OAuth client and RunPod secret.** Same document, section 3: a Tailscale OAuth client with
   `auth_keys` write, limited to `tag:gpu-pod`, stored as the RunPod secret `TS_OAUTH`.
3. **RunPod API key.** RunPod console, **Settings > API Keys**, create a key with read/write
   access. Store it in the macOS keychain; this prompts for the value, so it stays out of the
   shell history:
   ```bash
   security add-generic-password -U -a "$USER" -s runpod-api-key -w
   ```
   gpu-run reads `$RUNPOD_API_KEY` first, then this keychain item. The account key stays on the Mac;
   pods never get it.
4. **Image (GHCR).** Pushing a change to `Dockerfile` or `docker/` on `main` runs the workflow, which
   pushes with the workflow's `GITHUB_TOKEN`. You don't need a personal token. After the first build,
   make the package public once: GitHub, **enody-bot > Packages > openems-gpu > Package settings >
   Change visibility > Public**. RunPod then pulls it without a registry credential.
5. **On the Mac:** `uv` and the Tailscale app. gpu-run finds the app's CLI by itself.

## Day to day

```bash
./gpu-run path/to/job                       # A100 80 GB, 240 min limit
./gpu-run path/to/job --gpu h100 --max-min 90
./gpu-run path/to/job --gpu smoke           # cheapest sm_86 cards: RTX A4000 / A4500 / A5000 / 3090
./gpu-run path/to/job --keep                # leave the pod up for debugging; prints how to reach it
./gpu-run bench path/to/job                 # same job on an A100, then an H100: time and $ per run
./gpu-run list                              # every pod on the account
./gpu-run kill-all                          # terminate every gpu-run pod (asks first; --yes skips)
```

**A job** is a directory with a `run.sh`. On the pod it is `/scratch/job` (without its `results/`),
`run.sh` runs there with the openEMS venv's `python`, `openEMS` and `nvidia-smi` on PATH, and its
output streams to your terminal. If the connection drops, gpu-run reconnects and resumes the stream
where it left off. The job keeps running in the meantime. Every file the job creates or changes comes back to
`<job>/results/<UTC time>-<gpu>/`, with `gpu-run.log` (the output) and `gpu-run.json` (GPU, boot,
job and pod times, rate, estimated cost). Run openEMS with `engine='gpu'` and check its log for
`Create FDTD engine (GPU, backend: HIP ...)`: openEMS silently falls back to the CPU when it finds
no usable GPU, as `examples/smoke/run.sh` shows.

**Cost.** gpu-run prints the GPU type, the job's wall time, the pod's lifetime (created to
terminated) and an estimate of the pod's lifetime times its hourly rate. RunPod bills per second,
so the estimate is close but not an invoice. `bench` also shows the job-only cost.

**Limits.** `--max-min` (default 240) bounds the whole run from pod creation. When it passes,
gpu-run copies back what the job has written and terminates the pod. Separately, the pod's
watchdog tries to stop the pod `--max-min` + 10 minutes after it started, in case gpu-run itself
died (laptop asleep, network gone).

### Safety

- **Secrets.** The image and the repo hold none. gpu-run sends RunPod only the
  `{{ RUNPOD_SECRET_TS_OAUTH }}` placeholder. The entrypoint never prints `TS_OAUTH`, keeps it from
  `tailscaled`'s environment and unsets it after `tailscale up`. The RunPod key reaches `curl` on
  stdin, not on a command line.
- **Network.** The pod exposes no public ports. It joins as an ephemeral `tag:gpu-pod` node, which
  the policy lets admins and `tag:gpu-controller` reach but lets reach nothing, and logs out on
  exit. Jobs run on `/scratch`, the pod's container disk on local NVMe, never a network mount.
- **Self-stop.** RunPod gives each pod a pod-scoped `RUNPOD_API_KEY` (its docs call it
  "Pod-scoped API key" and say no more). The watchdog uses that key, never your account key. It
  tries terminate (REST v1, REST v2, GraphQL) and then stop, and logs each HTTP status. Community
  reports differ on whether that key may stop its own pod. **Until a smoke test shows one of those
  calls succeeding, count on gpu-run's `finally` and `gpu-run list` / `kill-all`, not on the
  watchdog.**

## CUDA version

The image uses CUDA **12.6.3** (devel to build, base at run time):

- It is what the 3080 box builds and validates with (Windows driver 566.36 = CUDA 12.7, and there
  is no 12.7 toolkit), so the same image runs there under WSL.
- The AMD 7.0.3 HIP headers the backend compiles against work with CUDA 12.6. 7.2.x needs CUDA 12.8+
  (`cudaMemcpyBatchAsync`), and 7.1.1 fails.
- It supports sm_90 for the H100.
- RunPod lets a pod request host CUDA versions (`allowedCudaVersions`; its API lists 11.8 to 13.0).
  gpu-run asks for 12.6 to 13.0, so the pod never lands on a host whose driver is older than the
  image.

## Local check on the 3080 box

Without the tailnet, the image's `gpu-job run` runs a job directly:

```bash
docker pull ghcr.io/enody-bot/openems-gpu:latest
docker run --rm --gpus all -v "$PWD/examples/gpu-engine-check:/scratch/job" --entrypoint gpu-job ghcr.io/enody-bot/openems-gpu:latest run
docker run --rm --gpus all -v "$PWD/examples/smoke:/scratch/job" --entrypoint gpu-job ghcr.io/enody-bot/openems-gpu:latest run
```

`docker run ... --entrypoint cat <image> /opt/openEMS/BUILD-INFO.txt` shows the pinned sources, the
code architectures and the Python packages.
