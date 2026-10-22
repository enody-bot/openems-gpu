# openems-gpu

Run openEMS simulations on rented cloud GPUs, one or many at a time. `gpu-run` starts a RunPod pod
from this repo's image, waits for it to join the tailnet as an ephemeral, locked-down node, copies the
job over Tailscale SSH, streams its output, copies the results back and terminates the pod. `gpu-run
sweep` runs many jobs at once under a budget, on RunPod's cheap cards first and on Modal (B200 / H200 /
H100 sandboxes from the same image) when RunPod has none free.

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
| `gpu-run` | the launcher (single-file Python run by `uv`; its only dependency is `modal`) |
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
3. **RunPod API key.** RunPod console, **Settings > API Keys**, create a restricted key with
   **GraphQL: Read/Write** (pods are created, listed and terminated through it, also over the REST
   API) and **api.runpod.ai: None** (that is the serverless API, which gpu-run doesn't use). A key
   with GraphQL read only can list pods but gets an empty HTTP 403 on every create or terminate.
   Store it in the macOS keychain; this prompts for the value, so it stays out of the shell history:
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
6. **Modal (for `--backend modal` and `sweep --modal N`).** Sign in once; this opens a browser and writes
   the token to `~/.modal.toml` (it never goes into the repo or a sandbox):
   ```bash
   uvx modal token new
   ```
   The Starter plan includes $30 of compute a month and runs at most 10 GPUs at a time (Team: 50).

## Day to day

```bash
./gpu-run path/to/job                       # A100 80 GB, 240 min limit
./gpu-run path/to/job --gpu h100 --max-min 90
./gpu-run path/to/job --gpu smoke           # cheapest cards first: A4000 ... 3090, A6000, L4 (Secure Cloud stock varies)
./gpu-run gpus                              # Secure Cloud GPUs in stock now, with prices
./gpu-run path/to/job --keep                # leave the pod up for debugging; prints how to reach it
./gpu-run bench path/to/job                 # same job on an A100, then an H100: time and $ per run
./gpu-run bench path/to/job --gpus a100,h100,"NVIDIA L40S" --parallel 3 --budget 10 --max-min 20
                                            # any GPUs, 3 at a time; never starts a run whose worst case
                                            # (rate x --max-min) could take spending past the budget
./gpu-run path/to/job --backend modal --gpu fast   # one job in a Modal sandbox: B200, else H200, else H100
./gpu-run list                              # every pod on the account, and gpu-run's Modal sandboxes
./gpu-run kill-all                          # terminate every gpu-run pod and sandbox (asks first; --yes skips)
```

**Sweeps.** One job directory per variant; each runs on its own GPU and its results land in that
directory's `results/`, as with a single run.

```bash
./gpu-run sweep variants/*/ --budget 15 --max-min 90                    # RunPod "cheap" pool, 4 at a time
./gpu-run sweep variants/*/ --budget 40 --max-min 90 --runpod 8 --modal 10
                                            # 8 RunPod pods and up to 10 Modal sandboxes at once
./gpu-run sweep variants/*/ --budget 30 --max-min 45 --runpod 6 --runpod-gpu fast
```

- A job takes the first backend with a free slot: RunPod, then Modal. When RunPod has no card of the
  pool free, that pool cools down for 5 minutes and its jobs go to Modal (or wait).
- `--budget` is required. A run starts only if what is spent, plus the worst case (rate x `--max-min`) of
  every run going, plus its own worst case stays within it; a job that could never fit is not run.
- RunPod pools, from the benchmark (`src/openems_bench` in the workspace): **cheap** (cost per run): RTX
  A5000, RTX PRO 4500 Blackwell, RTX A6000, RTX PRO 4000 Blackwell, RTX 4090, RTX 6000 Ada; **fast**
  (speed): B200, H200, H100 SXM, H100 NVL, RTX PRO 6000 Blackwell. Also `a100`, `h100`, `smoke` or a
  RunPod GPU id. Modal: **fast** (B200, H200, H100), **cheap** (L40S, A100 80 GB), `b200`, `h200`,
  `h100`, `a100`, `l40s` or a Modal GPU name. For LP01-sized runs the cheap pool is about 3080 speed
  for about $0.30 a run; the fast pool is about 3x faster for about $2-3.

**Modal** runs the same image (its tailnet entrypoint cleared) in a Modal Sandbox: no tailnet, no
SSH. gpu-run resolves the image tag to its digest first, because Modal caches image builds by name.
The job goes in and its new or changed files come back as tarballs (up to 5 GB). The sandbox has 4
cores and 32 GiB for the model setup, billed with the GPU (prices in `gpu-run`, from modal.com/pricing,
October 2026). Modal ends the sandbox at `--max-min` even if gpu-run dies. The first sandbox from a new
image waits a few minutes while Modal builds it.

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
  "Pod-scoped API key" and say no more). The watchdog uses that key, never your account key. Tested
  2026-10-22 on an RTX A6000 pod: the REST v2 terminate succeeded (HTTP 204, the pod and its tailnet
  node gone within 15 s) and REST v1 was refused (403). The watchdog tries v2 first, then the others,
  and logs each HTTP status. gpu-run's `finally` stays the main guard; `gpu-run list` / `kill-all`
  clean up anything left.

## CUDA version

The image uses CUDA **12.6.3** (devel to build, base at run time):

- It is what the 3080 box builds and validates with (Windows driver 566.36 = CUDA 12.7, and there
  is no 12.7 toolkit), so the same image runs there under WSL.
- The AMD 7.0.3 HIP headers the backend compiles against work with CUDA 12.6. 7.2.x needs CUDA 12.8+
  (`cudaMemcpyBatchAsync`), and 7.1.1 fails.
- It supports sm_90 for the H100.
- The host's driver must support CUDA 12.6 or later. RunPod's REST filter for host CUDA versions
  stops at 13.0, while its A100 and H100 hosts already run 13.2 (October 2026). So gpu-run reads the
  driver's version on the pod with `nvidia-smi` and terminates the pod at once if it is older.

## Local check on the 3080 box

Without the tailnet, the image's `gpu-job run` runs a job directly:

```bash
docker pull ghcr.io/enody-bot/openems-gpu:latest
docker run --rm --gpus all -v "$PWD/examples/gpu-engine-check:/scratch/job" --entrypoint gpu-job ghcr.io/enody-bot/openems-gpu:latest run
docker run --rm --gpus all -v "$PWD/examples/smoke:/scratch/job" --entrypoint gpu-job ghcr.io/enody-bot/openems-gpu:latest run
```

`docker run ... --entrypoint cat <image> /opt/openEMS/BUILD-INFO.txt` shows the pinned sources, the
code architectures and the Python packages.
