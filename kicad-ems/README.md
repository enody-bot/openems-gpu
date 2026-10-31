# kicad-ems: full-wave EM review of a KiCad board on rented GPUs

kicad-ems turns a KiCad board into openEMS models of the signals that matter, runs them on cloud GPUs through
this repo's `gpu-run`, and writes a PDF report of what it found. It can also render a Blender animation of the
fields on any line. It is the workflow Enody used on LP01, a six-layer lighting controller, cut down to five steps.

The work splits by cost. Steps 1 to 3 are free and run on your machine. Step 4 is the only one that costs money:
a 25 M-cell run of one bus is about **$0.05 on an RTX A5000** or **$0.13 on an RTX 5090**. LP01's whole
emission study, 15 full-wave runs of two boards joined by a flex, cost **$22.26**.

The easiest way in is [PROMPT.md](PROMPT.md). Give it to Claude Code (or another agent with a shell) in the board's
project folder, and it runs these steps with you, asking before anything that costs money.

> **The solver is new.** The image runs openEMS's GPU engine from SeanMollet/openEMS `GPU_merge` at ee45c6d, the
> branch of thliebig/openEMS PR #235. That work went into openEMS master on 2026-10-17 as c3a956b, and master has
> changed it since (the HIP backend is now a runtime plugin). It is not in a tagged release: v0.37.0 is older.
> The `:latest` image is still ee45c6d. An image of master d686f1d, built from this repo's `openems-master` branch,
> gave the same bus and emission results ([METHOD.md](METHOD.md), *Validation*). The HIP backend is compiled for
> NVIDIA with nvcc (CUDA 12.6), so on these cards it is effectively a CUDA engine. The repo's
> `examples/gpu-engine-check/` compares it with the CPU engines. Treat results as you would any
> pre-release solver: cross-check a new kind of model against a CPU run or a 2-D field solve once.

## What you get

| Step | Command | Cost | Output |
|---|---|---|---|
| 1 Triage | `python -m kems.triage board.kicad_pcb --out study/` | free | `TRIAGE.md`: ranked lines (edges, length, reference-plane gaps and changes, return vias), power nets, opens hidden from DRC; figures |
| 2 Model | `python -m kems.model study/lines/SCLK.json` | free | `models/SCLK.json`: the openEMS model as plain JSON |
| 3 Check | `python -m kems.check models/SCLK.json --png` | free | pass / fail before you pay: terminals on copper, nets in one piece on the mesh, no shorts, sheets on mesh lines, stability lint, time and $ per GPU |
| 4 Run | `python -m kems.make_job ... && ../gpu-run ...` | $ | traces, spectra, far field, field planes |
| 5 Report | `python -m kems.post si\|emi\|pdn ...` then `python -m kems.report study/` | free | figures, `FINDINGS_*.md`, `report.pdf` |
| + Render | `python -m kems.render.textures ...`, `scene.py`, `make_render_job` | $ | an MP4 of the fields on one line |

## Setup, once

1. This repo's `gpu-run` set up as in the [top README](../README.md) (RunPod key in the keychain, the Tailscale
   OAuth secret, optionally Modal). The rest of this guide assumes `../gpu-run gpus` works.
2. Python 3.10+ for the local steps:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r kicad-ems/requirements.txt
   ```
   openEMS's own Python (CSXCAD) is optional. With it, the mesh is smoothed by CSXCAD and you can run a short
   local smoke test, on the CPU or on Apple Silicon through the GPU branch's Metal backend. Without it, kicad-ems
   uses its own mesh grading.
3. KiCad's `kicad-cli` (KiCad 8 or later) if you want DRC cross-checks. The board file is only ever read.

## The five steps

### 1. Triage: which lines are worth a full-wave run

```bash
.venv/bin/python -m kems.triage my_board.kicad_pcb --out study/
```

This step reads the board, with no simulation. For every routed net it reports:
- the edge rate it assumes from the driver, and the length against that edge (is the line electrically long);
- every stretch where the adjacent copper under the track is not its reference net (a gap, a split or another
  rail);
- layer changes and how far the nearest return via is;
- the distance to the board edge;
- switch nodes, found by pin function.

It ranks the lines and lists the power nets with their decoupling. It also reports nets that are open in copper
but that KiCad's DRC passes: `jumper_pad_groups` on flow-through parts hide such opens.

Read `study/TRIAGE.md`. Pick the lines to simulate, and decide whether you want power-plane impedance runs (step
5, `pdn`).

### 2. Model: one study file per line or bus

Copy [examples/study.json](examples/study.json) to `study/lines/<NAME>.json` and fill in:
- the nets, with the one that carries the source;
- the driver pads with their output resistance (from the data sheet's drive strength);
- the receiver capacitance;
- the band (`fmax_hz`);
- the excitation: a Gaussian pulse for spectra, emission and impedance; an edge for waveforms and the animation;
- the field planes to record, if you want a render.

```bash
.venv/bin/python -m kems.model study/lines/SCLK.json
```

The model holds every fill, track, pad and via inside a region around the nets, and the stackup's dielectrics,
sub-layers included. Copper outside the region becomes one plane per layer. Decoupling capacitors become their ESL,
and two-pad parts become lumped elements bridged above their pads. The mesh follows the thirds rule at the
modelled tracks' edges. [METHOD.md](METHOD.md) gives the rules and why each one is there.

### 3. Check: catch a broken model before you pay for it

```bash
.venv/bin/python -m kems.check study/models/SCLK.json --png
```

It exits non-zero on anything that would waste a run:
- a port or load that is not on its copper;
- a modelled net that falls apart into pieces on the mesh;
- a short to another net, where a clearance is narrower than the cells;
- a copper sheet off its mesh line: openEMS silently drops it;
- a series R-L-C: openEMS's float32 update for it goes unstable.

It prints the cell count, the time step and an estimate of minutes and dollars on each GPU class.

### 4. Run: a smoke run first, then the full runs under a budget

```bash
.venv/bin/python -m kems.make_job study/models/SCLK.json --dest jobs/SCLK_smoke --smoke -- --td-planes
../gpu-run jobs/SCLK_smoke --gpu smoke --max-min 20
.venv/bin/python -m kems.make_job study/models/SCLK.json --dest jobs/SCLK -- --td-planes
../gpu-run sweep jobs/SCLK jobs/MOSI jobs/BUCK_SW --budget 5 --max-min 45
```

The smoke job runs 6 % of the steps. On the pod it checks that:
- the source drives its net: the ratio u / (R i) is about Z0 / R. An open port reads in the hundreds; on LP01
  this is how the dropped-sheet bug above was found;
- the edge reaches the other pads;
- no other net rings louder than the driven one;
- the energy did not grow.

`run.sh` fails if openEMS fell back to the CPU without saying so. The `--budget` cap is hard: `gpu-run` never
starts a run whose worst case, rate times `--max-min`, could take the spending past it. `--max-min` counts from pod
creation, so it includes the boot: set it to about 2x the expected job, because a slow host once took 18 minutes to
come up. Results come back to
`jobs/<NAME>/results/<UTC>-<gpu>/out/<NAME>/`.

### 5. Report: numbers, figures, PDF

```bash
OUT=jobs/SCLK/results/*/out/SCLK
.venv/bin/python -m kems.post si  $OUT study/models/SCLK.json --swing 3.3 --vil 0.3 --vih 0.7
.venv/bin/python -m kems.post emi $OUT study/models/SCLK.json --source clock:12e6:3.3:0.5:0.8e-9
.venv/bin/python -m kems.post pdn $OUT study/models/VCORE.json --target 0.05
cp -r jobs/SCLK/results/*/out/SCLK study/runs/SCLK       # the report collects study/runs/**/FINDINGS_*.md
.venv/bin/python -m kems.report study/ --title "My board 1.2.0: SI and EMI review"
```

- **`si`** reports, for each pad: delay, overshoot, undershoot, any dip back through the input threshold band, and
  crosstalk on the neighbours.
- **`emi`** applies the source's harmonics to the board's simulated far field. That gives the field at 3 m in
  the worst direction, against FCC 15.109 and CISPR 32 Class B.
- **`pdn`** gives a rail's impedance seen from a load pin, with its resonances.

`report.py` builds the PDF from Markdown, in this order:
1. `study/SUMMARY.md`: the ranked faults and what to change. You or Claude write it last.
2. `TRIAGE.md`.
3. Every run's findings.
4. `METHOD.md`.

### Render: watch the edge run down the line

```bash
.venv/bin/python -m kems.render.textures study/models/SCLK.json $OUT --dest render/SCLK --swing 3.3
blender -b --factory-startup --python kems/render/scene.py -- render/SCLK examples/shot.json render/SCLK.blend
.venv/bin/python -m kems.render.make_render_job render/SCLK examples/shot.json --dest jobs/render_SCLK --preview
../gpu-run jobs/render_SCLK --gpu "NVIDIA RTX A5000" --max-min 30
```

- **The run.** It needs an edge excitation with field planes: `"excitation": {"kind": "edge", ...}` and
  `td_planes` in the study, then `--td-planes` on the job.
- **Textures.** `textures.py` turns each copper layer and each field plane into PNG frames, on one log scale per
  field kind.
- **The scene.** `scene.py` stacks the layers as see-through sheets and puts each field plane between the layers it
  belongs to. It keys a camera that starts straight down while the edge runs, then swings down as the layers come
  apart. Building the `.blend` is free; open it to frame the shot.
- **The render.** `make_render_job.py` packages the scene for a GPU pod with the official Blender 4.1.1 build.
  `--preview` is a quarter of the pixels at 16 samples.

**Any NVIDIA card of Ampere or Ada class runs `scene.py` unchanged.** The LP01 bus shot, 300 frames at 1080p, took
21 minutes on a local RTX 3080: `blender -b ... -- SCENE SHOT OUT.blend --render FRAMES/`, then ffmpeg. The
cloud-pod wrapper `make_render_job.py` has not yet been run on gpu-run (see its docstring). Blender 4.1 has no kernels
for Blackwell cards, so on RunPod use an A5000 / L40S / 4090 class GPU.

## What it is not

- **It does not lay out or route anything.** It reads the board you drew and tells you what the fields do on it.
- **It is not a compliance test.** The emission numbers model the board's own structure in free space, or in an
  enclosure you add. They do not model the cables, the chamber or the detector. Treat a margin under about 6 dB
  as "measure this", not "passes".
- **It is not a DC solver.** FR-4 loss is modelled as a conductivity fixed at 1 GHz, which leaks at DC. Use a DC
  grid solver for IR drop.
- **It is not fully tested.** It is young code, packaged from the LP01 study scripts and exercised on one six-layer
  board and its cards. [METHOD.md](METHOD.md), *Validation*, says what has been checked and what has not.

## Files

| Path | What |
|---|---|
| `kems/board.py`, `sexpr.py`, `geom.py` | KiCad 8-10 board reader (stackup with sub-layers, fills, pads, vias, footprints, jumper groups) |
| `kems/triage.py` | step 1 |
| `kems/model.py` | step 2 |
| `kems/check.py` | step 3 |
| `kems/make_job.py`, `fdtd_run.py`, `smoke_check.py` | step 4 (the last two run on the pod; numpy and h5py only) |
| `kems/post.py`, `report.py` | step 5 |
| `kems/render/` | textures, the Blender scene, the render job |
| `PROMPT.md` | the agent prompt that runs all of it with you |
| `METHOD.md` | modelling rules, conventions, limits, validation |
| `examples/` | a study file and a shot |
