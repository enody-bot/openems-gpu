# Method, conventions and limits

These are the rules kicad-ems encodes. Most were learned the hard way on LP01, each from a run that gave a wrong
answer without failing. The report appends this file, so a reader can see how each number was made.

## The solver

openEMS is an FDTD solver: it steps E and H on a rectilinear (Yee) mesh in time. The GPU engine comes from
SeanMollet/openEMS `GPU_merge` at ee45c6d (thliebig/openEMS PR #235), with its HIP backend compiled for NVIDIA. The
work went into openEMS master on 2026-10-17 as c3a956b and has changed there since; it is not in a tagged release.
The numbers below are from ee45c6d unless they say otherwise; *Validation* compares master d686f1d. The repo's
`examples/gpu-engine-check/` runs openEMS's own GPU-against-CPU comparison: 19 cases at ee45c6d, 28 at d686f1d.

| Machine | Measured on LP01 models |
|---|---|
| Apple M1, CPU engine | ~0.25 GCells/s |
| Apple M1, the branch's Metal backend | ~0.5-0.6 GCells/s |
| RTX 3080 | ~5.1 GCells/s |
| RTX 5090 (RunPod, ~$1.19/h when measured) | 13.8 GCells/s |
| B200 (Modal) | 20.5 GCells/s |

One run of a 25-45 M-cell, 8 ns model is 3-6 minutes of solver time on a 5090. Pod start, image pull and copy-back
add 5-10 minutes of wall time, which is billed.

**The packaged tool, benchmarked** (2026-10-29): the LP01 LEFT bus model, 35.2 M cells, 135,700 steps (8 ns), one
pod per class at once on RunPod Secure Cloud. Prices are RunPod's on the day.

| GPU | $/h | Solver | GCells/s | openEMS setup | Job wall | Cost |
|---|---|---|---|---|---|---|
| RTX PRO 4500 Blackwell | 0.72 | 12.2 min | 6.66 | 5.4 min | 18.2 min | $0.23 |
| A100 SXM4 80 GB | 1.79 | 11.6 min | 7.00 | 6.3 min | 18.7 min | $0.58 |
| RTX PRO 6000 Blackwell Server | 2.49 | 6.4 min | 12.58 | 2.5 min | 9.5 min | $0.92 |
| H100 80 GB HBM3 | 3.99 | 5.8 min | 13.97 | 8.4 min | 14.8 min | $1.09 |
| A40 | 0.59 | did not finish in the 30 min cap | < 5.4 | | 30 min | $0.30 |

- **The four that finished gave bit-identical waveforms.** The engine is deterministic across GPU generations.
- **openEMS's setup is single-threaded on the pod's CPU** and took 2.5-8.4 minutes: as long as the solve on the fast
  cards. It depends on the host CPU, not the GPU, so the fastest GPU is not always the fastest finish.
- A RTX PRO 6000 Blackwell solved the 20.4 M-cell, 697,055-step buck model at 13.67 GCells/s.

## Building the model

**Copper.** Each copper layer is a zero-thickness PEC sheet:
- the outer layers sit on their inner face;
- the inner layers sit mid-copper.

Every fill, track, pad and via inside the region is kept exactly as saved. Outside the region, each layer
becomes one plane of its dominant net, with antipads where the other nets' vias pass.

**Dielectrics.** The stackup is used as KiCad stores it, sub-layers included. Loss is a conductivity
σ = 2π f ε0 εr tanδ, fixed at 1 GHz: right at 1 GHz and too lossy far below it. As a result, receivers settle a
few percent under the swing, and DC results are wrong; see *Limits*.

**Parts.**
- Drivers are lumped ports from the pad down to the reference layer, with the data sheet's output resistance. The
  excited one carries the source.
- Receivers are lumped capacitors from the pad to the reference layer.
- Two-pad series parts are lumped R, L or C bridged in the air above their pads, the way the part's body sits.
- Decoupling capacitors between a rail and ground are their ESL alone.
- **Never a series R-L-C.** openEMS runs it as a float32 recursion, and at decoupling values that recursion
  diverges.
- **No lumped L over 100 nH.** A series power inductor is modelled as its winding capacitance: well above its
  self-resonance (tens of MHz) that is what it is, and as a lumped L it breaks the solver. On LP01 the buck's
  3.3 µH L701 took the run to NaN within 3,000 steps, on the GPU and on the Metal engine alike; without it the run
  was finite. The default winding C is 2 pF; give `self_c` per part from the data sheet's SRF,
  C = 1 / ((2π SRF)² L).

**Ports and every lumped element sit on mesh nodes.** openEMS decides a PEC edge by its centre: a point on a
boundary counts as inside, and the highest priority wins. A port off a node may drive the dielectric beside its
pad. `model.py` moves every terminal onto a node; `check.py` verifies it.

**Copper sheets sit exactly on z mesh lines.** openEMS places a zero-thickness polygon only on a z line equal to
its elevation. A stackup sum such as 1.5740000000000003 misses the line at 1.574, and the sheet is dropped with
nothing but an "Unused primitive" warning in the log. kicad-ems rounds every z to the mesh's 10 nm grid and
fails the check on any mismatch. On LP01 this dropped F.Cu and In1 from a whole bus model: the driver's port
then fed an isolated pad and read |u| / R|i| = 182 instead of ~0.8.

**The mesh.**
- **The thirds rule** at every long straight edge of the modelled tracks: one line h/3 inside the metal and one
  2h/3 outside. This places the edge's current crowding.
- **A uniform lattice in the region**, offset a quarter cell from the board's 0.05 mm grid, so that copper edges
  never fall exactly on lines.
- **Grading** to the board and air cells, at no more than ~1.3 from one cell to the next.
- **z lines** at every sheet and a few per dielectric.
- **Why the thirds rule.** Thin strips on a uniform mesh run slow: 13 % at 0.05 mm cells, 43 % at 0.1 mm. With the
  thirds rule the delay matches a 2-D field solve.
- **No needless small cells.** The time step follows the smallest cell, so one stray 10 µm cell doubles the run.
  `model.py` drops lattice lines closer than two thirds of the fine cell to a fixed line.

**Boundaries.** PML around air boxes. Planes that run into a PML or Mur boundary blow up, so a model that cuts
planes off at its edge uses PEC walls and runs longer.

## Checking before paying (`check.py`)

| Test | Why |
|---|---|
| sheets, ports and lumped ends exactly on z lines | an off-line sheet is silently dropped |
| every port, load and probe terminal on a node of its own net, at both ends | a port on the wrong copper drives nothing, or shorts |
| each modelled net in one piece on the mesh, via barrels and part posts included | a neck narrower than a cell opens the net in the model even when the board is fine |
| no node of a modelled net owned by another net | a clearance narrower than a cell shorts the nets |
| no series R-L-C | float32 instability |
| no lumped L over 100 nH | NaN within a few thousand steps |
| cells, time step, steps, memory, minutes and $ per GPU | the quote you approve |

**The smoke run** (`smoke_check.py`, on the pod) runs 6 % of the steps and tests:
- the drive ratio |u| / R|i| at the source, between 0.05 and 20 (a line reads about Z0 / R; an open port reads in
  the hundreds);
- that the edge reaches the other pads of the net;
- that no other net rings louder than the driven one;
- that the energy did not grow.

`run.sh` fails a run that fell back to the CPU, or whose energy grew past 30× its peak (the watchdog). The
watchdog kills a run at once when its energy reads NaN or inf. An earlier version missed that case and let a NaN run
go its full 18 minutes ($0.81).

**Always smoke first.** The one run in this validation that went straight to a full job was the one that blew up;
its smoke run would have cost about $0.05.

**Judge by ratios, not volts.** openEMS's lumped source is a soft source whose EMF depends on the port's R. kicad-ems
scales every waveform and field by the source EMF u + R·i of the driven port, to the swing you give.

## Signal integrity (`post.py si`)

**Delay.** Measured at 50 % of the swing, after the source crosses 50 %.

**The dip in the band.** The deepest fall back while the edge is between V_IL and V_IH: a dip there can
double-clock an edge-triggered input.

**Crosstalk.** The peak voltage at every pad of the other modelled nets.

**Gaussian runs.** A Gaussian run gives an edge response by V(f) / Vsrc(f) inside the excited band, times the
spectrum of a raised-cosine edge.

## Emission (`post.py emi`)

The far field comes from an NF2FF box round the board, per volt of source, in every direction. In the conventions
used for LP01:

**Source spectrum.** Each harmonic n f of a trapezoid of amplitude A, duty d and edge tr has
|c_n| = 2 A d |sinc(n d)| · min(1, 1 / (π n tr f)).
- For a clock, the larger of duty d and d - 2 % (the sinc nulls move with duty).
- For a switcher, the duty term is taken as its envelope, min(1, 1 / (π n d)), since its duty moves with load.

**Field.** |E| = |c_n| · |E/V|(f_n):
- in the worst direction (θ 0-180°, φ 0-350° in 10° steps), both polarisations;
- interpolated in dB between solver frequencies;
- as an RMS value (÷√2) at 3 m;
- plus 6 dB for the ground reflection of an OATS or semi-anechoic site.

**Limits.** FCC 15.109 Class B at 3 m: 40 / 43.5 / 46 / 54 dBµV/m from 30 / 88 / 216 / 960 MHz. CISPR 32 Class B
at 10 m is scaled to 3 m (+10.5 dB), giving 40.5 / 47.5 dBµV/m below / above 230 MHz.

**Not modelled.** Spread spectrum, the quasi-peak detector, cables and the antenna height scan.

**Run length.** Emission runs use a fixed length, not the -40 dB energy stop. The stop is tested every ~2,000
steps while a ringing board's energy swings ±5 dB, so it fires on a dip. On LP01 it ended two flash-clock runs at
less than 40 % of their length, with the radiating current still only 1.5-14 dB down.

## Power-plane impedance (`post.py pdn`)

A port with a small R (1 Ω) at a load's power pin. The rail's capacitors are lumped C bridged over their pads: the
bridge's geometry gives the mounting inductance. Z(f) = U / I of the port.
- **ESR is not modelled**, so resonances read high. That flags them; it does not size them.
- **The low end is weak.** Below ~20 MHz a Gaussian run to fmax resolves the plane capacitance and the
  capacitors' C poorly. Use a circuit model there.
- **DC drop is not covered.** Use a DC grid solver.

## Rendering (`render/`)

**Textures.** Each field plane, as |E| or |H tangential| per frame, scaled to the swing, is coded 8-bit on a log
scale per field kind, with hi = the 99.9th percentile of the strongest frame and lo = 50 dB below. All frames of
one kind share the scale, so brightness compares across planes and runs.

**The scene.** Copper layers are unlit see-through sheets: lit, a stack of layers turns into an opaque slab that
hides the fields inside it. Fields are emission added over what is behind them.

**Rendering.** Cycles on the GPU, in Blender 4.1.1. There are no kernels for Blackwell cards. Measured on an
RTX 3080 (CUDA, OIDN): the full LEFT bus shot, 300 frames at 1920 × 1080 and 32 samples with 4 field sheets, took
4.3 s a frame, 21 minutes in all, in 2.5 GB of VRAM. The 960 × 540, 16-sample preview took 1.4 s a frame.

**Image sequences and frame_offset.** Blender clamps the scene frame to the sequence's `frame_duration` *before* it
adds `frame_offset`. With a duration equal to the field frames, a shot longer than them ran off the sequence's end,
and every sheet rendered as a flat slab of the missing-texture colour. `scene.py` sets the duration to the shot's
length.

## Limits

These hold for every result:
- **The FR-4 model.** Fixed-frequency loss, isotropic εr, no glass weave.
- **The parts.** Receivers are capacitors and drivers are resistors; there are no IBIS models.
- **The copper.** Zero-thickness, lossless PEC: no skin-effect loss, so ringing reads a little high.
- **Emission.** The board alone, or in a simple enclosure you add. No cables, unless you model them, and no
  chamber.
- **The mesh.** Clearance gaps narrower than two cells are not resolved. The check reports those that short;
  narrow but unshorted gaps still carry a few percent of error in the coupling across them.

## Validation

What has been checked, as of 2026-10-30. The runs, numbers and PDF are in LP01's
`simulation/kems_validation/` (SUMMARY.md, `kems_validation_report.pdf`, `master_check/RESULTS.md`).

**Against openEMS master d686f1d** (2026-10-30), on the same GPU classes as the ee45c6d runs:
- The engine check passed 28 of 28 cases on HIP, worst deviation 4.2e-7 of the peak.
- The bus model: every delay, peak, minimum and dip identical to 0.001 ns and 0.001 V; crosstalk within 0.5 mV.
- The buck emission: both FCC margins identical, +22.3 and +17.2 dB at 1466 MHz.
- Master is not bit-identical to ee45c6d: the quiet lines differ by about 2e-4 of the drive, on Metal and on HIP alike,
  while the two backends agree to 2e-7. The change was not traced to a commit. Master's drawing tolerance and its
  lumped-RLC frequency limit are the candidates.

**Triage against KiCad DRC.** Three LP01 boards: the main board, the PD card and the Wi-Fi card. Triage's copper
connectivity matched kicad-cli DRC on every net. It found, independently:
- the ten opens behind `jumper_pad_groups` on the LP01 main board, which the LP01 study had found by its own
  copper walk;
- two on the PD card (USB_JTAG_DP / DN through U1's flow-through pads), which DRC confirms once the groups are
  removed.

**SI against the LP01 study's own model.** The LP01 LEFT SPI bus, eight nets, seven drivers, built by `model.py`
straight from the board: 35.2 M cells, 8 ns, on cloud GPUs. Against the LP01 study's separately built model of
the same bus (2026-10-24, RTX 5090):
- delays at the four receivers within 0.6 % (1.749 / 1.738 ns at the far end);
- the fall on the rising edge within 0.02 V;
- peaks within 1.5 %;
- MOSI crosstalk 110-153 mV against 110-157 mV.

**Determinism.** The same job on an RTX PRO 4500, RTX PRO 6000 Blackwell, A100 and H100 gave bit-identical
waveforms.

**Emission.** The LM5148 buck's switch node on one LP01 main board, in free space, 60 ns:
- **The physics.** The board's half-wave mode shows at 440 MHz, against ~460-490 MHz expected for a 305 mm strip.
- **The assumption.** The assumed winding capacitance of the 3.3 µH choke, swept over 0.5-6 pF, moved the worst
  FCC margin by 1.8 dB (+23.1 to +21.3 dB at ~1.47 GHz).
- **Not compared.** The LP01 study's emission result was for two boards and the flex, a 600 mm assembly, so it
  is not a like-for-like check.

**Model and check.**
- The bus and buck models pass the check.
- The check caught three problems:
  - the dropped-sheet bug, in an earlier build of the bus model;
  - a VCORE neck that opens at a 0.1 mm mesh;
  - switch-node clearances that short at 0.25 mm.
- The buck's NaN (a lumped 3.3 µH L) was not caught before the run. The check now fails it.

**Rendering.** The bus run's four field planes, rendered with `scene.py` on an RTX 3080: 300 frames at 1080p.

**The report.** `report.py` built the validation PDF from SUMMARY.md, four runs' findings and this file.

**Pending.**
- A PDN run. The VCORE model still shorts to GND at a 0.05 mm mesh.
- The cloud render wrapper, `make_render_job.py`, on gpu-run.
- An emission comparison with a chamber measurement.
