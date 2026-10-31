# The kicad-ems prompt

Paste the block below into Claude Code, or another agent with a shell, started in your KiCad project's folder.
Before you do:
- clone this repo next to the project;
- set up `gpu-run` (the top README, *One-time setup*);
- make the venv (kicad-ems/README.md, *Setup*).

Edit the first three lines.

The agent works on copies and never edits your board. It asks you before every step that costs money, and gives
the estimate first.

---

```text
You are running a full-wave electromagnetic review of a KiCad board with kicad-ems.

BOARD:        <path to the .kicad_pcb>             (read only: never modify, save or refill it)
REPO:         <path to the openems-gpu clone>       (kicad-ems/ and gpu-run are in it)
BUDGET:       $<n> for cloud GPU runs                (a hard cap; ask before each paid step)

Read REPO/kicad-ems/README.md and REPO/kicad-ems/METHOD.md first. They hold the commands, the modelling rules and
the limits. Then work through these phases. Stop at each "ASK" and wait for my answer.

1. TRIAGE
   - Copy the board into study/board/. Run kems.triage on the copy into study/. Read TRIAGE.md and its figures.
   - Read the schematic too, if it is there: the .kicad_sch next to the board.
   - Use the schematic and the parts' data sheets to fill in what the board cannot tell you, for each candidate:
     - the driver part and its drive strength (output resistance);
     - the receivers and their input capacitance;
     - the edge rate, or the fastest a part can make it;
     - the clock or switching frequency;
     - for switchers: the input voltage, duty and switch-node edge.
     Write these into study/assume.json and re-run triage with --assume. Say where every number came from:
     data sheet and page, or "assumed".
   - If triage lists nets that are open in copper but pass DRC (jumper pad groups), report them first, as
     faults. Confirm each one with kicad-cli DRC on a copy with the groups removed.
   - ASK. Present a short ranked table, at most ~12 rows: line, why it is a risk (edge vs length, reference gaps
     and changes, return-via distance, board edge, switch node), and the study you propose for it:
     - SI: waveforms at the receivers;
     - EMI: radiated emission against FCC 15.109 / CISPR 32 Class B;
     - both.
     Then list the power nets with their decoupling. Ask me:
     (a) which lines to simulate;
     (b) "Do you want power-plane (PDN impedance) analysis? If yes, which rails, from which load pins, and what
         target impedance?";
     (c) whether the board ships in an enclosure that should be modelled. If yes, ask for its inner dimensions,
         material and how the board is mounted.

2. MODELS (free)
   - Write one study/lines/<NAME>.json per agreed line or bus, from REPO/kicad-ems/examples/:
     - study.json for SI: edge excitation, with td_planes if I may want a render;
     - study_emi.json for emission: Gaussian, far field on, region "board";
     - study_pdn.json for PDN.
   - Model a bus together: the driven line plus its neighbours, held by their drivers, so crosstalk shows.
   - Build each with kems.model, then run kems.check --png. Look at the mesh image.
   - A check failure is a modelling problem; fix it before anything else:
     - a net in pieces;
     - a short to another net;
     - a port off its copper;
     - a sheet off its mesh line;
     - a series R-L-C.
     The usual fixes: a finer "fine" / "region" mesh near the narrow clearances, a smaller region, or a different
     driver pad. Never paper over a failure.
     One exception: a net in pieces that kicad-cli DRC also reports as unconnected is a fault on the board, not
     in the model. Report it as a finding and leave that net out of the model.
   - Keep each model under ~60 M cells unless I agree otherwise.

3. ASK BEFORE PAYING
   - Show a table, one row per job: cells, steps, and minutes and dollars on each GPU class (from the check),
     plus the total.
   - Propose the smoke runs, and the full runs after them. Default to the fastest finish within the budget:
     several jobs in parallel with gpu-run sweep, RunPod Secure Cloud only.
   - Give the exact gpu-run commands with --budget and --max-min. Wait for my "yes".

4. RUN
   - Package with kems.make_job: --smoke first, then the full job.
   - Run the smoke jobs. Read smoke.json. A failed smoke run means a broken model: an open port reads
     u / (R i) in the hundreds. Fix it before the full runs; never "retry and hope".
   - Run the full jobs under the agreed budget.
   - Keep a run table (job, GPU, solver seconds, wall, $) in study/RUNS.md.
   - Never put credentials in a job, a log or a note.

5. RESULTS
   - Post-process each run with kems.post (si / emi / pdn). Put each run's out/ folder under study/runs/<NAME>/.
   - Cross-check every headline number with a quick independent estimate before you believe it:
     - delays against length / (c / sqrt(er_eff));
     - Z0 against a closed form;
     - crosstalk against a coupled-line estimate;
     - for emission, the dominant mechanism: which structure is the antenna.
     When they disagree by more than ~2x, find out why before reporting.
   - Write study/SUMMARY.md, in this order:
     1. the faults, ranked by risk to the product;
     2. for each fault: the number, the limit it is judged against and its source, the mechanism, and a concrete
        change to the layout or firmware;
     3. what passed, briefly;
     4. what was not modelled.
     Be plain about the limits: emission margins under ~6 dB mean "measure it". Do not apply any change to the
     board.
     Keep it about this board. Leave out kicad-ems itself: its benchmarks, GPU prices, validation results, where its
     rules came from, and anything else from METHOD.md that isn't needed to read a number here.
   - Build the PDF with kems.report study/ and give me its path.

6. RENDER (optional; ask)
   - ASK whether I want an animation of a line. If yes:
     1. Use a run with an edge excitation and td_planes; build textures with kems.render.textures.
     2. Write a shot.json from examples/shot.json: overhead while the edge runs, then down into the layers.
     3. Build the .blend locally with scene.py, without rendering. I can open it to frame the shot.
     4. Quote the render (GPU class: Ada or older; Blender 4.1 has no Blackwell kernels), and wait for my yes.
     5. Run make_render_job --preview first, then the full render.

Rules throughout:
- Keep originals untouched; work on copies under study/.
- Do not confuse a file on disk with a verified result: say what was run and what was only built.
- Report failures as failures, with the log line.
```
