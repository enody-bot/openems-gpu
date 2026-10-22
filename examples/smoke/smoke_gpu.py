#!/usr/bin/env python3
"""T0002 GPU smoke test for openEMS (SeanMollet GPU_merge ee45c6d, HIP on NVIDIA).

Free-space 2.4 m cube, uniform 10 mm cells (240^3 = 13.8 M cells), PML_8 on all six faces,
z-directed 50-ohm lumped port 0.3 m long at the center, Gaussian pulse f0 = 0, fc = 1 GHz,
no end criterion (fixed number of timesteps).

Usage:  python smoke_gpu.py ENGINE NRTS SIM_DIR
  ENGINE: gpu | multithreaded | gpu-reference | ...   (passed to FDTD.Run(engine=...))
Prints one JSON line with the openEMS-reported speed and the port voltage trace summary.
Run it with ~/venvs/ep01-twin/bin/python. Keep SIM_DIR on local ext4, never on /mnt/nas.
"""
import ctypes, ctypes.util, json, os, re, sys, time
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

engine  = sys.argv[1] if len(sys.argv) > 1 else "gpu"
nrts    = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
sim_dir = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else f"smoke_{engine}")

unit = 1e-3                          # drawing unit: mm
L, d = 2400.0, 10.0                  # box edge and cell size, mm
port_len, R = 300.0, 50.0            # port length (mm) and resistance (ohm)

FDTD = openEMS(NrTS=nrts, EndCriteria=0)     # EndCriteria=0: run exactly NrTS steps
FDTD.SetGaussExcite(0, 1e9)                  # f0 = 0, fc = 1 GHz
FDTD.SetBoundaryCond(["PML_8"] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
lines = np.arange(-L / 2, L / 2 + d / 2, d)  # 241 lines -> 240 cells per axis
for ax in "xyz":
    mesh.SetLines(ax, lines)

port = FDTD.AddLumpedPort(1, R, [0, 0, -port_len / 2], [0, 0, port_len / 2], "z", excite=1.0)

# capture the C++ console output of the openEMS library (it writes to fd 1)
log_path = sim_dir + ".log"
sys.stdout.flush()
saved = os.dup(1)
t0 = time.time()
with open(log_path, "w") as log:
    os.dup2(log.fileno(), 1)
    try:
        FDTD.Run(sim_dir, cleanup=True, engine=engine)   # NB: Run() chdirs into sim_dir
    finally:
        try:
            ctypes.CDLL(ctypes.util.find_library("c")).fflush(None)
        except Exception:
            pass
        os.dup2(saved, 1)
        os.close(saved)
wall = time.time() - t0
text = open(log_path).read()

def grab(pat, cast=str, last=False):
    ms = re.findall(pat, text, re.M)
    return cast(ms[-1] if last else ms[0]) if ms else None

ut = np.loadtxt(os.path.join(sim_dir, "port_ut_1"), comments="%")   # columns: t [s], u [V]
dt = grab(r"FDTD timestep is:\s*([0-9.eE+-]+) s", float)
res = {
    "engine_requested": engine,
    "engine_created": grab(r"Create FDTD engine \((.*)\)\s*$"),
    "cells": grab(r"Time for \d+ iterations with ([0-9.]+) cells", lambda v: int(float(v))),
    "timestep_s": dt,
    "iterations": grab(r"Time for (\d+) iterations", int),
    "mcells_per_s": grab(r"Speed:\s*([0-9.]+) MCells/s", float),
    "wall_s_total": round(wall, 1),
    "port_ut_samples": int(len(ut)),
    "port_ut_last": [float(ut[-1, 0]), float(ut[-1, 1])],   # [t s, U V]
    "energy_last_db": grab(r"Energy: ~\S+ \(\s*(-?\s*[0-9.]+)dB\)\s*$", lambda v: float(v.replace(" ", "")), last=True),
    "port_ut_peak_abs_V": float(np.max(np.abs(ut[:, 1]))),
    "log": log_path,
}
if dt:
    res["port_ut_last_step"] = int(round(ut[-1, 0] / dt))
print(json.dumps(res))
