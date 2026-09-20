"""Close the documented gaps (2026-07-27).

G1  Yelp compression headline (h=32, gate off) needs seed 1 -> 3-seed number
    matching the lean h=64 reference.
G2  Photo multi-seed under the CORRECTED config (the duplicate `lambda_mixup`
    key in data/photo/outpost.yaml silently forced 0.1 instead of the swept
    0.2; removed 2026-07-27). 3 matched training seeds.
G3  Photo lean-vs-full: does removing synthesis help or HURT on a homophilous
    (clustered-anomaly) benchmark? The thesis predicts it HURTS here while it
    helps on scattered real fraud (Yelp) — a falsifiable, directional
    prediction, so this run tests the paper's central claim.

Runs the existing drivers as subprocesses so semantics match all prior runs
exactly; the parent holds a process-scoped keep-awake for the whole batch.
Results appended line-by-line (crash-safe) to batch_gapclose.results.
"""
import ctypes
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Farhan\miniconda3\envs\py311\python.exe"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
try:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    _awake = True
except Exception as e:
    _awake = False
    print(f"[batch] WARN keep-awake failed: {e}")

LEAN = {"use_mixup": False, "use_halo": False}

# (driver, overrides, tag, note)
JOBS = [
    # --- G1: compression headline, 3rd seed ---
    ("driver_yelp.py", {**LEAN, "hidden_dim": 32, "use_fview_gate": False, "seed": 1},
     "G1_yelp_h32_gateOff_s1", "compression headline seed 3/3"),

    # --- G2: Photo full config (corrected lambda_mixup=0.2), matched seeds ---
    ("driver.py", {"use_fview_gate": False},
     "G2_photo_full_stream42", "photo full, corrected cfg, stream-42"),
    ("driver.py", {"use_fview_gate": False, "seed": 42, "train_seed": 0},
     "G2_photo_full_t0", "photo full, corrected cfg, train-seed 0"),
    ("driver.py", {"use_fview_gate": False, "seed": 42, "train_seed": 1},
     "G2_photo_full_t1", "photo full, corrected cfg, train-seed 1"),

    # --- G3: Photo LEAN (synthesis off), same matched seeds ---
    ("driver.py", {**LEAN, "use_fview_gate": False},
     "G3_photo_lean_stream42", "photo lean, stream-42"),
    ("driver.py", {**LEAN, "use_fview_gate": False, "seed": 42, "train_seed": 0},
     "G3_photo_lean_t0", "photo lean, train-seed 0"),
    ("driver.py", {**LEAN, "use_fview_gate": False, "seed": 42, "train_seed": 1},
     "G3_photo_lean_t1", "photo lean, train-seed 1"),
]

res_path = os.path.join(HERE, "batch_gapclose.results")
with open(res_path, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== gap-close batch {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} ===\n")

for driver, ov, tag, note in JOBS:
    t0 = time.time()
    cmd = [PY, driver, json.dumps(ov), tag]
    try:
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=6 * 3600)
        tail = [l for l in p.stdout.strip().splitlines()
                if "AGG" in l or "BEST" in l or "rot" in l]
        out = " || ".join(tail[-3:]) if tail else f"(no summary; rc={p.returncode})"
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    line = f"[{tag}] ({note}) {out} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res_path, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("[batch] gap-close done", flush=True)
