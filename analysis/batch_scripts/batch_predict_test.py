"""Stage 2: test the PRE-REGISTERED SimSample predictions on Computers and CS.

Predictions (committed ad25c69, before this ran):
  computers -> HURTS (large negative), rule R2 clustered anomalies
  cs        -> NEUTRAL (small |effect|), rule R3 features already separate

Neither dataset has ever been run through OUTPOST, so each needs a matched
baseline AND a SimSample arm. 2 datasets x 2 arms x 3 matched seeds = 12 runs.
Matched seeds use the same split/train-seed scheme as the Photo experiments.

Computers has 5 anomaly-class rotations and CS has 8, so each run is several
rotations; CS runs are the slowest here.
"""
import ctypes
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "python"

try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    _awake = True
except Exception:
    _awake = False

SEEDS = [("stream42", {}), ("t0", {"seed": 42, "train_seed": 0}),
         ("t1", {"seed": 42, "train_seed": 1})]

JOBS = []
for ds in ("computers", "cs"):
    for arm, extra in (("base", {}), ("ss", {"sim_topk_frac": 1.0})):
        for stag, sd in SEEDS:
            JOBS.append((ds, {"use_fview_gate": False, **extra, **sd},
                         f"P_{ds}_{arm}_{stag}", f"{ds} {arm} {stag}"))

res = os.path.join(HERE, "batch_predict_test.results")
with open(res, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== prediction test {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} | PREDICTED: computers HURTS, cs NEUTRAL ===\n")

for ds, ov, tag, note in JOBS:
    t0 = time.time()
    try:
        p = subprocess.run([PY, "driver_generic.py", ds, json.dumps(ov), tag],
                           cwd=HERE, capture_output=True, text=True, timeout=8 * 3600)
        keep = [l for l in p.stdout.strip().splitlines() if "AGG" in l]
        out = " || ".join(keep[-2:]) if keep else f"(no summary; rc={p.returncode}) {p.stderr.strip()[-200:]}"
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    line = f"[{tag}] ({note}) {out} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
print("[batch] prediction test done", flush=True)
