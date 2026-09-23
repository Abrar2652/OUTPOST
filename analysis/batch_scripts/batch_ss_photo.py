"""SimSample on Photo — the falsifiable prediction test.

The Phase-0 law says SimSample should help where anomalies are SCATTERED and
neighbourhoods are full of camouflage edges (Yelp: same-frac 0.16, avg degree
~167, +0.051 PR). Photo's anomalies are CLUSTERED (same-frac 0.56-0.94) and the
graph is far sparser, so purifying neighbourhoods has little left to fix: the
law predicts a small or neutral effect here, NOT a second large gain.

This is a genuine prediction, stated before running. If SimSample gives a big
Photo gain too, the "camouflage-edge" explanation is wrong (it would just be a
generically better sampler) and the paper's mechanism story must be revised.

Baseline (Photo full config, corrected yaml, gate off, matched seeds):
    PR 0.6417 / 0.5326 / 0.5375, mean 0.5706 ; ROC mean 0.8640
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

BASE = {"use_fview_gate": False, "sim_topk_frac": 1.0}
JOBS = [
    ({**BASE}, "SSP_photo_stream42", "photo simsample stream-42"),
    ({**BASE, "seed": 42, "train_seed": 0}, "SSP_photo_t0", "photo simsample t0"),
    ({**BASE, "seed": 42, "train_seed": 1}, "SSP_photo_t1", "photo simsample t1"),
]

res = os.path.join(HERE, "batch_ss_photo.results")
with open(res, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== simsample-photo {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} | baseline mean PR 0.5706 "
             f"(0.6417/0.5326/0.5375) ===\n")

for ov, tag, note in JOBS:
    t0 = time.time()
    try:
        p = subprocess.run([PY, "driver.py", json.dumps(ov), tag],
                           cwd=HERE, capture_output=True, text=True, timeout=6 * 3600)
        tail = [l for l in p.stdout.strip().splitlines() if "AGG" in l]
        out = " || ".join(tail[-2:]) if tail else f"(no summary; rc={p.returncode})"
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    line = f"[{tag}] ({note}) {out} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
print("[batch] simsample-photo done", flush=True)
