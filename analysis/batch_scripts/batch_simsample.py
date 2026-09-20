"""Similarity-ordered neighbour sampling (SimSample) — 3 seeds x 2 strengths.

Zero-parameter mechanism: neighbours are pre-sorted by feature cosine
similarity, and `sim_topk_frac` of each hop's budget is filled from the most
similar ones instead of uniformly. Targets the measured cause of the Yelp
ceiling (dense camouflage edges averaging scattered fraud into normality).

Pre-flight mechanism check (already passed): sampled-neighbour label agreement
0.8590 (uniform) -> 0.8784 (frac 0.5) -> 0.8861 (frac 1.0).

Baseline to beat — h=32, gate OFF, lean, 3 seeds:
    PR 0.3297 / 0.3803 / 0.3459, mean 0.3520 ; ROC mean 0.7344
Params unchanged at 7,361 (0.22x DEMO) — the mechanism adds none.
"""
import ctypes
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Farhan\miniconda3\envs\py311\python.exe"

try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    _awake = True
except Exception:
    _awake = False

BASE = {"use_mixup": False, "use_halo": False, "use_fview_gate": False,
        "hidden_dim": 32}
JOBS = [({**BASE, "sim_topk_frac": f, "seed": s}, f"SS{int(f*100)}_yelp_s{s}",
         f"simsample frac={f} seed={s}")
        for f in (0.5, 1.0) for s in (42, 0, 1)]

res = os.path.join(HERE, "batch_simsample.results")
with open(res, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== simsample batch {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} | baseline mean PR 0.3520 "
             f"(0.3297/0.3803/0.3459) ===\n")

for ov, tag, note in JOBS:
    t0 = time.time()
    try:
        p = subprocess.run([PY, "driver_yelp.py", json.dumps(ov), tag],
                           cwd=HERE, capture_output=True, text=True, timeout=6 * 3600)
        tail = [l for l in p.stdout.strip().splitlines() if "BEST" in l]
        out = tail[-1] if tail else f"(no summary; rc={p.returncode})"
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    line = f"[{tag}] ({note}) {out} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
print("[batch] simsample done", flush=True)
