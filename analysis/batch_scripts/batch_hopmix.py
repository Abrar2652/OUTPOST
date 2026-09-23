"""HopMix evaluation: per-node adaptive fusion of a propagation-free view.

Matched against the efficient baseline (Yelp, lean, h=32, gate OFF, 7361
params) whose 3 seeds gave PR 0.3297 / 0.3803 / 0.3459 (mean 0.3520).
HopMix adds 2,210 params (9,571 total = 0.28x DEMO).

Pass criterion: mean PR must clear the baseline by more than the run-to-run
noise, and w must differ between anomalies and normals (evidence the per-node
adaptivity — not just an extra head — is doing the work).
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

BASE = {"use_mixup": False, "use_halo": False, "use_fview_gate": False,
        "hidden_dim": 32}
JOBS = [({**BASE, "use_hybrid": True, "seed": s}, f"HM_yelp_h32_s{s}",
         f"HopMix seed {s}") for s in (42, 0, 1)]

res = os.path.join(HERE, "batch_hopmix.results")
with open(res, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== hopmix batch {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} | baseline h32 gateOff PR "
             f"0.3297/0.3803/0.3459 mean 0.3520 ===\n")

for ov, tag, note in JOBS:
    t0 = time.time()
    try:
        p = subprocess.run([PY, "driver_yelp.py", json.dumps(ov), tag],
                           cwd=HERE, capture_output=True, text=True, timeout=6 * 3600)
        tail = [l for l in p.stdout.strip().splitlines() if "BEST" in l]
        out = tail[-1] if tail else f"(no summary; rc={p.returncode})"
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    # pull the final learned mixing weights out of the runlog
    wline = ""
    lp = os.path.join(HERE, f"{tag}.runlog")
    if os.path.exists(lp):
        ws = [l for l in open(lp, encoding="utf-8", errors="ignore") if "| w " in l]
        if ws:
            wline = " | final " + ws[-1].split("| w ")[-1].strip()
    line = f"[{tag}] ({note}) {out}{wline} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
print("[batch] hopmix done", flush=True)
