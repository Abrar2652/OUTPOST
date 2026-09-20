"""Complete the synthesis-crossover evidence: full-config Yelp at seeds 0 and 1.

The paper's central claim is an INTERACTION: anomaly synthesis helps where
anomalies are clustered (Photo, same-frac 0.56-0.94) and hurts where they are
scattered (Yelp, 0.16). Photo has 3 paired seeds; Yelp had only one (full
config was only ever run at seed 42), so the Yelp half rested on n=1.

These two runs give 3 paired seeds on both sides:

  yelp full + gate, h=64:  s42 = 0.3012 (have), s0 = ?, s1 = ?
  yelp lean + gate, h=64:  s42 = 0.3293, s0 = 0.4079, s1 = 0.3426 (have)

Config matches the original full-config Yelp run (commit d101c5e): synthesis
ON, spectral gate ON, hidden_dim 64 — only the seed varies.
"""
import ctypes
import json
import os
import subprocess
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

FULL = {"use_mixup": True, "use_halo": True, "use_fview_gate": True,
        "hidden_dim": 64}
JOBS = [
    ({**FULL, "seed": 0}, "X_yelp_full_s0", "crossover: full-config Yelp seed 0"),
    ({**FULL, "seed": 1}, "X_yelp_full_s1", "crossover: full-config Yelp seed 1"),
]

res_path = os.path.join(HERE, "batch_crossover.results")
with open(res_path, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== crossover batch {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} ===\n")

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
    with open(res_path, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("[batch] crossover done", flush=True)
