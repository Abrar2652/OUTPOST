"""Stage 2, CS half: the six CS runs of the pre-registered SimSample test.

Split out from batch_predict_test.py because the Computers half is being run on
Colab (T4) while CS runs here: CS is 8 rotations of 6805-dim features, ~2.1 h per
run locally, and free Colab reclaims its VM long before six of those finish.

Identical protocol to batch_predict_test.py -- same tags, same overrides, same
driver. Only the dataset filter differs, plus resumability so an interrupted
overnight run continues instead of restarting.

Pre-registered prediction (committed ad25c69, before any of this ran):
  cs -> NEUTRAL (small |effect|), rule R3 features already separate
Reference effects: yelp +0.051 PR, photo -0.081 PR.
"""
import ctypes
import json
import os
import re
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
for arm, extra in (("base", {}), ("ss", {"sim_topk_frac": 1.0})):
    for stag, sd in SEEDS:
        JOBS.append(("cs", {"use_fview_gate": False, **extra, **sd},
                     f"P_cs_{arm}_{stag}", f"cs {arm} {stag}"))

res = os.path.join(HERE, "batch_predict_test.results")

# ---- resume: skip tags that already have a successful summary ---------------
done = set()
if os.path.exists(res):
    for line in open(res, encoding="utf-8"):
        m = re.match(r"\[(P_\w+)\]", line.strip())
        if m and "AGG best" in line:
            done.add(m.group(1))
print(f"[resume] {len(done)} runs already complete "
      f"({sorted(t for t in done if t.startswith('P_cs'))} for cs)", flush=True)

with open(res, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== prediction test CS half {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} | resumed, {len(done)} done "
             f"| PREDICTED: cs NEUTRAL ===\n")

for ds, ov, tag, note in JOBS:
    if tag in done:
        print(f"[skip] {tag} already done", flush=True)
        continue
    t0 = time.time()
    try:
        p = subprocess.run([PY, "driver_generic.py", ds, json.dumps(ov), tag],
                           cwd=HERE, capture_output=True, text=True,
                           timeout=12 * 3600)
        keep = [l for l in p.stdout.strip().splitlines() if "AGG" in l]
        out = (" || ".join(keep[-2:]) if keep
               else f"(no summary; rc={p.returncode}) {p.stderr.strip()[-300:]}")
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    line = f"[{tag}] ({note}) {out} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
print("[batch] CS half done", flush=True)
