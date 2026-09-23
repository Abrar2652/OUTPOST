"""Placebo control + confound disentangling for SimSample.

`sim_topk_frac=1.0` changes TWO things at once: (a) neighbour purity, and
(b) sampling determinism (no randomness left in which neighbours are drawn).
Both the Yelp gain (+0.051 PR) and the Photo damage (-0.081 PR) are therefore
confounded. Two experiments separate them:

C1  Yelp PLACEBO: sim_shuffle=True, frac=1.0 — identical determinism, but the
    neighbour ordering is random instead of similarity-based. If this also
    gains ~+0.05, the effect was never about neighbour purity and the
    mechanism story is wrong. If it is flat/negative, purity is the cause.

C2  Photo frac=0.5: keeps half the sampling stochastic. If the -0.081 damage
    largely disappears, Photo was hurt by lost stochasticity (which our own
    calibration found to be the key driver there), not by purity per se.

References:
  Yelp  baseline 0.3520 PR | SimSample1.0 0.4025 PR
  Photo baseline 0.5706 PR | SimSample1.0 0.4898 PR
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

YBASE = {"use_mixup": False, "use_halo": False, "use_fview_gate": False,
         "hidden_dim": 32}
JOBS = []
# C1: Yelp placebo (random ordering, same determinism)
for s in (42, 0, 1):
    JOBS.append(("driver_yelp.py",
                 {**YBASE, "sim_topk_frac": 1.0, "sim_shuffle": True, "seed": s},
                 f"C1_yelp_placebo_s{s}", f"PLACEBO random-order seed {s}"))
# C2: Photo at frac=0.5 (retains half-stochastic sampling)
for tag, extra in (("stream42", {}), ("t0", {"seed": 42, "train_seed": 0}),
                   ("t1", {"seed": 42, "train_seed": 1})):
    JOBS.append(("driver.py",
                 {"use_fview_gate": False, "sim_topk_frac": 0.5, **extra},
                 f"C2_photo_ss50_{tag}", f"photo frac=0.5 {tag}"))

res = os.path.join(HERE, "batch_control.results")
with open(res, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== control batch {time.strftime('%Y-%m-%d %H:%M')} "
             f"keepawake={_awake} | yelp base 0.3520 / SS1.0 0.4025 ; "
             f"photo base 0.5706 / SS1.0 0.4898 ===\n")

for driver, ov, tag, note in JOBS:
    t0 = time.time()
    try:
        p = subprocess.run([PY, driver, json.dumps(ov), tag],
                           cwd=HERE, capture_output=True, text=True, timeout=6 * 3600)
        keep = [l for l in p.stdout.strip().splitlines()
                if "BEST" in l or "AGG" in l]
        out = " || ".join(keep[-2:]) if keep else f"(no summary; rc={p.returncode})"
    except Exception as e:
        out = f"FAILED: {type(e).__name__}: {e}"
    line = f"[{tag}] ({note}) {out} | {time.time() - t0:.0f}s"
    print(line, flush=True)
    with open(res, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
print("[batch] control done", flush=True)
