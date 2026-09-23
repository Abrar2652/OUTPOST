"""ogbn-mag, one seed, full protocol - the last empty main-table cell.

Deferred throughout because on the RTX A5000 it was 77 s per post-warmup epoch,
i.e. 128 GPU-hours per seed. Re-measured on the A6000: 45 s/epoch, 75 GPU-hours
per seed, and five idle cards make that ~15 h wall-clock. The agreed condition -
run one seed if time remains after the machine-confound re-run, the cs DEMO arm
and the hyperparameter sweeps - is now met; all three are complete.

Run at the FULL 400-epoch protocol budget, deliberately. A shortened run is
cheaper and would bias the result toward confirming our own pre-registered
prediction (P5 says OUTPOST stays below 0.60, and fewer epochs under
best-over-epochs selection can only lower the score). See
analysis/tables/prediction_ogb.md.

This tests P5 (mag is at chance for OUTPOST too, as it is for every published
method) and P8 (class 263 remains the best-detected, being feature-visible rather
than reachable by propagation). One seed cannot support a variance claim and is
not meant to - the cell exists to be reported with n=1 stated, or to falsify P5.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = [{"dataset": "ogbn-mag", "seed": 42, "tag": "A_main", "priority": 1,
         "shard_rotations": True, "mem_gb": 14.0}]
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/mag.json", "w"), indent=1)
print(f"{len(jobs)} job -> 15 rotation-jobs")
