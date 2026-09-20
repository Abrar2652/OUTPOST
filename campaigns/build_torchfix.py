"""Re-run every seed that predates the mid-campaign torch upgrade.

torch was upgraded from 2.0.1+cu117 to 2.7.1+cu126 on 2026-08-11, by someone
else, in shared site-packages (analysis/REVIEWER_PROOFING.md section 13). Six
result cells ended up with seeds on both sides of it, including Photo's main-table
cell and BOTH arms of the Computers paired comparison - the strongest paired win
in the paper. No shift was detectable, but with 1-5 seeds a side that establishes
only "not visible at this power".

The fix is to re-run the pre-upgrade seeds under the current environment so every
cell is single-version. It cannot be done the other way round: the old torch is
gone from the shared installation.

  computers  A_main   seeds 1, 42        computers  B_demo   seeds 2, 3
  photo      A_main   seeds 0,1,2,3,42   photo      E_demo_energy seeds 0, 1
  photo      T_hidden32 seed 42          cs         A_main   seeds 0,1,2,42

68 rotation-jobs. The re-run values WILL differ from the originals - that is the
point, not a problem - and results.csv keeps the newest row per
(dataset, method, seed, train_seed, tag), so re-running replaces rather than
duplicates.

Ordering: Photo first (cheap, and it carries a main-table cell), then Computers
(the paired comparison), then cs (most expensive, and its cell is the least
affected - 4 of 5 seeds pre-upgrade but the measured pre/post gap is 0.0011).
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEAN = {"use_atlas_gate": False, "use_pl": False}
jobs = []
for s in [0, 1, 2, 3, 42]:
    jobs.append({"dataset": "photo", "seed": s, "tag": "A_main", "priority": 1,
                 "shard_rotations": True})
for s in [0, 1]:
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E_demo_energy", "priority": 1,
                 "shard_rotations": True, "set": {"energy_loss": True}})
jobs.append({"dataset": "photo", "seed": 42, "tag": "T_hidden32", "priority": 1,
             "shard_rotations": True, "set": {"hidden_dim": 32}})
for s in [1, 42]:
    jobs.append({"dataset": "computers", "seed": s, "tag": "A_main",
                 "priority": 2, "shard_rotations": True})
for s in [2, 3]:
    jobs.append({"dataset": "computers", "seed": s, "method": "demo",
                 "tag": "B_demo", "priority": 2, "shard_rotations": True,
                 "set": {"energy_loss": False}})
for s in [0, 1, 2, 42]:
    jobs.append({"dataset": "cs", "seed": s, "tag": "A_main", "priority": 3,
                 "shard_rotations": True})
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/torchfix.json", "w"), indent=1)
print(f"{len(jobs)} jobs before sharding")
