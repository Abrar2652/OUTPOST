"""The CS arms that a 24 GB card could not hold.

The campaign moved from ckg10 (RTX A5000, 24.5 GB) to ckg12 (RTX A6000, 49 GB).
Three CS arms were recorded in METHODOLOGY section 10 as out of reach on the
smaller card, with measured requirements:

  DEMO on cs              ~22 GB   the paired cs comparison
  OUTPOST + SimSample     ~25 GB   P14, the test of section 5.1's mechanism

Both fit 49 GB comfortably. P14 is the more valuable of the two: ogbn-arxiv
refuted the subsampling-removal explanation for why SimSample harms sparse graphs
(+0.0008 measured against -0.083 predicted), and cs is the discriminating case -
89% of its nodes sit in the band where subsampling could be removed, against
Photo's 52.4%, so the mechanism predicts cs should show the LARGEST harm of any
dataset. The prediction is registered in analysis/tables/prediction_budget.md and
was written before this hardware existed.

Runs on GPUs 3-5, which are idle; 6 and 7 are carrying cs_open and torchfix.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = []
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "cs", "seed": s, "tag": "C_sim", "priority": 1,
                 "shard_rotations": True, "mem_gb": 27.0,
                 "set": {"sim_topk_frac": 1.0}})
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "cs", "seed": s, "method": "demo", "tag": "B_demo",
                 "priority": 2, "shard_rotations": True, "mem_gb": 24.0,
                 "set": {"energy_loss": False}})
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/unblocked.json", "w"), indent=1)
print(f"{len(jobs)} jobs before sharding ({len(jobs)*8} rotation-jobs)")
