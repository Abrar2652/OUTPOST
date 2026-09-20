"""The two CS experiments that were blocked on GPU memory.

CS needs ~21 GB and both cards were held by another user's inference server from
2026-08-11. They freed on 2026-08-18; this queue is what was waiting.

1. C_sim  - the decisive test of WHY SimSample harms sparse graphs.
   The recorded mechanism is that it removes a dropout-like edge subsampling on
   nodes below the sampling budget. CS has 89.0% of its nodes in the band where
   that could happen (1 < degree <= 25), against Photo's 52.4%, so the mechanism
   predicts CS should show the LARGEST harm of any dataset - larger than Photo's
   -0.0318. ogbn-arxiv already contradicted the mechanism (+0.0008 measured
   against -0.083 predicted, P13 falsified), so if CS also comes out near zero
   the explanation is dead and the harm on Photo/Computers is something else.
   See analysis/tables/prediction_budget.md.

2. L_lean - CS is the only dataset where the lean configuration is untested.
   Yelp prefers lean (+0.0058 ROC, p=0.027); Photo prefers full (-0.0388, 0/9,
   p=0.0039), but three quarters of Photo's preference is extra oracle bonus
   rather than deployable quality. CS is the tie-breaker for whether the
   two-regime story holds: its anomalies are feature-visible, so self-training
   should matter there as it does on Photo.

Sharded by rotation: 8 rotations x ~40 min, so an interruption costs one rotation
rather than a 5-hour job. One CS job per card - it does not share.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = []
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "cs", "seed": s, "tag": "C_sim", "priority": 1,
                 "shard_rotations": True, "set": {"sim_topk_frac": 1.0}})
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "cs", "seed": s, "tag": "L_lean", "priority": 2,
                 "shard_rotations": True,
                 "set": {"use_atlas_gate": False, "use_pl": False}})
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/cs_open.json", "w"), indent=1)
print(f"{len(jobs)} jobs before rotation sharding")
