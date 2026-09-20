"""Generate the GPU-6 queue: the two OGB graphs, now that cs has finished.

cs owned this card until it completed (5 seeds, 40 rotation-jobs). The OGB graphs
inherit it because they are the remaining bulk of the matrix and because they are
sharded per rotation, so they parallelise where cs could not: 128-dimensional
features put a job at 3-4 GB, and several fit at once.

  ogbn-arxiv   4 rotations x 5 seeds  = 20 jobs, ~45 min each   (tests P6)
  ogbn-mag    15 rotations x 3 seeds  = 45 jobs, ~100 min each  (tests P5)

Three seeds on ogbn-mag rather than five: 15 rotations x 400 epochs is ~25
GPU-hours per seed, and it is the graph where every published method already sits
at chance - the pre-registered prediction (analysis/tables/prediction_ogb.md, P5)
is that OUTPOST will too.

arxiv first: it is a quarter of the cost and its prediction (P6, parity with the
published field rather than chance) is the more informative of the two.

These bands were moved OUT of campaigns/gpu7.json when this file was created. Two
runners must never hold jobs for the same dataset, or both can dispatch the same
rotation at once - the shard-resume check only skips work that has already
FINISHED.
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = []

for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "ogbn-arxiv", "seed": s, "tag": "A_main",
                 "priority": 1, "shard_rotations": True})

# Computers moved here from campaigns/gpu7.json once arxiv finished. It is the
# highest-value work left: it completes the third paired OUTPOST-vs-DEMO cell,
# and at 2 seeds it is the one dataset where OUTPOST leads its re-run baseline by
# a wide margin (+0.056 ROC, +0.108 PR). Sharded by rotation - five rotations of
# ~1 h each rather than one 5-hour job - so an interruption costs a rotation.
# It must NOT also appear in the GPU-7 queue: the shard-resume check only skips
# work that has FINISHED, so two runners could dispatch the same rotation.
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "computers", "seed": s, "tag": "A_main",
                 "priority": 2, "shard_rotations": True})
    jobs.append({"dataset": "computers", "seed": s, "method": "demo",
                 "tag": "B_demo", "priority": 2, "shard_rotations": True,
                 "set": {"energy_loss": False}})
    jobs.append({"dataset": "computers", "seed": s, "tag": "C_sim",
                 "priority": 3, "shard_rotations": True,
                 "set": {"sim_topk_frac": 1.0}})
# ogbn-mag is NOT queued. Measured on this hardware: 11.8 GB peak and 77 s per
# post-warmup epoch, so one rotation of the 400-epoch protocol is 8.6 hours and
# one seed over 15 rotations is 128 GPU-hours (73 at batch_size 1024, which is
# numerically inert here but does not change the verdict). Three seeds would be
# most of a fortnight on this card.
#
# It is deliberately NOT run at a reduced epoch budget either. P5 predicts
# OUTPOST stays below 0.60 on ogbn-mag, and fewer epochs under best-over-epochs
# selection can only lower the score - a shortened run would bias the test toward
# confirming our own prediction. See analysis/tables/prediction_ogb.md.

json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/gpu6.json", "w"), indent=1)
print(f"{len(jobs)} jobs before rotation sharding")
