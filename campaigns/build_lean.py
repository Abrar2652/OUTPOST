"""Run the method the paper intends to describe, as one configuration.

At 10 matched seeds on Yelp the atlas gate is +0.0011 (CI [-0.0004, +0.0029])
and pseudo-labelling off is +0.0038 (CI [-0.0010, +0.0089]) - both null. The
honest paper therefore describes a smaller method than `config.json` implements.
But a described method must be a RUN method: reporting `A_main` numbers while
describing a system with two components removed would be reporting a
configuration nobody executed, and adding up single-flag ablations is not the
same as running the combination.

There is a specific reason to expect the combination to differ. The conformal
threshold acts only inside the pseudo-label branch - `thr_a` is computed under
`if use_pl` - so `use_pl=False` also silently disables the one small component
that does measurably help (turning conformal off costs -0.0028). Gate-off and
PL-off have never been run together, and their joint effect may not be the sum
of the separate ones.

  L_lean    use_atlas_gate=False, use_pl=False   (both nulls removed)
  L_nogate  use_atlas_gate=False                 (already have this as C_nogate)

Yelp and Photo first, at the seed counts their A_main cells use, because they are
cheap and decide the question. Computers, CS and ogbn-arxiv follow only if the
lean configuration holds up.

    python campaigns/build_lean.py
    python analysis/scripts/run_campaign.py campaigns/lean.json --gpus 7
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEAN = {"use_atlas_gate": False, "use_pl": False}
jobs = []

for s in [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
    jobs.append({"dataset": "yelp", "seed": s, "tag": "L_lean",
                 "priority": 1, "set": LEAN})
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "photo", "seed": s, "tag": "L_lean",
                 "priority": 2, "set": LEAN})

json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/lean.json", "w"), indent=1)
print(f"{len(jobs)} jobs")
