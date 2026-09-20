"""Generate the multi-seed sensitivity campaign.

METHODOLOGY section 8 reports four hyperparameters over thirteen measured points
and then says the plain truth about them: "With 1-3 runs per point against
+-0.03-0.07 variance, the lambda_un and alpha_plus curves are indicative only. A
publication-grade sensitivity figure requires multi-seed sweeps that have not
been run." This is that sweep.

Every point gets the same five seeds as the main table, so each is a cell with a
standard deviation rather than a single draw, and the curve can be read as a
curve instead of as a sequence of coin flips. The default value of each
hyperparameter is NOT re-run: the A_main rows already are that point, at those
seeds, and analysis/scripts/stats.py joins them in.

  yelp  hidden_dim   16, 64          (32 is the default -> A_main)
  yelp  alpha_plus   0.01, 0.10      (0.05 is the default -> A_main)
  photo lambda_un    0.5, 1.25, 1.5  (1.0 is the default -> A_main)

Run after the core queue, on whichever GPU is free:

    python analysis/scripts/run_campaign.py campaigns/sensitivity.json --gpus 7
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = [42, 0, 1, 2, 3]
jobs = []

for h in [16, 64]:
    for s in S:
        jobs.append({"dataset": "yelp", "seed": s, "tag": f"D_hidden{h}",
                     "priority": 1, "set": {"hidden_dim": h}})
for ap in [0.01, 0.10]:
    for s in S:
        jobs.append({"dataset": "yelp", "seed": s, "tag": f"D_alpha{ap}",
                     "priority": 2, "set": {"alpha_plus": ap}})
for lu in [0.5, 1.25, 1.5]:
    for s in S:
        jobs.append({"dataset": "photo", "seed": s, "tag": f"D_lambda{lu}",
                     "priority": 3, "set": {"lambda_un": lu}})

json.dump({"_comment": __doc__, "seeds": S, "jobs": jobs},
          open("campaigns/sensitivity.json", "w"), indent=1)
print(len(jobs), "sensitivity jobs")
