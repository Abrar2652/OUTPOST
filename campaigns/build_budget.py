"""Generate the sampling-budget sweep on Yelp.

Predictions P10-P12 are pre-registered in analysis/tables/prediction_budget.md,
written before these runs. Read that first: this experiment is built to be able
to destroy the paper's main effect, and the conditions for that are fixed there.

The criterion under test is "SimSample helps iff degree > sampling budget". It
has so far been tested by varying DEGREE across graphs at a fixed budget of 25 -
observational, since degree co-varies with everything else about a graph. This
varies the BUDGET on one fixed graph, so nothing about Yelp changes except the
quantity the criterion names. Yelp's median degree is 168, so the inequality
flips between b=100 and b=200, inside the sweep.

  b in {50, 100, 200} x {sim_topk_frac 1.0, 0.0} x 5 seeds
  b = 25 is already measured at 10 seeds per arm (A_main and C_nosim)

Measured, not extrapolated: peak memory is 1.56 GB at b=50 and 2.69 GB at b=200
against ~1.2 GB at b=25, so these pack several to a card. Largest budget first,
since it is the slowest and the one P11 turns on.
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = [42, 0, 1, 2, 3]
jobs = []

# largest budget first: b=200 is the arm P11 predicts will show no effect, and it
# is the slowest, so it should not be the thing left unfinished
for prio, b in enumerate([200, 100, 50], start=1):
    for frac in [1.0, 0.0]:
        for s in SEEDS:
            jobs.append({
                "dataset": "yelp", "seed": s,
                "tag": f"F_b{b}_sim{frac}", "priority": prio,
                "set": {"sampling_sizes": [b, 10], "sim_topk_frac": frac},
            })

json.dump({"_comment": __doc__, "seeds": SEEDS, "jobs": jobs},
          open("campaigns/budget.json", "w"), indent=1)
print(f"{len(jobs)} jobs ({len(jobs)//len(SEEDS)} arms x {len(SEEDS)} seeds)")
