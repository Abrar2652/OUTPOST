"""The two experiments the review said were compute-fixable.

A. THE CRITERION CURVE, WITHIN ONE GRAPH  (P17-P19, prediction_curve.md)
   The SimSample effect is non-monotone across datasets and three mechanisms for
   it have been falsified. With five points from five different graphs, a referee
   can call the ordering confounded with graph identity. Photo spans the whole
   range on its own once the budget is varied - 85.1% of nodes above budget at
   b=5, 3.8% at b=100 - so the cross-dataset pattern can be tested with nothing
   varying but the budget. b=25 is already measured.

B. THE DEMO REPRODUCTION GAP  (diagnostic, no prediction registered)
   Two independent implementations land ~0.07 below published DEMO on Photo,
   with the config verified key-for-key against the authors' repo. The Devil's
   Advocate calls this CRITICAL: "you beat your reproduction of DEMO, not DEMO."
   This varies one protocol quantity at a time to see whether ANY plausible
   setting reaches their 0.9023 - a larger label budget, more normals, or the
   longer epoch budget. A null result is still worth having: it bounds the gap
   to something not reachable by the protocol knobs we can guess, which is a
   more defensible sentence than "unexplained".

Photo is the right vehicle for both - two rotations, ~13 min each - so the whole
thing is ~20 GPU-hours. Runs on GPUs 3-5; ogbn-mag keeps 6-7.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = []
# A. criterion curve
for b in [5, 10, 50, 100]:
    for frac in [1.0, 0.0]:
        for s in [42, 0, 1, 2, 3]:
            jobs.append({"dataset": "photo", "seed": s, "tag": f"G_b{b}_sim{frac}",
                         "priority": 1, "mem_gb": 15.0,
                         "set": {"sampling_sizes": [b, 10], "sim_topk_frac": frac}})
# B. DEMO protocol sensitivity, full published DEMO each time
for tag, st in [("H_anom100",  {"train_anormaly_num": 100}),
                ("H_anom200",  {"train_anormaly_num": 200}),
                ("H_norm0.10", {"train_normal_ratio": 0.10}),
                ("H_norm0.20", {"train_normal_ratio": 0.20}),
                ("H_ep400",    {"num_epochs": 400})]:
    for s in [42, 0, 1]:
        j = {"dataset": "photo", "seed": s, "method": "demo", "tag": tag,
             "priority": 2, "mem_gb": 16.0, "set": dict(st)}
        j["set"].update({"mixup": True, "energy_loss": False})
        jobs.append(j)
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/push.json", "w"), indent=1)
print(f"{len(jobs)} jobs (A: {4*2*5} curve, B: {5*3} DEMO diagnostic)")
