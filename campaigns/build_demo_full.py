"""DEMO as published - mixup ON - for every dataset carrying a paired claim.

Every DEMO arm run so far had `mixup: False`, the setting the authors' repo ships
with. Mixup is DEMO's headline contribution, so those arms are the baseline's own
`w/o Mix` ablation. Reporting them as "DEMO" in a paired comparison is a rejection
risk and would deserve to be one.

The rule adopted: a baseline we make a PAIRED claim about must be the full
published method, or we make no paired claim. Concretely:

  yelp   run here, mixup ON      PPR 8.4 GB float32, built in 74 s
  photo, computers, cs           run by Farhan on Kaggle with mixup ON
  ogbn-arxiv                     PPR is 115 GB dense - possible in host RAM,
                                 not attempted; no paired claim
  ogbn-mag                       PPR is 2.2 TB - structurally impossible; no
                                 paired claim, and worth saying so in the paper

Yelp gets the same ten seeds as its OUTPOST arm and the same 400-epoch protocol
budget, so the pairing is on identical splits. The existing mixup=False arm is
kept, retagged in analysis as DEMO w/o Mix: having both gives the mixup ablation
of the baseline for free, which is worth reporting alongside.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = [{"dataset": "yelp", "seed": s, "method": "demo", "tag": "B_demo_mix",
         "priority": 1, "mem_gb": 16.0,
         "set": {"mixup": True, "energy_loss": False, "num_epochs": 400}}
        for s in [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/demo_full.json", "w"), indent=1)
print(f"{len(jobs)} jobs")
