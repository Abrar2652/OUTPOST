"""Generate the DEMO energy-gradient campaign.

METHODOLOGY section 4 records that DEMO's energy-gradient reweighting was never
run to completion because it needs per-training-node second-order gradients, and
that every reproduced DEMO number therefore excludes it. As written that is an
assertion about a baseline's missing component, which is exactly the kind of
thing a referee will not take on trust: a baseline that was not run at full
strength is a baseline that was weakened.

This campaign turns the assertion into two measurements on Photo, the cheapest
dataset (2 rotations, 200 epochs, ~380 labelled training nodes per rotation):

  E_demo_energy    DEMO with energy_loss ON, three seeds, run to completion
  E_demo_noenergy  the same three seeds with it OFF

These arms now live in campaigns/gpu7.json (band 4) rather than in a runner of
their own: one runner per GPU, or their memory budgets double-book the card.
This script is kept because it records why the experiment exists.

The pair answers both halves of the objection - what the component costs, and
whether it would have changed the comparison - rather than only the first.

`compute_beta` takes a batch_size=1 loader over the training set and does a
create_graph backward per node per epoch, so the cost scales with labelled
training nodes x epochs. Photo is the only dataset where that is affordable;
for the others the measured per-epoch ratio from here is the reported figure.
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = []
for s in [42, 0, 1]:
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E_demo_energy", "priority": 1,
                 "set": {"energy_loss": True}})
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E_demo_noenergy", "priority": 1,
                 "set": {"energy_loss": False}})

json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/energy.json", "w"), indent=1)
print(len(jobs), "jobs")
