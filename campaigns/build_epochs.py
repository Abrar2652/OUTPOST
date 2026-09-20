"""Both arms at 400 epochs on the small graphs - a protocol correction, urgently.

The reproduction-gap diagnostic found it. Varying one protocol quantity at a time
on Photo with full published DEMO:

  protocol default (200 ep)   0.8403 / 0.5300     gap to published +0.062 / +0.103
  50 -> 100 anomalies         0.8331 / 0.4891     gap widens
  50 -> 200 anomalies         0.7875 / 0.3464     gap widens sharply
  5% -> 10% normals           0.8203 / 0.5209     gap widens
  5% -> 20% normals           0.8161 / 0.5262     gap widens
  **200 -> 400 epochs**       **0.8919 / 0.5923**  **gap +0.010 / +0.041**

Four knobs move DEMO further from its published Photo number; the epoch budget
moves it almost all the way there. METHODOLOGY section 3 sets "200 epochs on
small-scale graphs, 400 on large-scale" and describes that as inherited from
DEMO. That inheritance now looks wrong for the small graphs.

This matters for fairness in BOTH directions and cannot be left half-applied.
Under best-over-epochs selection more epochs can only help, so running DEMO at
400 while OUTPOST stays at 200 would hand the baseline an advantage exactly as
surely as the reverse did earlier in this project. Both arms therefore move
together, at every seed, or neither does.

Photo first: it is the cheapest and it is the cell where OUTPOST currently ties.
If DEMO gains more than OUTPOST from the longer budget, the Photo result gets
worse for us, and that is the outcome this run exists to expose rather than avoid.
Computers follows; cs is queued last on cost.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S10 = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
jobs = []
for s in S10:
    jobs.append({"dataset": "photo", "seed": s, "tag": "E400_outpost",
                 "priority": 1, "mem_gb": 15.0, "set": {"num_epochs": 400}})
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E400_demo", "priority": 1, "mem_gb": 16.0,
                 "set": {"num_epochs": 400, "mixup": True, "energy_loss": False}})
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "computers", "seed": s, "tag": "E400_outpost",
                 "priority": 2, "mem_gb": 22.0, "shard_rotations": True,
                 "set": {"num_epochs": 400}})
    jobs.append({"dataset": "computers", "seed": s, "method": "demo",
                 "tag": "E400_demo", "priority": 2, "mem_gb": 24.0,
                 "shard_rotations": True,
                 "set": {"num_epochs": 400, "mixup": True, "energy_loss": False}})
json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/epochs.json", "w"), indent=1)
print(f"{len(jobs)} jobs (photo 20 unsharded, computers 10 -> 50 rotation-jobs)")
