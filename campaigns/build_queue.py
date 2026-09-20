"""One priority-ordered queue, in the order the work is worth doing.

  1  machine-confound re-run   the only band touching numbers already in the
                               paper; until it lands, Photo's main cell and both
                               Computers arms straddle the ckg10 -> ckg12 move
                               and are not citeable
  2  cs + DEMO                 cs is one of two wins against published constants
                               and currently rests on a transcribed number
  3  hyperparameter provenance Yelp/Computers/CS have no recorded selection
                               history; Yelp carries the headline result
                               (pre-registered in prediction_hparam_all.md)
  4  cs + SimSample (P14)      mechanism question; can wait

cs lean is already complete (39/39) and is not in this queue.
"""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jobs = []

# --- band 1: machine-confound re-run (the pre-ckg12 seeds) ---
for s in [0, 1, 2, 3, 42]:
    jobs.append({"dataset": "photo", "seed": s, "tag": "A_main", "priority": 1,
                 "shard_rotations": True})
for s in [0, 1]:
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E_demo_energy", "priority": 1, "shard_rotations": True,
                 "set": {"energy_loss": True}})
jobs.append({"dataset": "photo", "seed": 42, "tag": "T_hidden32", "priority": 1,
             "shard_rotations": True, "set": {"hidden_dim": 32}})
for s in [1, 42]:
    jobs.append({"dataset": "computers", "seed": s, "tag": "A_main",
                 "priority": 1, "shard_rotations": True})
for s in [2, 3]:
    jobs.append({"dataset": "computers", "seed": s, "method": "demo",
                 "tag": "B_demo", "priority": 1, "shard_rotations": True,
                 "set": {"energy_loss": False}})
for s in [0, 1, 2, 42]:
    jobs.append({"dataset": "cs", "seed": s, "tag": "A_main", "priority": 1,
                 "shard_rotations": True})

# --- band 2: cs + DEMO ---
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "cs", "seed": s, "method": "demo", "tag": "B_demo",
                 "priority": 2, "shard_rotations": True, "mem_gb": 45.0,
                 "set": {"energy_loss": False}})

# --- band 3: hyperparameter provenance, validation-only (P15/P16) ---
YELP = {"T_hidden16": {"hidden_dim": 16}, "T_hidden64": {"hidden_dim": 64},
        "T_Kp2": {"K_p": 2}, "T_Kp6": {"K_p": 6},
        "T_drop0.1": {"drop_out": 0.1}, "T_drop0.4": {"drop_out": 0.4},
        "T_warmup5": {"warmup_epochs": 5}, "T_warmup20": {"warmup_epochs": 20},
        "T_alpha0.01": {"alpha_plus": 0.01}, "T_alpha0.10": {"alpha_plus": 0.10},
        "T_lr0.003": {"lr": 0.003},
        "T_synth": {"use_mixup": True, "use_halo": True}}
BIG = {"T_hidden32": {"hidden_dim": 32}, "T_hidden128": {"hidden_dim": 128},
       "T_Kp4": {"K_p": 4}, "T_Kp12": {"K_p": 12},
       "T_drop0.3": {"drop_out": 0.3}, "T_warmup15": {"warmup_epochs": 15},
       "T_lambda_un0.5": {"lambda_un": 0.5}, "T_lambda_un1.5": {"lambda_un": 1.5},
       "T_nosynth": {"use_mixup": False, "use_halo": False}}
for tag, st in YELP.items():
    for s in [42, 0, 1]:
        jobs.append({"dataset": "yelp", "seed": s, "tag": tag, "priority": 3,
                     "set": st})
for ds in ["computers", "cs"]:
    for tag, st in BIG.items():
        for s in [42, 0, 1]:
            jobs.append({"dataset": ds, "seed": s, "tag": tag, "priority": 3,
                         "shard_rotations": True, "set": st})

# --- band 4: P14, cs + SimSample ---
for s in [42, 0, 1, 2, 3]:
    jobs.append({"dataset": "cs", "seed": s, "tag": "C_sim", "priority": 4,
                 "shard_rotations": True, "mem_gb": 45.0,
                 "set": {"sim_topk_frac": 1.0}})

json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/queue.json", "w"), indent=1)
from collections import Counter
c = Counter(j["priority"] for j in jobs)
print(f"{len(jobs)} jobs before sharding; by band: " +
      ", ".join(f"{k}:{v}" for k, v in sorted(c.items())))
