"""Generate the validation-only hyperparameter sweep on Photo.

The grid and the selection rule are pre-registered in
`analysis/tables/prediction_tuning.md`, written before any of these runs was
launched. Read that first: the point of this sweep is not the numbers it
produces but the fact that the rule producing them was fixed in advance.

One factor at a time from the repository default, three seeds each. The default
arm is NOT re-run - the `A_main` rows already are that point at those seeds.
Each arm doubles as a multi-seed sensitivity point, which is the other thing
METHODOLOGY section 8 says is missing.

These arms now live in campaigns/gpu7.json (band 6) rather than in a runner of
their own: one runner per GPU, or their memory budgets double-book the card.
This script is kept because it records the grid and why it was chosen.

    python analysis/scripts/select_hparams.py --dataset photo
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = [42, 0, 1]

GRID = {
    "T_lambda_un0.5": {"lambda_un": 0.5},
    "T_lambda_un1.5": {"lambda_un": 1.5},
    "T_Kp4":          {"K_p": 4},
    "T_Kp12":         {"K_p": 12},
    "T_hidden32":     {"hidden_dim": 32},
    "T_hidden128":    {"hidden_dim": 128},
    "T_warmup15":     {"warmup_epochs": 15},
    "T_drop0.3":      {"drop_out": 0.3},
    "T_lean":         {"use_mixup": False, "use_halo": False},
    "T_alpha0.01":    {"alpha_plus": 0.01},
    "T_gateq0.8":     {"atlas_gate_q": 0.8},
    "T_lr0.003":      {"lr": 0.003},
}

jobs = [{"dataset": "photo", "seed": s, "tag": tag, "priority": 1, "set": sets}
        for tag, sets in GRID.items() for s in SEEDS]

json.dump({"_comment": __doc__, "seeds": SEEDS, "grid": GRID, "jobs": jobs},
          open("campaigns/tuning.json", "w"), indent=1)
print(f"{len(jobs)} jobs ({len(GRID)} arms x {len(SEEDS)} seeds)")
