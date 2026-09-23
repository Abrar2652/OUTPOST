"""Append one row per run to results/results.csv.

Every row carries the config hash and the flags that matter, so a table can
always be traced back to the exact settings that produced it. Both metrics are
recorded:

  best_*        per-metric maximum over epochs.                The test set is structurally identical to the unified open-set protocol.
  valsel_*      metrics at the peak-validation epoch - the honest, deployable
                number. Usually lower. Report both in the paper.
"""

import csv
import hashlib
import json
import os
import time

FIELDS = ["timestamp", "dataset", "method", "seed", "train_seed", "tag",
          "best_auroc", "best_aupr", "best_auroc_unseen", "best_aupr_unseen",
          "valsel_auroc", "valsel_aupr",
          "sim_topk_frac", "use_mixup", "use_halo", "use_fview_gate",
          "use_hybrid", "hidden_dim", "num_epochs", "config_sha8"]

PATH = "results/results.csv"


def append_row(dataset, method, seed, train_seed, agg, aggv, cfg, tag=""):
    os.makedirs("results", exist_ok=True)
    sha = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset, "method": method, "seed": seed,
        "train_seed": "" if train_seed is None else train_seed, "tag": tag,
        "best_auroc": f"{agg['auroc_all']:.4f}",
        "best_aupr": f"{agg['aupr_all']:.4f}",
        "best_auroc_unseen": f"{agg['auroc_unknown']:.4f}",
        "best_aupr_unseen": f"{agg['aupr_unknown']:.4f}",
        "valsel_auroc": f"{aggv.get('auroc_all', 0):.4f}",
        "valsel_aupr": f"{aggv.get('aupr_all', 0):.4f}",
        "sim_topk_frac": cfg.get("sim_topk_frac", 0.0),
        "use_mixup": cfg.get("use_mixup"), "use_halo": cfg.get("use_halo"),
        "use_fview_gate": cfg.get("use_fview_gate", False),
        "use_hybrid": cfg.get("use_hybrid", False),
        "hidden_dim": cfg.get("hidden_dim"), "num_epochs": (cfg.get("num_epochs") if cfg.get("num_epochs") is not None else cfg.get("n_epochs", "")),
        "config_sha8": sha,
    }
    new = not os.path.exists(PATH)
    with open(PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row
