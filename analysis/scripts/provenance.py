"""Trace every reported number back to the runs that produced it.

`analysis/tables/legacy/reproducibility.csv` does this by hand for the archived
measurements, with a caveat column that is the most useful part of it. Hand
maintenance does not survive a 200-run campaign, so this regenerates the same
mapping from the run records: for each main-table cell, the seeds behind it,
each seed's value, the rotation files, the config hash, and the epoch budget.

A cell whose seeds disagree with its claimed n, or whose runs carry more than
one config hash, is a bookkeeping error that a reader cannot see in the table
and is reported here as a warning rather than left to be discovered.

    python analysis/scripts/provenance.py
"""

import glob
import json
import os
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def main():
    if not os.path.exists("results/results.csv"):
        print("no results/results.csv yet")
        return
    d = pd.read_csv("results/results.csv")
    d = d.drop_duplicates(subset=["dataset", "method", "seed", "train_seed",
                                  "tag"], keep="last")
    d = d[d.num_epochs >= 100]

    shards = defaultdict(list)
    for p in sorted(glob.glob("results/rotations/*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        shards[(r["dataset"], r["method"], r["seed"], r.get("tag", ""))].append(
            os.path.basename(p))

    rows, warnings = [], []
    for (ds, meth, tag), g in d.groupby(["dataset", "method", "tag"]):
        seeds = sorted(g.seed.tolist())
        hashes = sorted(set(g.config_sha8.astype(str)))
        files = sorted(f for s in seeds for f in shards.get((ds, meth, s, tag), []))
        rows.append({
            "dataset": ds, "method": meth, "tag": tag, "n_seeds": len(seeds),
            "seeds": " ".join(str(s) for s in seeds),
            "epochs": int(g.num_epochs.max()),
            "best_auroc": f"{g.best_auroc.mean():.4f}"
            + (f" +- {g.best_auroc.std(ddof=1):.4f}" if len(g) > 1 else ""),
            "best_aupr": f"{g.best_aupr.mean():.4f}"
            + (f" +- {g.best_aupr.std(ddof=1):.4f}" if len(g) > 1 else ""),
            "valsel_auroc": f"{g.valsel_auroc.mean():.4f}",
            "valsel_aupr": f"{g.valsel_aupr.mean():.4f}",
            "config_sha8": " ".join(hashes),
            "rotation_files": len(files),
        })
        if len(hashes) > 1:
            warnings.append(f"{ds}/{meth}/{tag}: {len(hashes)} different config "
                            f"hashes across seeds ({hashes}) - the cell averages "
                            f"runs that were not the same configuration")
        if len(set(seeds)) != len(seeds):
            warnings.append(f"{ds}/{meth}/{tag}: duplicate seeds {seeds}")

    t = pd.DataFrame(rows).sort_values(["dataset", "tag", "method"])
    t.to_csv("analysis/tables/provenance.csv", index=False)
    print(t.to_string(index=False))
    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print("  " + w)
    else:
        print("\nno bookkeeping warnings: every cell is one configuration "
              "across its seeds")
    print("\n-> analysis/tables/provenance.csv")


if __name__ == "__main__":
    main()
