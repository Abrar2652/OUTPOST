"""Measured cost of OUTPOST against the DEMO baseline, on one machine.

The efficiency claim in METHODOLOGY section 5 is a parameter count. A parameter
count is the weakest of the three things a referee means by "cheaper": it says
nothing about wall-clock or memory, and a method can be small and slow. This
builds the table from what the runs actually recorded - parameters, seconds per
rotation, and peak GPU allocation - for both methods on identical hardware, at
identical seeds, from the same campaign.

Everything comes from results/rotations/*.json, so no run is repeated to
produce it.

    python analysis/scripts/efficiency.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
# The tags the main table actually reports, per dataset - not a fixed pair. The
# old {"A_main", "B_demo"} set silently dropped Amazon and T-Finance (whose arms
# are A_sim*/B_demo_mix) and measured Photo/Computers/CS at their 200-epoch arms
# while the paper reports the 400-epoch ones. Parameter counts are budget-
# independent, but "which runs is this table describing" should not need a guess.
def _main_tags():
    sel = {}
    try:
        sel = json.load(open("analysis/tables/arm_selection.json"))
    except Exception:
        pass
    sel = {k: (v["arm"] if isinstance(v, dict) else v) for k, v in sel.items()}
    out = {
        "photo": ("E400_outpost", "E400_demo"),
        "computers": ("E400_outpost", "E400_demo"),
        "cs": ("E400_outpost", "E400_demo"),
        "yelp": ("A_main", "B_demo_mix"),
        "amazon": (sel.get("amazon", "A_sim0.0"), "B_demo_mix"),
        "tfinance": (sel.get("tfinance", "A_sim0.0"), "B_demo_mix"),
        "ogbn-arxiv": ("A_main", None),
        "ogbn-mag": ("A_main", None),
    }
    return {ds: {t for t in pair if t} for ds, pair in out.items()}


MAIN_TAGS = _main_tags()


def load():
    rows = []
    for p in sorted(glob.glob("results/rotations/*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("tag") not in MAIN_TAGS.get(r.get("dataset"), set()):
            continue
        for rot in r["rotations"]:
            rows.append({"dataset": r["dataset"], "method": r["method"],
                         "seed": r["seed"], "rotation": rot.get("rotation_class"),
                         "n_params": rot.get("n_params"),
                         "seconds": rot.get("seconds"),
                         "peak_gpu_gb": rot.get("peak_gpu_gb"),
                         "epochs": r.get("config", {}).get("num_epochs")})
    return pd.DataFrame(rows)


def main():
    d = load()
    if d.empty:
        print("no tagged main-table runs yet")
        return
    d = d[d.epochs.astype(float) >= 100]
    out = []
    for (ds, m), g in d.groupby(["dataset", "method"]):
        ep = float(g.epochs.iloc[0])
        out.append({
            "dataset": ds, "method": m, "runs": len(g),
            "params": int(g.n_params.median()),
            "sec_per_rotation": round(g.seconds.median(), 1),
            "sec_per_epoch": round(g.seconds.median() / ep, 3),
            "peak_gpu_gb": round(g.peak_gpu_gb.max(), 2)
            if g.peak_gpu_gb.notna().any() else None,
        })
    t = pd.DataFrame(out).sort_values(["dataset", "method"])

    # ratios, where both arms exist for a dataset
    ratios = []
    for ds, g in t.groupby("dataset"):
        o = g[g.method == "outpost"]
        b = g[g.method == "demo"]
        if o.empty or b.empty:
            continue
        ratios.append({
            "dataset": ds,
            "params_outpost": int(o.params.iloc[0]),
            "params_demo": int(b.params.iloc[0]),
            "param_ratio": round(o.params.iloc[0] / b.params.iloc[0], 3),
            "time_ratio": round(o.sec_per_rotation.iloc[0]
                                / b.sec_per_rotation.iloc[0], 3),
            "mem_ratio": (round(o.peak_gpu_gb.iloc[0] / b.peak_gpu_gb.iloc[0], 3)
                          if o.peak_gpu_gb.iloc[0] and b.peak_gpu_gb.iloc[0]
                          else None),
        })
    r = pd.DataFrame(ratios)

    print("=" * 78)
    print("MEASURED COST   (median over seeds x rotations, one RTX A5000)")
    print("=" * 78)
    print(t.to_string(index=False))
    t.to_csv("analysis/tables/efficiency_measured.csv", index=False)
    if not r.empty:
        print("\nOUTPOST relative to DEMO (<1 means OUTPOST is cheaper):")
        print(r.to_string(index=False))
        r.to_csv("analysis/tables/efficiency_ratio.csv", index=False)
        print("\nNote: sec_per_rotation and time_ratio are WALL CLOCK on a "
              "shared machine and are NOT a method comparison. The two arms were "
              "scheduled at different times under\ndifferent contention - the same "
              "Photo pair read 0.748 at the 200-epoch arms and 1.402 at the "
              "400-epoch ones, which is co-tenancy, not the method\ngetting slower. "
              "Quote param_ratio (exact) and mem_ratio (per-process peak) only; a "
              "timing claim needs an isolated run.")
        print("\nNote: both arms ran with DEMO's energy-gradient term OFF. That "
              "term needs a\nper-training-node second-order backward pass; its "
              "measured cost is reported\nseparately by campaigns/phaseB_energy "
              "and is not folded into these ratios.")
    print("\n-> analysis/tables/efficiency_measured.csv, efficiency_ratio.csv")


if __name__ == "__main__":
    main()
