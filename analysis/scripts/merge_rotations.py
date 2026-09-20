"""Turn per-rotation shards into results.csv rows.

A dataset's score is the mean over its complete rotation set. When a run is
sharded across GPUs with `main.py --rotations`, no single process sees the whole
set, so the row is assembled here instead. A group is written only when every
rotation of the dataset is present; incomplete groups are listed with the
rotations still missing, which doubles as the campaign's progress report.

    python analysis/scripts/merge_rotations.py            # write complete rows
    python analysis/scripts/merge_rotations.py --status   # report only
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# The complete rotation set per dataset, as main.anomaly_classes derives it.
# Hard-coded here so merging needs neither the datasets nor a GPU.
ROTATIONS = {
    "photo": [0, 7],
    "computers": [0, 3, 5, 6, 9],
    "cs": [0, 1, 3, 6, 8, 9, 12, 14],
    "yelp": [1], "amazon": [1], "tfinance": [1],
    "ogbn-arxiv": [4, 8, 10, 34],
    "ogbn-mag": [2, 39, 143, 151, 176, 195, 206, 215, 216, 231, 263, 282,
                 321, 327, 341],
}


def expected_rotations(dataset):
    if ROTATIONS.get(dataset):
        return ROTATIONS[dataset]
    from main import anomaly_classes
    from utils import load_data
    _, _, info = load_data(dataset)
    ROTATIONS[dataset] = [int(c) for c in anomaly_classes(dataset, info)]
    return ROTATIONS[dataset]


def load_shards():
    groups = defaultdict(dict)
    cfgs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results/rotations/*.json"))):
        try:
            r = json.load(open(p))
        except Exception as e:
            print(f"  unreadable, skipped: {os.path.basename(p)} ({e})")
            continue
        key = (r["dataset"], r["method"], r["seed"],
               r.get("train_seed"), r.get("tag", ""))
        for rot in r["rotations"]:
            groups[key][int(rot["rotation_class"])] = rot
        cfgs[key] = r.get("config", {})
    return groups, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report, do not write")
    ap.add_argument("--rewrite", action="store_true",
                    help="recompute rows from the CURRENT shards even when a row "
                         "already exists. Needed after re-running a group: "
                         "run_campaign --force overwrites the rotation records, "
                         "but this script skips any group already in results.csv, "
                         "so the aggregate silently keeps the superseded value. "
                         "That is how a re-run to remove the ckg10/ckg12 machine "
                         "confound left six cells still holding pre-move numbers.")
    a = ap.parse_args()

    os.chdir(ROOT)
    groups, cfgs = load_shards()
    have = set()
    path = "results/results.csv"
    if os.path.exists(path):
        for row in csv.DictReader(open(path)):
            have.add((row["dataset"], row["method"], int(row["seed"]),
                      int(row["train_seed"]) if row["train_seed"] else None,
                      row["tag"]))

    from utils import aggregate_rotations
    from results_writer import append_row

    # A tag names one configuration, so its shards must share one epoch budget.
    # DEMO never inherits a dataset's num_epochs (load_config takes only input_dim
    # and eval_batch_mult from the dataset block), so a campaign that forgets the
    # override silently produces 200-epoch shards under a 400-epoch tag - and
    # B_demo_mix is a MAIN TABLE tag, which nothing downstream filters on budget.
    # Detect the split here, where every shard passes, and refuse those rows.
    from collections import defaultdict as _dd
    # the config lives in cfgs[key], not in groups[key] (which holds rotation
    # dicts) - reading the wrong structure made this check silently find nothing
    budgets = _dd(set)
    for key, cfg in cfgs.items():
        ds, meth, seed, tseed, tag = key
        # NSReg records its budget as n_epochs and leaves num_epochs empty; reading
        # only num_epochs made every NSReg shard invisible to this check, so an
        # NSReg campaign that forgot its override (default 201) would have passed
        c_ = cfg or {}
        e = c_.get("num_epochs")
        if e is None:
            e = c_.get("n_epochs")
        if e is not None:
            budgets[(ds, meth, tag)].add(int(e))
    # Detecting a SPLIT is not enough: when a campaign forgets the override,
    # EVERY shard of that tag is wrong and they agree with each other. The check
    # that catches it is across methods - a dataset's main-table arms all report
    # the same protocol, so DEMO at 200 beside OUTPOST at 400 on the same graph is
    # the error, however self-consistent DEMO's own shards look. Budget arms that
    # differ on purpose (B_demo, DT_*, H_ep400) are not main-table tags and are
    # not checked.
    MAIN = {"outpost": {"photo": "E400_outpost", "computers": "E400_outpost",
                        "cs": "E400_outpost", "yelp": "A_main", "amazon": "A_sim0.0",
                        "tfinance": "A_sim0.0", "ogbn-arxiv": "A_main",
                        "ogbn-mag": "A_main"},
            "demo": {"photo": "E400_demo", "computers": "E400_demo", "cs": "E400_demo",
                     "yelp": "B_demo_mix", "amazon": "B_demo_mix",
                     "tfinance": "B_demo_mix", "ogbn-arxiv": "B_demo_mix"},
            "nsreg": {d: "E400_nsreg" for d in
                      ("photo", "computers", "cs", "yelp", "amazon", "tfinance",
                       "ogbn-arxiv")}}
    skip_keys = set()
    per_ds = _dd(dict)
    for (ds, meth, tag), eps in budgets.items():
        if MAIN.get(meth, {}).get(ds) == tag and eps:
            per_ds[ds][meth] = sorted(eps)
    cross = {ds: m for ds, m in per_ds.items()
             if len({tuple(v) for v in m.values()}) > 1}
    if cross:
        print("\n  MAIN-TABLE BUDGET MISMATCH - these graphs report methods at "
              "different epoch counts; their rows are NOT written:")
        for ds, m in sorted(cross.items()):
            print(f"    {ds}: " + ", ".join(f"{k}={v}" for k, v in sorted(m.items())))
        for key in groups:
            if key[0] in cross and MAIN.get(key[1], {}).get(key[0]) == key[4]:
                skip_keys.add(key)

    mixed = {k: v for k, v in budgets.items() if len(v) > 1}
    if mixed:
        print("\n  BUDGET SPLIT - these tags hold shards at more than one epoch "
              "count; their rows are NOT written:")
        for (ds, meth, tag), eps in sorted(mixed.items(), key=str):
            print(f"    {ds}/{meth}/{tag}: epochs {sorted(eps)}")
            for key in groups:
                if (key[0], key[1], key[4]) == (ds, meth, tag):
                    skip_keys.add(key)
        print("  Quarantine the wrong-budget shards (results/wrong_budget/) or "
              "re-run them, then merge again.\n")

    complete, partial, written = [], [], 0
    for key in sorted(groups, key=str):
        if key in skip_keys:
            continue
        ds, meth, seed, tseed, tag = key
        try:
            want = expected_rotations(ds)
        except Exception as e:
            print(f"  cannot resolve rotations for {ds}: {e}")
            continue
        got = groups[key]
        missing = [c for c in want if c not in got]
        if missing:
            partial.append((key, len(got), len(want), missing))
            continue
        complete.append(key)
        if a.status or (key in have and not a.rewrite):
            continue
        rots = [got[c] for c in want]
        agg = aggregate_rotations(rots)
        aggv = aggregate_rotations(rots, which="val_selected")
        append_row(ds, meth, seed, tseed, agg, aggv, cfgs[key], tag=tag)
        written += 1
        print(f"  wrote {ds}/{meth}/s{seed}/{tag}: "
              f"ROC {agg['auroc_all']:.4f} PR {agg['aupr_all']:.4f}")

    print(f"\n{len(complete)} complete group(s), {written} new row(s), "
          f"{len(partial)} in progress")
    for key, n, m, missing in sorted(partial, key=str):
        ds, meth, seed, tseed, tag = key
        show = missing if len(missing) <= 8 else missing[:8] + ["..."]
        print(f"  {ds}/{meth}/s{seed}/{tag}: {n}/{m} rotations, missing {show}")


if __name__ == "__main__":
    main()
