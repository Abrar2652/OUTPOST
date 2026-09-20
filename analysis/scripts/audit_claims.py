"""Check every number written in the prose against the numbers the runs produced.

A reader who checks cross-references finds prose/table drift; a reader who skims
does not. This repository's numbers moved substantially during the re-run - Photo
0.9066 -> 0.8310, pseudo-labelling -0.016 -> +0.0038, DEMO's Yelp AUC-PR 0.2238
-> 0.3516 - so stale figures in the documentation are the expected failure, not a
hypothetical one.

The audit extracts every decimal number of three or more places from the
documentation and asks whether the run record can produce it. The reference set
is built from everything downstream of the runs:

  results/results.csv      every per-run metric, and every group mean and sd
  analysis/tables/stats.json   paired deltas, p-values, CI bounds, cell summaries
  analysis/tables/*.csv    published baselines, spectral diagnostics, provenance
  analysis/tables/*.json   detectability law, hyperparameter selection

A number that appears nowhere in that set is not necessarily wrong - it may be a
hyperparameter, a threshold, a runtime, or a figure quoted from another paper -
so the output is a list to inspect rather than a pass/fail.

THE IMPORTANT CATEGORY IS "STALE", NOT "NOT FOUND". Set membership alone gives
false assurance: an out-of-date figure still appears somewhere in the record and
passes. This version separates the reference set in two.

  CURRENT     values derivable from the newest row of each
              (dataset, method, seed, train_seed, tag), and from group
              statistics over those rows
  SUPERSEDED  values from rows that a later run replaced - results.csv appends
              rather than overwrites, so every prior value is still on disk

A prose figure matching a SUPERSEDED value but not a CURRENT one is provably out
of date. That is the failure this script exists to catch and the one the previous
version missed twice: once when a re-run replaced six cells and `--rewrite` had
not yet been applied, and once when the DEMO energy table was written at n=2 of
3 seeds. Both passed a set-membership check.

    python analysis/scripts/audit_claims.py
    python analysis/scripts/audit_claims.py --show-ok
"""

import argparse
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

DOCS = ["METHODOLOGY.md", "README.md", "analysis/REVIEWER_PROOFING.md",
        "analysis/PHASE0_FINDINGS.md", "analysis/tables/prediction_ogb.md",
        "analysis/tables/prediction_tuning.md",
        "analysis/tables/prediction_budget.md"]

# numbers that are configuration or protocol, not measurements
IGNORE = {
    "0.05", "0.01", "0.10", "0.001", "0.0005", "0.95", "0.5", "0.2", "0.1",
    "0.25", "0.75", "0.9", "0.0003", "0.002", "0.02", "0.03", "0.07",
}
NUM_RE = re.compile(r"(?<![\w.])(\d\.\d{3,}|0\.\d{3,})(?![\w])")


def known_values(superseded=False):
    """Every number the run record can produce, as 3- and 4-dp strings."""
    vals = set()

    def add(x):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        if not (0 < abs(f) < 1e6):
            return
        for dp in (3, 4):
            vals.add(f"{abs(f):.{dp}f}")

    import pandas as pd
    import numpy as np
    if os.path.exists("results/results.csv"):
        raw = pd.read_csv("results/results.csv")
        key = ["dataset", "method", "seed", "train_seed", "tag"]
        cur = raw.drop_duplicates(subset=key, keep="last")
        # rows a later run replaced: still on disk, and the only way to tell a
        # stale figure from one that was never in the record at all
        d = raw[~raw.index.isin(cur.index)] if superseded else cur
        d = d[d.num_epochs >= 100]
        metrics = ["best_auroc", "best_aupr", "best_auroc_unseen",
                   "best_aupr_unseen", "valsel_auroc", "valsel_aupr"]
        for m in metrics:
            for v in d[m]:
                add(v)
        for _, g in ([] if superseded else d.groupby(["dataset", "method", "tag"])):
            for m in metrics:
                add(g[m].mean())
                if len(g) > 1:
                    add(g[m].std(ddof=1))
        # paired deltas between every tag pair on a dataset, since the prose
        # quotes many of them
        for ds, g in d.groupby("dataset"):
            tags = g.tag.unique()
            for t1 in tags:
                for t2 in tags:
                    if t1 == t2:
                        continue
                    a = g[g.tag == t1].set_index("seed")
                    b = g[g.tag == t2].set_index("seed")
                    common = a.index.intersection(b.index)
                    if len(common) < 2:
                        continue
                    for m in metrics:
                        add(a.loc[common, m].mean() - b.loc[common, m].mean())

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)):
            add(o)

    for p in ([] if superseded else glob.glob("analysis/tables/*.json")):
        try:
            walk(json.load(open(p)))
        except Exception:
            pass
    for p in ([] if superseded else
              glob.glob("analysis/tables/*.csv") + ["results/reference_runs.csv"]):
        if not os.path.exists(p):
            continue
        try:
            for row in csv.DictReader(open(p)):
                for v in row.values():
                    add(v)
                    # cells like "0.8310+-0.0174"
                    if isinstance(v, str):
                        for part in re.split(r"\+-|\+/-|±", v):
                            add(part.strip())
        except Exception:
            pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-ok", action="store_true")
    a = ap.parse_args()

    vals = known_values()
    stale_vals = known_values(superseded=True) - vals
    print(f"reference set: {len(vals)} current values, "
          f"{len(stale_vals)} superseded-only values\n")

    total = ok = 0
    unmatched, stale = [], []
    for doc in DOCS:
        if not os.path.exists(doc):
            continue
        for ln, line in enumerate(open(doc, encoding="utf-8"), 1):
            if line.lstrip().startswith(("#", "```")):
                continue
            for m in NUM_RE.finditer(line):
                s = m.group(1)
                if s in IGNORE:
                    continue
                total += 1
                f = float(s)
                hit = any(f"{f:.{dp}f}" in vals for dp in (3, 4))
                was = any(f"{f:.{dp}f}" in stale_vals for dp in (3, 4))
                if hit:
                    ok += 1
                    if a.show_ok:
                        print(f"  ok   {doc}:{ln}  {s}")
                elif was:
                    stale.append((doc, ln, s, line.strip()[:96]))
                else:
                    unmatched.append((doc, ln, s, line.strip()[:96]))

    # Calibrate before reporting. With ~5,000 distinct 3/4-dp strings covering
    # the range our metrics occupy, set membership is nearly vacuous: a number
    # drawn at random passes ~94% of the time. Quoting "624/625 reconcile" as
    # evidence of correctness - which this script's earlier output invited, and
    # which was done repeatedly - overstates it badly. The STALE count is the
    # informative one: it tests against ~50 superseded-only values, so a hit
    # there is specific rather than coincidental.
    import random
    rng = random.Random(0)
    fp = sum(any(f"{rng.uniform(0.30, 0.99):.{dp}f}" in vals for dp in (3, 4))
             for _ in range(4000)) / 4000
    print(f"{ok}/{total} numeric claims match a CURRENT value")
    print(f"  ... but a RANDOM number in [0.30, 0.99] also matches "
          f"{100*fp:.0f}% of the time.")
    print(f"  Treat that count as a weak smoke test. The STALE list below is the "
          f"signal:\n  it compares against {len(stale_vals)} superseded-only "
          f"values, where a hit means the\n  figure was correct once and a later "
          f"run replaced it.\n")
    if stale:
        print("*** STALE - matches a value a later run REPLACED. Fix these: ***")
        cur_doc = None
        for doc, ln, v, ctx in stale:
            if doc != cur_doc:
                print(f"\n  {doc}")
                cur_doc = doc
            print(f"    L{ln:<5} {v:<10} {ctx}")
        print()
    if unmatched:
        print("NOT FOUND in the run record - inspect each:")
        cur = None
        for doc, ln, s, ctx in unmatched:
            if doc != cur:
                print(f"\n  {doc}")
                cur = doc
            print(f"    L{ln:<5} {s:<10} {ctx}")
    print("\nA miss is not automatically an error: hyperparameters, runtimes, "
          "and figures\nquoted from other papers legitimately do not appear in "
          "our results. The point is\nthat none of them drifts unnoticed.")


if __name__ == "__main__":
    main()
