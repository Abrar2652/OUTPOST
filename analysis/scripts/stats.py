"""Statistics for every claim in the paper.

Three kinds of comparison appear in this work, and they do not admit the same
test. Conflating them is the usual way a benchmark table overstates itself, so
each is computed separately and labelled:

1. `summarise`  - a cell of the main table. n seeds, reported as mean, sample
   standard deviation (ddof=1) and a 95% interval. The interval is the
   bootstrap percentile interval over seeds, with the Student-t interval also
   reported; at n=5 the two disagree enough to be worth showing both.

2. `paired`     - our method against a baseline WE RAN, at identical seeds and
   identical splits. The pairing unit is (seed, rotation), which multiplies the
   available units by the rotation count and is legitimate because both arms
   see the same split for the same rotation. Wilcoxon signed-rank (no normality
   assumption, primary) plus a paired t-test and a bootstrap interval on the
   mean paired difference.

3. `vs_published` - our method against a number transcribed from another paper.
   No variance is available for the other side and the splits are not shared,
   so NO paired test is possible. What can be said is whether our seed
   distribution lies above the published constant: a one-sample test against
   that constant, plus the count of seeds that individually beat it. This is
   reported as a weaker form of evidence and is labelled as such in the output.

A family of related tests (the ablation table) is corrected with Holm-Bonferroni.

    python analysis/scripts/stats.py                    # everything, to stdout
    python analysis/scripts/stats.py --json out.json    # machine-readable too
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results/results.csv")
ROTDIR = os.path.join(ROOT, "results/rotations")
BOOT = 10000
RNG = np.random.default_rng(0)   # fixed: the intervals are reproducible


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

# The uniform 400-epoch protocol, per dataset and per method (mirrors
# make_paper_tables.py). Until 2026-09-04 every section below selected A_main
# for OUTPOST (200 epochs on photo/computers/cs) and B_demo for the baseline -
# DEMO's own "w/o Mix" ablation, not the published method. The paired tests
# the prose cites were therefore against the wrong DEMO.
MAIN_TAG = {
    "outpost": {"photo": "E400_outpost", "computers": "E400_outpost",
                "cs": "E400_outpost", "yelp": "A_main",
                "ogbn-arxiv": "A_main", "ogbn-mag": "A_main"},
    "demo":    {"photo": "E400_demo", "computers": "E400_demo", "cs": "E400_demo",
                "yelp": "B_demo_mix", "amazon": "B_demo_mix", "tfinance": "B_demo_mix"},
    "nsreg":   {ds: "E400_nsreg" for ds in ("photo", "computers", "cs", "yelp", "amazon", "tfinance")},
}
try:   # validation-selected SimSample arms (amazon, tfinance), from build_appendix
    MAIN_TAG["outpost"].update(json.load(open("analysis/tables/arm_selection.json")))
except Exception:
    pass
# the arm each ablation was run against (ablations ran at each dataset's
# native budget, i.e. the A_main base, except amazon whose full model is the
# validation-selected A_sim0.0)
ABL_BASE = {}
try:
    ABL_BASE.update(json.load(open("analysis/tables/arm_selection.json")))
except Exception:
    pass


def is_main(df):
    """Boolean mask: row is its (method, dataset)'s main-table arm."""
    return df.apply(lambda r: str(r["tag"]) ==
                    MAIN_TAG.get(r["method"], {}).get(r["dataset"], "\0"), axis=1)


def load_runs():
    """One row per (dataset, method, seed, tag): the aggregate over rotations."""
    if not os.path.exists(RESULTS):
        return pd.DataFrame()
    d = pd.read_csv(RESULTS)
    for c in ("best_auroc", "best_aupr", "best_auroc_unseen", "best_aupr_unseen",
              "valsel_auroc", "valsel_aupr"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # A campaign that is interrupted and resumed can append a second row for the
    # same run. Left in, that run would be counted twice and would shrink the
    # reported standard deviation of the cell it lands in. Keep the newest.
    before = len(d)
    d = d.drop_duplicates(subset=["dataset", "method", "seed", "train_seed",
                                  "tag"], keep="last").reset_index(drop=True)
    if len(d) < before:
        print(f"[stats] dropped {before - len(d)} duplicate run row(s)")
    return d


def load_rotations():
    """One row per (dataset, method, seed, tag, rotation) - the pairing unit."""
    rows = []
    for p in sorted(glob.glob(os.path.join(ROTDIR, "*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        for rot in r["rotations"]:
            rows.append({
                "dataset": r["dataset"], "method": r["method"],
                "seed": r["seed"], "train_seed": r.get("train_seed"),
                "tag": r.get("tag", ""), "rotation": rot.get("rotation_class"),
                "seconds": rot.get("seconds"), "n_params": rot.get("n_params"),
                "best_auroc": rot["best"]["auroc_all"],
                "best_aupr": rot["best"]["aupr_all"],
                "best_auroc_unseen": rot["best"]["auroc_unknown"],
                "best_aupr_unseen": rot["best"]["aupr_unknown"],
                "valsel_auroc": rot.get("val_selected", {}).get("auroc_all"),
                "valsel_aupr": rot.get("val_selected", {}).get("aupr_all"),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. summary of a table cell
# --------------------------------------------------------------------------

def boot_ci(x, stat=np.mean, alpha=0.05, n=BOOT):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    reps = stat(x[idx], axis=1)
    return tuple(np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def summarise(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    out = {"n": int(len(x)), "mean": float(np.mean(x)) if len(x) else float("nan"),
           "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
           "min": float(np.min(x)) if len(x) else float("nan"),
           "max": float(np.max(x)) if len(x) else float("nan"),
           "values": [round(float(v), 4) for v in x]}
    if len(x) > 1:
        se = out["sd"] / np.sqrt(len(x))
        t = sps.t.ppf(0.975, len(x) - 1)
        out["t_ci95"] = [out["mean"] - t * se, out["mean"] + t * se]
        out["boot_ci95"] = list(boot_ci(x))
    return out


# --------------------------------------------------------------------------
# 2. paired comparison against a baseline we ran ourselves
# --------------------------------------------------------------------------

def paired(a, b, label_a="A", label_b="B"):
    """a, b: arrays of matched observations (same seed, same rotation, same split).

    Wilcoxon is primary: n is small and the differences are not assumed normal.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired arrays must match"
    d = a - b
    out = {"label": f"{label_a} - {label_b}", "n_pairs": int(len(d)),
           "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
           "wins": int(np.sum(d > 0)), "ties": int(np.sum(d == 0)),
           "losses": int(np.sum(d < 0))}
    if len(d) < 2 or np.allclose(d, 0):
        out["p_wilcoxon"] = float("nan")
        out["p_ttest"] = float("nan")
        return out
    try:
        out["p_wilcoxon"] = float(sps.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        out["p_wilcoxon"] = float("nan")
    out["p_ttest"] = float(sps.ttest_rel(a, b).pvalue)
    out["boot_ci95_diff"] = list(boot_ci(d))
    sd = np.std(d, ddof=1)
    out["cohens_dz"] = float(np.mean(d) / sd) if sd > 0 else float("inf")
    # rank-biserial: the effect size that matches Wilcoxon
    nz = d[d != 0]
    if len(nz):
        r = sps.rankdata(np.abs(nz))
        out["rank_biserial"] = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
    return out


# --------------------------------------------------------------------------
# 3. our seeds against a published constant
# --------------------------------------------------------------------------

def vs_published(x, published, name=""):
    """One-sample test of our seed distribution against a transcribed constant.

    This is deliberately weaker than `paired`: the other method's split, seeding
    and variance are unknown, so the only defensible statement is about where
    OUR distribution sits relative to their reported point.
    """
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    out = {"name": name, "published": float(published), "n": int(len(x)),
           "our_mean": float(np.mean(x)) if len(x) else float("nan"),
           "seeds_above": int(np.sum(x > published)),
           "min_seed": float(np.min(x)) if len(x) else float("nan"),
           "evidence": "weak: unpaired, other side has no reported variance"}
    if len(x) > 1:
        out["p_onesample_t"] = float(sps.ttest_1samp(x, published).pvalue)
        out["p_wilcoxon_signed"] = float(
            sps.wilcoxon(x - published).pvalue) if not np.allclose(x, published) else float("nan")
        out["boot_ci95"] = list(boot_ci(x))
        out["ci_excludes_published"] = bool(out["boot_ci95"][0] > published
                                            or out["boot_ci95"][1] < published)
    return out


# --------------------------------------------------------------------------
# multiplicity
# --------------------------------------------------------------------------

def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. Returns adjusted p-values in the input order."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    order = np.argsort(np.where(ok, p, np.inf))
    m = int(ok.sum())
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(order[:m]):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def fmt_ci(ci):
    if ci is None or not np.isfinite(ci[0]):
        return "[  --  ,   --  ]"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def report(tag_filter=None):
    runs, rots = load_runs(), load_rotations()
    out = {"main": {}, "paired": [], "vs_published": [], "ablations": {}}
    if runs.empty:
        print("no results yet")
        return out

    if tag_filter:
        runs = runs[runs.tag.str.contains(tag_filter, na=False)]
        if not rots.empty:
            rots = rots[rots.tag.str.contains(tag_filter, na=False)]

    print("=" * 78)
    print("1. MAIN TABLE CELLS  (n seeds; sd is ddof=1; CI is bootstrap over seeds)")
    print("=" * 78)
    # strictly the main-table tag: falling back to "everything" would report an
    # ablation arm as a main-table cell in exactly the situation where nobody
    # would notice, namely before the main runs have finished
    main = runs[is_main(runs)] if not runs.empty else runs
    if main.empty:
        print("  (no main-table runs yet)")
    for (ds, meth), g in main.groupby(["dataset", "method"]):
        print(f"\n{ds}  [{meth}]  seeds={sorted(g.seed.tolist())}")
        cell = {}
        for metric in ("best_auroc", "best_aupr", "valsel_auroc", "valsel_aupr"):
            s = summarise(g[metric].values)
            cell[metric] = s
            if s["n"] > 1:
                print(f"  {metric:20s} {s['mean']:.4f} +- {s['sd']:.4f}   "
                      f"boot95 {fmt_ci(s['boot_ci95'])}   "
                      f"t95 {fmt_ci(s['t_ci95'])}   {s['values']}")
            elif s["n"] == 1:
                print(f"  {metric:20s} {s['mean']:.4f}  (single run)")
        out["main"][f"{ds}/{meth}"] = cell

    # ---- paired: OUTPOST vs any baseline we ran, matched on seed+rotation ----
    if not rots.empty:
        print("\n" + "=" * 78)
        print("2. PAIRED vs RE-RUN BASELINE  (unit = seed x rotation, same split)")
        print("=" * 78)
        # Main-table arms only. `rots` holds every rotation of every campaign,
        # so without this the OUTPOST side picks up each ablation arm as another
        # row at the same (seed, rotation) - the join then multiplies rows, the
        # two sides stop matching, and any number that did come out would be an
        # average of the method with its own ablations.
        main_rots = rots[is_main(rots)]
        for ds in sorted(main_rots.dataset.unique()):
            sub = main_rots[main_rots.dataset == ds]
            methods = sorted(sub.method.unique())
            if "outpost" not in methods:
                continue
            for other in [m for m in methods if m != "outpost"]:
                A = sub[sub.method == "outpost"].set_index(["seed", "rotation"])
                B = sub[sub.method == other].set_index(["seed", "rotation"])
                if A.index.has_duplicates or B.index.has_duplicates:
                    print(f"  {ds}: duplicate (seed, rotation) entries, skipped")
                    continue
                common = A.index.intersection(B.index)
                if len(common) < 2:
                    continue
                for metric in ("best_auroc", "best_aupr", "valsel_auroc",
                               "valsel_aupr"):
                    a = A.loc[common, metric].values
                    b = B.loc[common, metric].values
                    if not (np.isfinite(a).all() and np.isfinite(b).all()):
                        continue
                    r = paired(a, b, "outpost", other)
                    r.update(dataset=ds, metric=metric)
                    out["paired"].append(r)
                    print(f"  {ds:11s} {metric:18s} d={r['mean_diff']:+.4f}  "
                          f"W/T/L {r['wins']}/{r['ties']}/{r['losses']}  "
                          f"p_wilcoxon={r['p_wilcoxon']:.4g}  "
                          f"p_t={r['p_ttest']:.4g}  "
                          f"n={r['n_pairs']}")

    # ---- vs published constants ----
    pub_path = os.path.join(ROOT, "analysis/tables/published_baselines.csv")
    if os.path.exists(pub_path):
        pub = pd.read_csv(pub_path)
        print("\n" + "=" * 78)
        print("3. vs PUBLISHED CONSTANTS  (unpaired; weaker evidence, see docstring)")
        print("=" * 78)
        colmap = {"photo": "Photo", "computers": "Computers", "cs": "CS",
                  "yelp": "Yelp", "ogbn-arxiv": "ogbn-arxiv", "ogbn-mag": "ogbn-mag"}
        ours = main[main.method == "outpost"]
        for ds, g in ours.groupby("dataset"):
            for metric, suffix in (("best_auroc", "AUC-ROC"), ("best_aupr", "AUC-PR")):
                col = f"{colmap.get(ds, ds)}_{suffix}"
                if col not in pub.columns:
                    continue
                comp = pub[pub.Method != "OUTPOST"][["Method", col]].dropna()
                if comp.empty:
                    continue
                best_row = comp.loc[comp[col].idxmax()]
                r = vs_published(g[metric].values, float(best_row[col]),
                                 name=f"{ds}/{metric} vs best published "
                                      f"({best_row['Method']})")
                out["vs_published"].append(r)
                if r["n"] > 1:
                    print(f"  {ds:11s} {suffix:8s} ours {r['our_mean']:.4f} vs "
                          f"{best_row['Method']} {r['published']:.4f}  "
                          f"seeds_above={r['seeds_above']}/{r['n']}  "
                          f"p_t={r['p_onesample_t']:.4g}  "
                          f"CI_excl={r['ci_excludes_published']}")

    # ---- ablations and the sensitivity sweep, Holm-corrected per dataset ----
    # The full-system arm is the A_main run itself, at the same seeds: an
    # ablation differs from it by exactly one flag, so re-running the default
    # configuration under a second tag would only spend GPU hours to reproduce
    # numbers that already exist.
    abl = runs[runs.tag.str.match(r"^([CD]_|L_lean)", na=False)]
    if not abl.empty:
        print("\n" + "=" * 78)
        print("4. ABLATIONS AND SENSITIVITY")
        print("   paired on seed against the full system (its base arm, same seeds)")
        print("   Holm-Bonferroni within each dataset's family of arms")
        print("=" * 78)
        for ds, g in abl.groupby("dataset"):
            base_tag = ABL_BASE.get(ds, "A_main")
            full = runs[(runs.dataset == ds) & (runs.tag == base_tag)
                        & (runs.method == "outpost")]
            if full.empty:
                print(f"  {ds}: no {base_tag} runs yet to compare against")
                continue
            ps, recs = [], []
            for tag, gg in sorted(g.groupby("tag")):
                common = sorted(set(full.seed) & set(gg.seed))
                if len(common) < 2:
                    continue
                for metric in ("best_aupr", "best_auroc"):
                    a = full.set_index("seed").loc[common, metric].values
                    b = gg.set_index("seed").loc[common, metric].values
                    r = paired(b, a, tag, "full")
                    r.update(dataset=ds, metric=metric, n_seeds=len(common))
                    if metric == "best_aupr":       # the corrected family
                        ps.append(r.get("p_wilcoxon", np.nan))
                        recs.append(r)
                    out.setdefault("ablation_rows", []).append(r)
            if not recs:
                continue
            adj = holm(ps)
            print(f"\n  {ds}   (delta = arm minus full system, AUC-PR)")
            for r, p in zip(recs, adj):
                r["p_holm"] = None if not np.isfinite(p) else float(p)
                ph = "  --  " if not np.isfinite(p) else f"{p:.4f}"
                print(f"    {r['label']:26s} n={r['n_seeds']}  "
                      f"d={r['mean_diff']:+.4f}  "
                      f"CI95 {fmt_ci(r.get('boot_ci95_diff', [np.nan, np.nan]))}  "
                      f"W/T/L {r['wins']}/{r['ties']}/{r['losses']}  "
                      f"p={r.get('p_wilcoxon', float('nan')):.4f}  p_holm={ph}")
            out["ablations"][ds] = recs
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    res = report(a.tag)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
        print(f"\nwritten -> {a.json}")
