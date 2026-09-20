"""The detectability law, tested rather than just reported.

The claim of METHODOLOGY section 6 is that an anomaly class's same-class
neighbour fraction predicts its best achievable AUC-ROC. It was reported as a
single Spearman rho over 16 classes from 4 datasets. Three things were missing,
and each is a question a referee asks:

1. Is rho significant once the datasets are respected? The 16 classes are not
   independent draws: classes from one graph share its construction. The
   asymptotic Spearman p-value ignores this. Reported here instead are (a) a
   permutation test that shuffles WITHIN each dataset, which destroys the
   between-class signal while preserving every dataset-level idiosyncrasy, and
   (b) a bootstrap over datasets (cluster bootstrap), which is the interval
   that admits "one dataset is driving it".

2. Does it hold out of sample? A correlation fitted and reported on the same 16
   points says nothing about prediction. Leave-one-dataset-out cross-validation
   is reported: fit on three graphs, predict the fourth.

3. Does it survive more classes? The stated threat to validity was n=16. With
   ogbn-arxiv and ogbn-mag the sample is 35 classes over 6 graphs.

    python analysis/scripts/detectability_law.py
    python analysis/scripts/detectability_law.py --no-ogb   # the original 16
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SMALL = ["photo", "computers", "cs", "yelp", "amazon", "tfinance"]
OGB = ["ogbn-arxiv", "ogbn-mag"]
RNG = np.random.default_rng(0)
N_PERM = 20000
N_BOOT = 10000


def load(datasets):
    rows = []
    for ds in datasets:
        p = f"analysis/tables/spectral_{ds}.csv"
        if not os.path.exists(p):
            print(f"  [missing] {p} - run spectral_diagnostic.py --dataset {ds}")
            continue
        d = pd.read_csv(p)
        d["dataset"] = ds
        d["best_auc"] = d[[f"auc_hop{k}" for k in range(4)]].max(axis=1)
        rows.append(d)
    if not rows:
        sys.exit("no spectral_*.csv found")
    return pd.concat(rows, ignore_index=True)


def perm_test_within_dataset(x, y, groups, n=N_PERM):
    """Permute y within each dataset. Preserves per-dataset level and spread, so
    the null is 'same_frac carries no BETWEEN-class information beyond whatever
    the dataset itself explains'."""
    obs = sps.spearmanr(x, y).statistic
    groups = np.asarray(groups)
    y = np.asarray(y, float)
    count = 0
    for _ in range(n):
        yp = y.copy()
        for g in np.unique(groups):
            m = groups == g
            yp[m] = RNG.permutation(yp[m])
        if abs(sps.spearmanr(x, yp).statistic) >= abs(obs) - 1e-12:
            count += 1
    return obs, (count + 1) / (n + 1)


def cluster_bootstrap(df, n=N_BOOT):
    """Resample DATASETS with replacement, not classes: the unit that could have
    come out differently is the benchmark graph."""
    ds = df.dataset.unique()
    reps = []
    for _ in range(n):
        pick = RNG.choice(ds, size=len(ds), replace=True)
        sub = pd.concat([df[df.dataset == d] for d in pick], ignore_index=True)
        if sub.med_same_frac.nunique() < 3:
            continue
        r = sps.spearmanr(sub.med_same_frac, sub.best_auc).statistic
        if np.isfinite(r):
            reps.append(r)
    return np.percentile(reps, [2.5, 97.5]), len(reps)


def loco(df):
    """Leave-one-dataset-out: fit rank regression on the others, predict this one.

    Reported as the Spearman correlation between predicted and actual on the
    held-out graph, plus mean absolute error in AUC units from an OLS fit.
    """
    out = []
    for ds in df.dataset.unique():
        tr, te = df[df.dataset != ds], df[df.dataset == ds]
        if len(te) < 2 or len(tr) < 3:
            out.append({"held_out": ds, "n": len(te), "note": "too few points"})
            continue
        b, a = np.polyfit(tr.med_same_frac, tr.best_auc, 1)
        pred = a + b * te.med_same_frac
        r = sps.spearmanr(pred, te.best_auc).statistic if len(te) > 2 else np.nan
        out.append({"held_out": ds, "n": int(len(te)),
                    "mae": float(np.mean(np.abs(pred - te.best_auc))),
                    "spearman_in_heldout": None if not np.isfinite(r) else float(r)})
    return out


def two_regime(df):
    """The law as Amazon says it must be stated.

    Naive form: best_auc ~ same_frac, one line through every class. Amazon
    (same_frac 0.080, hop-0 AUC 0.878) was predicted at 0.615 by that line and
    came in at 0.953 - 0.105 above the 95% band. The class was FEATURE-VISIBLE:
    its anomalies are nearly separable from raw features and propagation only
    hurts (smoothing gain -0.174). Such classes are the second regime P8 named
    in ogbn-mag class 263, and the same-class fraction does not govern them.

    Two-regime form: split on the sign of the smoothing gain. Classes that
    propagation HELPS (gain > 0) follow the same_frac line; classes it HURTS are
    pinned near their hop-0 AUC and the line does not apply. Reported: the fit
    and residual sd within each regime, and the naive line's residual on the
    feature-visible classes, which is where it fails.
    """
    prop = df[df.smoothing_gain > 0]
    feat = df[df.smoothing_gain <= 0]
    out = {"n_propagation_regime": int(len(prop)), "n_feature_visible": int(len(feat))}
    if len(prop) >= 3:
        b, a = np.polyfit(prop.med_same_frac, prop.best_auc, 1)
        res = prop.best_auc - (a + b * prop.med_same_frac)
        rho = sps.spearmanr(prop.med_same_frac, prop.best_auc).statistic
        out["propagation_fit"] = {"intercept": float(a), "slope": float(b),
                                  "residual_sd": float(res.std(ddof=1)), "spearman": float(rho)}
    if len(feat) >= 3:
        # in this regime the predictor is the hop-0 AUC, not same_frac
        rho_sf = sps.spearmanr(feat.med_same_frac, feat.best_auc).statistic
        rho_h0 = sps.spearmanr(feat.auc_hop0, feat.best_auc).statistic
        gap = feat.best_auc - feat.auc_hop0
        out["feature_visible"] = {"spearman_same_frac": float(rho_sf),
                                  "spearman_hop0": float(rho_h0),
                                  "best_minus_hop0_mean": float(gap.mean()),
                                  "best_minus_hop0_sd": float(gap.std(ddof=1))}
    # the naive line's error on the two regimes
    b, a = np.polyfit(df.med_same_frac, df.best_auc, 1)
    for name, sub in (("propagation", prop), ("feature_visible", feat)):
        if len(sub):
            out[f"naive_line_mae_{name}"] = float(np.mean(np.abs(sub.best_auc - (a + b * sub.med_same_frac))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ogb", action="store_true",
                    help="the original four graphs only, for comparison")
    ap.add_argument("--json", default="analysis/tables/detectability_law.json")
    ap.add_argument("--holdout", default=None,
                    help="fit on every other graph and predict this one. "
                         "--holdout amazon reproduces P23/P24: the law was "
                         "fitted on 35 classes and Amazon's prediction was "
                         "registered before any model ran on it.")
    a = ap.parse_args()

    datasets = SMALL if a.no_ogb else SMALL + OGB
    df = load(datasets)
    df = df[np.isfinite(df.med_same_frac) & np.isfinite(df.best_auc)]
    held = None
    if a.holdout:
        held = df[df.dataset == a.holdout].copy()
        df = df[df.dataset != a.holdout]
        print(f"[holdout] {a.holdout}: {len(held)} class(es) removed from the fit")

    print("=" * 74)
    print(f"DETECTABILITY LAW   n = {len(df)} anomaly classes over "
          f"{df.dataset.nunique()} graphs")
    print("=" * 74)
    for ds, g in df.groupby("dataset", sort=False):
        print(f"  {ds:12s} {len(g):2d} classes   "
              f"same_frac {g.med_same_frac.min():.2f}-{g.med_same_frac.max():.2f}   "
              f"best AUC {g.best_auc.min():.2f}-{g.best_auc.max():.2f}")

    rho, p_asym = sps.spearmanr(df.med_same_frac, df.best_auc)
    print(f"\n  Spearman rho            {rho:+.3f}   (asymptotic p = {p_asym:.2e})")
    print( "                          the asymptotic p treats 35 classes as 35")
    print( "                          independent draws, which they are not")

    obs, p_perm = perm_test_within_dataset(df.med_same_frac.values,
                                           df.best_auc.values, df.dataset.values)
    print(f"  within-dataset permutation p    {p_perm:.4g}   ({N_PERM} shuffles)")

    ci, nrep = cluster_bootstrap(df)
    print(f"  cluster bootstrap 95% CI on rho [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
          f"({nrep} usable resamples of the {df.dataset.nunique()} graphs)")

    print("\n  leave-one-dataset-out prediction:")
    lo = loco(df)
    for r in lo:
        if "note" in r:
            print(f"    {r['held_out']:12s} n={r['n']:2d}  {r['note']}")
        else:
            s = ("  rho_in_heldout " + f"{r['spearman_in_heldout']:+.3f}"
                 if r["spearman_in_heldout"] is not None else "")
            print(f"    {r['held_out']:12s} n={r['n']:2d}  MAE {r['mae']:.3f} AUC{s}")

    # the real-vs-semisynthetic split, which is the benchmark-validity claim
    REAL = ["yelp", "amazon", "tfinance"]
    real = df[df.dataset.isin(REAL)]
    synth = df[~df.dataset.isin(REAL) & ~df.dataset.isin(OGB)]
    for _, r in real.iterrows():
        below = int((synth.med_same_frac < r.med_same_frac).sum())
        print(f"\n  real fraud ({r.dataset}) same_frac {r.med_same_frac:.3f}   "
              f"semi-synthetic classes below it: {below}/{len(synth)}")

    tr = two_regime(df)
    print(f"\n  TWO-REGIME FORM   propagation regime n={tr['n_propagation_regime']}, "
          f"feature-visible n={tr['n_feature_visible']}")
    if "propagation_fit" in tr:
        f = tr["propagation_fit"]
        print(f"    propagation:      best_auc = {f['intercept']:.3f} + {f['slope']:.3f} x same_frac   "
              f"rho {f['spearman']:+.3f}   residual sd {f['residual_sd']:.3f}")
    if "feature_visible" in tr:
        f = tr["feature_visible"]
        print(f"    feature-visible:  rho(same_frac) {f['spearman_same_frac']:+.3f}   "
              f"rho(hop-0) {f['spearman_hop0']:+.3f}   best - hop0 = "
              f"{f['best_minus_hop0_mean']:+.3f} +- {f['best_minus_hop0_sd']:.3f}")
    print(f"    naive line MAE:   propagation {tr.get('naive_line_mae_propagation', float('nan')):.3f}   "
          f"feature-visible {tr.get('naive_line_mae_feature_visible', float('nan')):.3f}")

    if held is not None and len(held):
        b, a_ = np.polyfit(df.med_same_frac, df.best_auc, 1)
        sd = (df.best_auc - (a_ + b * df.med_same_frac)).std(ddof=1)
        print(f"\n  OUT-OF-SAMPLE on {a.holdout}:")
        for _, r in held.iterrows():
            naive = a_ + b * r.med_same_frac
            regime = "feature-visible" if r.smoothing_gain <= 0 else "propagation"
            print(f"    class {int(r['class'])}: same_frac {r.med_same_frac:.3f}, {regime}, "
                  f"hop-0 {r.auc_hop0:.3f}")
            print(f"      naive law predicts {naive:.3f} (band {naive-2*sd:.3f}-{naive+2*sd:.3f}); "
                  f"diagnostic best-over-hops {r.best_auc:.3f}")

    out = {"n_classes": int(len(df)), "n_datasets": int(df.dataset.nunique()),
           "datasets": list(df.dataset.unique()),
           "holdout": a.holdout,
           "two_regime": tr,
           "spearman_rho": float(rho), "p_asymptotic": float(p_asym),
           "p_permutation_within_dataset": float(p_perm),
           "cluster_bootstrap_ci95": [float(ci[0]), float(ci[1])],
           "leave_one_dataset_out": lo,
           "classes": df[["dataset", "class", "med_same_frac", "best_auc",
                          "smoothing_gain"]].to_dict("records")}
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"\nwritten -> {a.json}")


if __name__ == "__main__":
    main()
