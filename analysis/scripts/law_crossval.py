#!/usr/bin/env python3
"""Does the detectability law predict anything out of sample, once the
definitionally-true part is removed?

Two findings, and the second is the one that matters.

1. Naively cross-validating the two-regime form against the one-regime form makes
   the two-regime form look overwhelming (6/6 folds, MAE 0.031 vs 0.143). That
   result is an ARTEFACT and is not reported as evidence. `best_auc` is defined as
   max(auc_hop0..auc_hop3), and the feature-visible regime is defined as
   auc_hop3 <= auc_hop0. Conditioning on propagation not helping makes hop-0 the
   maximum: in 7 of 10 feature-visible classes best_auc IS auc_hop0 exactly, and
   the mean gap is 0.005. Predicting best_auc from auc_hop0 there is predicting a
   variable from itself.

2. The law's real content is the PROPAGATION branch: for classes propagation
   helps, does the same-class fraction predict the achievable AUC on a graph the
   fit never saw? That is tested here against an intercept-only baseline, leaving
   one dataset out. If the line cannot beat "predict the training mean", the law
   has no out-of-sample content even in the branch that is not definitional.

    python analysis/scripts/law_crossval.py  ->  analysis/tables/law_crossval.json
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "analysis/scripts")
from detectability_law import load, SMALL  # noqa: E402


def main():
    ds = sorted(f.split("spectral_")[1][:-4]
                for f in os.listdir("analysis/tables") if f.startswith("spectral_"))
    df = load(ds).dropna(subset=["med_same_frac", "best_auc", "auc_hop0",
                                 "smoothing_gain"])

    # 1. how definitional is the feature-visible branch?
    feat = df[df.smoothing_gain <= 0]
    circ = {"n_feature_visible": int(len(feat)),
            "best_equals_hop0": int((feat.best_auc == feat.auc_hop0).sum()),
            "mean_abs_gap": round(float((feat.best_auc - feat.auc_hop0).abs().mean()), 5)}

    # 2. the real test, inside the propagation regime only
    prop_all = df[df.smoothing_gain > 0]

    def cv(sub):
        F, L, M = [], [], []
        for held in sorted(sub.dataset.unique()):
            tr, te = sub[sub.dataset != held], sub[sub.dataset == held]
            if len(tr) < 4 or te.empty:
                continue
            b, a = np.polyfit(tr.med_same_frac, tr.best_auc, 1)
            el = np.abs(te.best_auc.values - (a + b * te.med_same_frac.values))
            em = np.abs(te.best_auc.values - tr.best_auc.mean())
            F.append({"held_out": held, "n_classes": int(len(te)),
                      "mae_same_frac_line": round(float(el.mean()), 4),
                      "mae_intercept_only": round(float(em.mean()), 4),
                      "line_better": bool(el.mean() < em.mean())})
            L += list(el); M += list(em)
        return F, np.array(L), np.array(M)

    # the same test with the two OGB graphs removed: they hold 13 of the 21
    # propagation-regime classes, so whether they are in decides the answer
    no_ogb = prop_all[~prop_all.dataset.str.startswith("ogbn")]
    f2, l2, m2 = cv(no_ogb)
    out_no_ogb = {"n_folds": len(f2), "n_classes": int(len(l2)),
                  "pooled_mae_same_frac_line": round(float(l2.mean()), 4) if len(l2) else None,
                  "pooled_mae_intercept_only": round(float(m2.mean()), 4) if len(m2) else None,
                  "folds_line_better": sum(x["line_better"] for x in f2),
                  "caveat": "rests on very few classes; not evidence on its own"}

    prop = prop_all
    folds, e_line, e_mean = [], [], []
    for held in sorted(prop.dataset.unique()):
        tr, te = prop[prop.dataset != held], prop[prop.dataset == held]
        if len(tr) < 4 or te.empty:
            continue
        b, a = np.polyfit(tr.med_same_frac, tr.best_auc, 1)
        el = np.abs(te.best_auc.values - (a + b * te.med_same_frac.values))
        em = np.abs(te.best_auc.values - tr.best_auc.mean())
        folds.append({"held_out": held, "n_classes": int(len(te)),
                      "mae_same_frac_line": round(float(el.mean()), 4),
                      "mae_intercept_only": round(float(em.mean()), 4),
                      "line_better": bool(el.mean() < em.mean())})
        e_line += list(el); e_mean += list(em)
    L, M = np.array(e_line), np.array(e_mean)
    out = {"fold_unit": "dataset", "regime": "propagation (smoothing gain > 0)",
           "feature_visible_is_definitional": circ,
           "n_folds": len(folds), "n_classes": int(len(L)),
           "pooled_mae_same_frac_line": round(float(L.mean()), 4) if len(L) else None,
           "pooled_mae_intercept_only": round(float(M.mean()), 4) if len(M) else None,
           "improvement": round(float(M.mean() - L.mean()), 4) if len(L) else None,
           "folds_line_better": sum(f["line_better"] for f in folds),
           "folds": folds,
           "without_ogb": out_no_ogb,
           "classes_per_dataset": {k: int(v) for k, v in
                                   prop_all.groupby("dataset").size().items()}}
    if len(L) > 5:
        try:
            from scipy.stats import wilcoxon
            out["p_wilcoxon_classes"] = round(float(wilcoxon(M, L).pvalue), 4)
        except Exception:
            pass
    json.dump(out, open("analysis/tables/law_crossval.json", "w"), indent=1)

    print("FEATURE-VISIBLE BRANCH IS DEFINITIONAL, not predictive:")
    print(f"  best_auc == auc_hop0 in {circ['best_equals_hop0']}/{circ['n_feature_visible']}"
          f" classes; mean |gap| {circ['mean_abs_gap']}")
    print("  (best_auc is max over hops; the regime is defined by hop3 <= hop0)")
    print("\nPROPAGATION REGIME, leave-one-dataset-out:")
    print(f"{'held out':12s} {'n':>3s} {'MAE line':>10s} {'MAE mean':>10s}  better")
    for f in folds:
        print(f"{f['held_out']:12s} {f['n_classes']:3d} {f['mae_same_frac_line']:10.4f} "
              f"{f['mae_intercept_only']:10.4f}  {'line' if f['line_better'] else 'MEAN'}")
    if len(L):
        print(f"\npooled MAE   same_frac line {L.mean():.4f}   intercept-only "
              f"{M.mean():.4f}   improvement {M.mean()-L.mean():+.4f}")
        print(f"folds where the line wins: {out['folds_line_better']}/{len(folds)}"
              + (f"   p={out['p_wilcoxon_classes']}" if 'p_wilcoxon_classes' in out else ""))
    o = out["without_ogb"]
    print(f"\nsame test without the two OGB graphs: line {o['pooled_mae_same_frac_line']} "
          f"vs mean {o['pooled_mae_intercept_only']} over only {o['n_classes']} classes "
          f"in {o['n_folds']} folds -- too thin to carry a claim")
    print("\n-> analysis/tables/law_crossval.json")


if __name__ == "__main__":
    main()
