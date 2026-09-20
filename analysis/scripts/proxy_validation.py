"""Step-1 validation: label-free proxies for the scattered-anomaly signature.

Context
-------
Phase 0 (see PHASE0_FINDINGS.md) established that scattered anomalies -- nodes
with few same-class neighbors -- are erased by low-pass propagation and then
certified normal by confidence-gated self-training. The Phase-0 instrument
(same-class neighbor fraction) uses ground-truth labels, so a deployable gate
needs *label-free* per-node proxies. This script tests the candidates.

Candidate proxies (all label-free)
----------------------------------
  dirichlet   local Dirichlet energy of the raw feature signal (sym-norm)
  feat_dissim 1 - cos(x_v, neighbor mean)
  hop_dis     signed cross-hop disagreement of the prototype detector:
              rank(s_0)(v) - rank(s_K)(v). Positive = the node carries hop-0
              anomaly evidence that propagation destroys -- the predicted
              signature of a scattered anomaly.
  s0          hop-0 prototype score itself (unsmoothed evidence)
  deg_inv     1/degree (low-degree nodes smooth toward few neighbors)

Validation criteria
-------------------
V1 (tracking)  Among anomaly nodes only: Spearman(proxy, same-class frac).
    Negative rho expected for proxies that flag scatteredness (scattered =
    low same-frac = high proxy).
V2 (gate simulation -- the criterion that matters)  Let C be the pseudo-normal
    candidate region: nodes in the bottom q% of the *smoothed* score s_K
    (q in {30, 50}), mirroring the tau_minus region of self-training. Report:
      - contamination: fraction of C that is truly anomalous (the poisoning
        rate Phase 0 identified);
      - per-proxy ROC-AUC separating anomalies from normals *within C*;
      - caught@10: recall of C's anomalies in the top decile of the proxy.
    A proxy is gate-worthy if it materially exceeds 0.5 AUC within C, where
    the smoothed score is uninformative by construction.

Outputs: analysis/tables/proxy_<dataset>.csv + stdout tables.
Reproducibility: seed 42 throughout (inherited from spectral_diagnostic).

Usage:  python analysis/proxy_validation.py --dataset all
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spectral_diagnostic import (  # noqa: E402
    HOPS, load_dataset, node_metrics, prototype_scores_by_hop)


def rank01(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def run(dataset):
    print(f"\n{'=' * 72}\n[{dataset}]")
    x, edge_index, y, anomaly_classes, normal_classes = load_dataset(dataset)
    m = node_metrics(x, edge_index, y)
    scores = prototype_scores_by_hop(x, edge_index, y, normal_classes)
    K = max(HOPS)
    is_anom = np.isin(y, anomaly_classes)

    proxies = {
        "dirichlet": m["dirichlet"],
        "feat_dissim": m["feat_dissim"],
        "hop_dis": rank01(scores[0]) - rank01(scores[K]),
        "s0": scores[0],
        "deg_inv": 1.0 / np.maximum(m["deg"], 1.0),
    }

    # ---- V1: tracking scatteredness among anomaly nodes ----
    am = is_anom & m["has_nbrs"]
    print(f"V1 tracking (n_anom={int(am.sum())}): Spearman(proxy, same_frac) "
          f"-- negative = flags scatteredness")
    v1 = {}
    for name, p in proxies.items():
        rho, pv = spearmanr(p[am], m["same_frac"][am])
        v1[name] = rho
        print(f"  {name:>12}: rho={rho:+.3f} (p={pv:.1e})")

    # ---- V2: gate simulation in the pseudo-normal candidate region ----
    sK = scores[K]
    rows = []
    for q in (30, 50):
        thr = np.percentile(sK, q)
        C = sK <= thr
        cont = float(is_anom[C].mean())
        n_anom_C = int(is_anom[C].sum())
        print(f"\nV2 gate simulation: C = bottom {q}% of smoothed score "
              f"(|C|={int(C.sum())}, contamination={cont:.4f}, "
              f"anomalies inside={n_anom_C})")
        # how much of each anomaly class is hiding in C
        for c in anomaly_classes:
            n_c = int(((y == c) & C).sum())
            print(f"    class {c}: {n_c}/{int((y == c).sum())} "
                  f"({n_c / max(int((y == c).sum()), 1):.1%}) inside C")
        if n_anom_C < 10:
            print("    (too few anomalies in C for stable AUC; skipping)")
            continue
        yC = is_anom[C].astype(int)
        for name, p in proxies.items():
            auc = roc_auc_score(yC, p[C])
            top = p[C] >= np.percentile(p[C], 90)
            caught = float(yC[top].sum() / max(yC.sum(), 1))
            print(f"  {name:>12}: AUC-in-C={auc:.3f}  caught@10={caught:.1%}")
            rows.append({"dataset": dataset, "region_q": q, "proxy": name,
                         "auc_in_C": auc, "caught_at_10": caught,
                         "contamination": cont, "v1_rho": v1[name]})
        # reference: the smoothed score itself inside C (should be ~chance)
        auc_ref = roc_auc_score(yC, sK[C])
        print(f"  {'[smoothed]':>12}: AUC-in-C={auc_ref:.3f}  (reference)")

    os.makedirs("analysis/tables", exist_ok=True)
    out = f"analysis/tables/proxy_{dataset}.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["photo", "computers", "cs", "yelp", "all"])
    args = ap.parse_args()
    targets = (["photo", "computers", "cs", "yelp"]
               if args.dataset == "all" else [args.dataset])
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.manual_seed(42)
    np.random.seed(42)
    for d in targets:
        run(d)
