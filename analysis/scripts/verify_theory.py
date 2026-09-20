"""Verify Corollary 1 of THEORY.md against the Phase-0 measurements.

Prediction:  sign(smoothing_gain_c) == sign(h_c - d_c^{-1/2})
where h_c = median same-class neighbor fraction (already measured) and
d_c = median closed-neighborhood degree of class c's anomaly nodes.

No training. Recomputes per-class degree via node_metrics (fast), reads the
smoothing gains from analysis/tables/spectral_*.csv, reports agreement.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spectral_diagnostic import load_dataset, node_metrics

DATASETS = ["photo", "computers", "cs", "yelp"]
rows = []
for ds in DATASETS:
    x, edge_index, y, anomaly_classes, normal_classes = load_dataset(ds)
    m = node_metrics(x, edge_index, y)
    csv = pd.read_csv(f"analysis/tables/spectral_{ds}.csv")
    for c in anomaly_classes:
        mask = (y == c) & m["has_nbrs"]
        d_c = float(np.median(m["deg"][mask]))          # closed-nbhd degree (incl self via +1? see note)
        d_eff = d_c + 1.0                                # closed neighborhood
        h_c = float(np.median(m["same_frac"][mask]))
        gain = float(csv.loc[csv["class"] == c, "smoothing_gain"].iloc[0])
        h_star = d_eff ** -0.5
        pred_pos = h_c > h_star
        obs_pos = gain > 0
        rows.append(dict(dataset=ds, cls=c, h=h_c, d=d_eff, h_star=round(h_star, 3),
                         gain=round(gain, 3), pred="AMP" if pred_pos else "ERASE",
                         obs="AMP" if obs_pos else "ERASE", ok=(pred_pos == obs_pos)))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
acc = df["ok"].mean()
print(f"\nCorollary-1 sign agreement: {df['ok'].sum()}/{len(df)} = {acc:.0%}")
# also report rank correlation of the continuous margin vs gain
from scipy.stats import spearmanr
margin = df["h"] - df["d"] ** -0.5
rho, p = spearmanr(margin, df["gain"])
print(f"Spearman( h - d^-1/2 , smoothing_gain ) = {rho:+.3f} (p={p:.2e})")
