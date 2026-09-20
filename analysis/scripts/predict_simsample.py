"""PRE-REGISTERED prediction: where will SimSample help, and where will it hurt?

Run and COMMITTED BEFORE the Computers/CS experiments. This turns the Phase-0
detectability law from an explanation of past results into a forecast on
datasets it has never been tested against.

The rule (fixed from the two known points, not fitted afterwards)
----------------------------------------------------------------
SimSample purifies neighbourhoods. From the measured law:

  R1  If anomalies are SCATTERED (low same-class neighbour fraction) and the
      graph is DENSE (many neighbours to choose from) and features alone are
      WEAK, purification removes camouflage edges -> SimSample HELPS.
  R2  If anomalies are CLUSTERED (high same-class fraction), propagation is
      already amplifying them; purification narrows the receptive field and
      removes the contrast that made them visible -> SimSample HURTS.
  R3  If features alone ALREADY separate the class (high hop-0 AUC), the model
      barely depends on propagation, so changing the sampler should matter
      LITTLE either way -> SimSample is roughly NEUTRAL.

Calibration points (already measured):
  Yelp  : same-frac 0.16, deg ~167, hop-0 AUC 0.63 (weak)  -> +0.051 PR  [R1]
  Photo : same-frac 0.75, deg ~24,  hop-0 AUC 0.59 (weak)  -> -0.081 PR  [R2]

Predictions recorded by this script for the untested datasets are written to
analysis/tables/prediction_simsample.csv and must not be edited afterwards.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from spectral_diagnostic import load_dataset, node_metrics  # noqa: E402

CAL = {  # measured, for reference
    "yelp":  dict(effect=+0.051),
    "photo": dict(effect=-0.081),
}

rows = []
for ds in ("photo", "computers", "cs", "yelp"):
    x, edge_index, y, anomaly_classes, normal_classes = load_dataset(ds)
    m = node_metrics(x, edge_index, y)
    spec = pd.read_csv(f"analysis/tables/spectral_{ds}.csv")
    anom = np.isin(y, anomaly_classes) & m["has_nbrs"]
    same_frac = float(np.median(m["same_frac"][anom]))     # anomaly scatter
    deg = float(np.median(m["deg"][anom]))                 # choice available
    hop0 = float(spec["auc_hop0"].mean())                  # feature visibility

    # --- apply the rule ---
    if hop0 >= 0.80:
        pred, why = "NEUTRAL (small |effect|)", "R3 features already separate"
    elif same_frac <= 0.35 and deg >= 50:
        pred, why = "HELPS (positive)", "R1 scattered + dense + weak features"
    else:
        pred, why = "HURTS (negative)", "R2 clustered anomalies"

    rows.append(dict(dataset=ds, anomaly_same_frac=round(same_frac, 3),
                     median_degree=round(deg, 1), mean_hop0_auc=round(hop0, 3),
                     PREDICTION=pred, rule=why,
                     measured_effect_PR=CAL.get(ds, {}).get("effect", "UNTESTED")))

df = pd.DataFrame(rows)
os.makedirs("analysis/tables", exist_ok=True)
df.to_csv("analysis/tables/prediction_simsample.csv", index=False)
df.to_json("analysis/tables/prediction_simsample.json", orient="records", indent=2)
print(df.to_string(index=False))
print("\nPredictions written. Computers/CS are UNTESTED at the time of writing;")
print("this file is committed before those experiments run.")
