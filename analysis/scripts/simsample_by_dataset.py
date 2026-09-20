"""SimSample effect against degree and budget, every dataset that has both arms.

Feeds figures/regime_decomposition.png. Was an inline script; now part of the
refresh so a new dataset (tfinance) enters the figure the moment its arms land.

    python analysis/scripts/simsample_by_dataset.py  # -> analysis/tables/simsample_by_dataset.json
"""
import json, os, os, sys
import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from utils import load_data

d = pd.read_csv("results/results.csv").drop_duplicates(
    subset=["dataset", "method", "seed", "train_seed", "tag"], keep="last")
d = d[d.num_epochs >= 100]
# (on-arm, off-arm) per dataset; yelp's main arm is ON, the binary real graphs run both
ARMS = {"photo": ("C_sim", "A_main"), "computers": ("C_sim", "A_main"), "cs": ("C_sim", "A_main"),
        "ogbn-arxiv": ("C_sim", "A_main"), "yelp": ("A_main", "C_nosim"),
        "amazon": ("A_sim1.0", "A_sim0.0"), "tfinance": ("A_sim1.0", "A_sim0.0")}


def pair(ds, on, off, m):
    a = d[(d.dataset == ds) & (d.method == "outpost") & (d.tag == on)].set_index("seed")
    b = d[(d.dataset == ds) & (d.method == "outpost") & (d.tag == off)].set_index("seed")
    k = sorted(set(a.index) & set(b.index))
    if len(k) < 3:
        return None
    dl = (a.loc[k, m] - b.loc[k, m]).astype(float)
    return {"n": len(k), "delta": round(float(dl.mean()), 4), "W": int((dl > 0).sum()),
            "p": round(float(wilcoxon(dl).pvalue), 4)}


out = {}
for ds, (on, off) in ARMS.items():
    r = pair(ds, on, off, "best_auroc")
    if r is None:
        continue                                   # arms not run yet
    try:
        g, _, _ = load_data(ds)
    except (FileNotFoundError, OSError) as e:
        # a fresh clone has no data/ -- the datasets are downloaded or unpacked on
        # first use. The committed artifact stays as it is rather than being
        # rewritten from whatever subset happens to be on disk.
        print(f"  [skip] {ds}: raw graph not available ({type(e).__name__})")
        continue
    N = g.x.shape[0]
    deg = torch.bincount(g.edge_index[1], minlength=N).numpy()
    out[ds] = {"n_nodes": int(N), "median_degree": float(np.median(deg)), "mean_degree": float(deg.mean()),
               "frac_above_budget25": float((deg > 25).mean()),
               "sim_delta_roc": r, "sim_delta_pr": pair(ds, on, off, "best_aupr"), "arms": {"on": on, "off": off}}
    print(f"  {ds:11s} deg {out[ds]['median_degree']:6.1f}  >25 {100*out[ds]['frac_above_budget25']:5.1f}%  dROC {r['delta']:+.4f} (n={r['n']})")
DST = "analysis/tables/simsample_by_dataset.json"
# Never replace a complete artifact with a shorter one. Without this, running here
# with only some of data/ present would silently drop graphs from the regime figure,
# and the figure would still render -- just with fewer bars than the paper reports.
prev = {}
if os.path.exists(DST):
    try:
        prev = json.load(open(DST))
    except Exception:
        prev = {}
if prev and len(out) < len(prev):
    missing = sorted(set(prev) - set(out))
    print(f"  REFUSING to overwrite {DST}: it has {len(prev)} graphs, this run produced "
          f"{len(out)}. Missing: {', '.join(missing)}. Kept the existing file.")
else:
    json.dump(out, open(DST, "w"), indent=1)
    print(f"-> {DST}")
