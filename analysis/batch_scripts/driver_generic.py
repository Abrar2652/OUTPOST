"""Generic multi-rotation OUTPOST driver (photo / computers / cs).

Same protocol as driver.py but dataset-parameterised: iterates the dataset's
anomaly-class rotations, aggregates over them exactly as main.py does, and
never writes results.csv.

Usage: python driver_generic.py <dataset> '<json overrides>' <tag>
"""
import sys, os, json, logging, contextlib, time

def _find_repo():
    """Repo root = nearest ancestor containing trainer.py.

    Works whether this file sits in analysis/yelp_ablations/ (laptop) or is
    copied to the repo root (Colab). OUTPOST_REPO overrides.
    """
    env = os.environ.get("OUTPOST_REPO")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "trainer.py")):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "cannot locate OUTPOST repo root (no trainer.py in any parent); "
                "set OUTPOST_REPO")
        d = parent
    return d


REPO = _find_repo()
sys.path.insert(0, REPO)
os.chdir(REPO)

import yaml
import numpy as np
from addict import Dict
from utils import set_seed, load_data, ad_split_num, aggregate_rotations
from trainer import train_outpost_v4

dataset = sys.argv[1]
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
tag = sys.argv[3] if len(sys.argv) > 3 else f"{dataset}-run"
here = os.path.dirname(os.path.abspath(__file__))
runlog = os.path.join(here, f"{tag}.runlog")

seed = int(overrides.pop("seed", 42))
train_seed = overrides.pop("train_seed", None)
set_seed(seed)

cfg_path = f"data/{dataset.replace('-', '_')}/outpost.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg.update(overrides)
args = Dict(cfg)
args.dataname = dataset
args.num_classes = 2
args.train_normal_ratio = 0.05
args.train_anormaly_num = 50
args.val_normal_ratio = 0.01
args.val_anormaly_num = 30

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("driver_generic")

graph, labels, dset_info = load_data(dataset)
labels_np = np.asarray(labels)
anomaly_class_idx = np.where((dset_info["class_per"] <= 0.05) &
                             (dset_info["class_per"] >= 0.0))[0]

t0 = time.time()
rots = []
with open(runlog, "w", encoding="utf-8") as lf:
    for ri, idx in enumerate(anomaly_class_idx):
        anomaly_info = {"known_anomaly": idx,
                        "unknown_anomaly": [i for i in anomaly_class_idx if i != idx],
                        "normal": [i for i in dset_info["class_idx"]
                                   if i not in anomaly_class_idx],
                        "all_anomaly": anomaly_class_idx}
        split_info = ad_split_num(labels_np, args, anomaly_info)
        if train_seed is not None:
            set_seed(int(train_seed))
        with contextlib.redirect_stdout(lf):
            res = train_outpost_v4(split_info, labels_np, graph, args,
                                   anomaly_info, logger)
        rots.append(res)
        b = res["best"]
        print(f"  rot {idx}: ROC {b['auroc_all']:.4f} PR {b['aupr_all']:.4f}",
              flush=True)

agg = aggregate_rotations(rots)
aggv = aggregate_rotations(rots, which="val_selected")
print(f"[{tag}] AGG best: ROC {agg['auroc_all']:.4f} PR {agg['aupr_all']:.4f} | "
      f"unseen ROC {agg['auroc_unknown']:.4f} PR {agg['aupr_unknown']:.4f} | "
      f"{time.time()-t0:.0f}s", flush=True)
print(f"[{tag}] AGG val-selected: ROC {aggv['auroc_all']:.4f} "
      f"PR {aggv['aupr_all']:.4f}", flush=True)
