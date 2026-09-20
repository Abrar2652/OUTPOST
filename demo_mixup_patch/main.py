"""OUTPOST entry point.

Runs OUTPOST (default) or the DEMO baseline on one dataset, over every
anomaly-class rotation, and appends the aggregated row to results/results.csv.

Examples
--------
    python main.py --dataset yelp                    # OUTPOST, seed 42
    python main.py --dataset ogbn-arxiv              # large-scale (Table 2)
    python main.py --dataset photo --seed 0
    python main.py --dataset yelp --method demo      # DEMO baseline
    python main.py --dataset computers --set sim_topk_frac=1.0   # ablation
    python main.py --dataset photo --epochs 3        # quick smoke test

Config comes from config.json ('default' merged with the per-dataset block);
--set overrides any key on the command line, which is how the ablations in
analysis/ were produced.
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from addict import Dict

from utils import (aggregate_rotations, ad_split_num, compute_ppr, load_data,
                   set_seed)


def get_ppr(dataset, graph):
    """PPR matrix for DEMO's multi-sample mixup, cached on disk after the first
    call (the authors' main.py does the same, at data/<name>/ppr_matrix.npy).

    Only needed when mixup is on. It is a dense N x N matrix, so it is feasible
    on the small graphs and not on the large ones:
        photo 0.5 GB | computers 1.5 GB | cs 2.7 GB | yelp 16.9 GB
        ogbn-arxiv 229 GB | ogbn-mag 4.3 TB
    compute_ppr() also needs dgl for anything other than photo/computers.
    """
    path = os.path.join("data", dataset.replace("-", "_"), "ppr_matrix.npy")
    if os.path.exists(path):
        print(f"[ppr] loading cached {path}")
        return np.load(path)
    print(f"[ppr] computing PPR for {dataset} (dense N x N, one-off) ...")
    ppr = compute_ppr(graph, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, ppr)
    print(f"[ppr] cached to {path}  shape={ppr.shape}")
    return ppr


def load_config(dataset, method="outpost", overrides=None):
    """default -> (demo_default if method==demo) -> per-dataset -> CLI overrides."""
    cfg_all = json.load(open("config.json"))
    cfg = dict(cfg_all.get("default", {}))
    if method == "demo":
        cfg.update(cfg_all.get("demo_default", {}))
    ds = dict(cfg_all.get(dataset, {}))
    if method == "demo":
        # The DEMO baseline keeps its own hyperparameters, so the dataset block's
        # OUTPOST settings must not leak in. But two of its keys are properties of
        # the experiment rather than of the method and MUST carry over, or the
        # comparison is not like-for-like:
        #   input_dim   - a property of the data
        #   num_epochs  - the protocol's budget (200 small / 400 large), which
        #                 applies to every method equally. Dropping it silently
        #                 gave DEMO 200 epochs against OUTPOST's 400 on the three
        #                 large graphs, and under best-over-epochs selection that
        #                 is a direct advantage to OUTPOST.
        ds = {k: v for k, v in ds.items() if k in ("input_dim", "num_epochs")}
    cfg.update(ds)
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return cfg


def parse_set(pairs):
    """--set key=value ... with json-ish coercion"""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def anomaly_classes(dataset, dset_info):
    """Open-set protocol: minority classes are 'anomalies'.

    Binary fraud graphs (yelp) have a single anomaly class and therefore no
    unseen-class rotation.
    """
    per = np.asarray(dset_info["class_per"])
    if dataset in ("yelp", "tfinance"):
        return np.array([1])
    if dataset in ("photo", "computers", "cs"):
        return np.where((per <= 0.05) & (per >= 0.0))[0]
    if dataset == "ogbn-arxiv":
        return np.where((per <= 0.05) & (per >= 0.03))[0]
    if dataset == "ogbn-mag":
        return np.where((per <= 0.0003) & (per >= 0.0))[0]
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp",
                    choices=["photo", "computers", "cs",
                             "yelp", "ogbn-arxiv", "ogbn-mag"])
    ap.add_argument("--method", default="outpost", choices=["outpost", "demo"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_seed", type=int, default=None,
                    help="vary training randomness while holding the split fixed")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override any config key, e.g. --set sim_topk_frac=1.0")
    ap.add_argument("--tag", default=None, help="label for the results row")
    args_cli = ap.parse_args()

    ov = parse_set(args_cli.set)
    if args_cli.epochs is not None:
        ov["num_epochs"] = args_cli.epochs

    cfg = load_config(args_cli.dataset, args_cli.method, ov)
    args = Dict(cfg)
    # config.json carries DEMO's original `device: 0`, which its mixup path
    # passes straight to .to(). Resolve it the same way trainer.py does so the
    # baseline runs on a CPU-only machine too.
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataname = args_cli.dataset
    args.num_classes = 2
    # the open-set label budget (fixed by the protocol, same for all baselines)
    args.train_normal_ratio = 0.05
    args.train_anormaly_num = 50
    args.val_normal_ratio = 0.01
    args.val_anormaly_num = 30

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, filename=f"logs/{args_cli.dataset}_{stamp}.log",
        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"cli={vars(args_cli)}")
    logger.info(f"config={cfg}")

    set_seed(args_cli.seed)
    graph, labels, dset_info = load_data(args_cli.dataset)
    labels_np = np.asarray(labels)
    acls = anomaly_classes(args_cli.dataset, dset_info)
    print(f"[{args_cli.dataset}] anomaly classes (rotations): {list(acls)}")

    from trainer import train, train_outpost_v4

    rotations = []
    for ri, idx in enumerate(acls):
        info = {"known_anomaly": idx,
                "unknown_anomaly": [i for i in acls if i != idx],
                "normal": [i for i in dset_info["class_idx"] if i not in acls],
                "all_anomaly": acls}
        split = ad_split_num(labels_np, args, info)
        if args_cli.train_seed is not None:
            set_seed(args_cli.train_seed)
        if args_cli.method == "demo":
            # State the variant in every run's output. Running with mixup off and
            # labelling the result "DEMO" is the paper's 'w/o Mix' ablation, not
            # the method, so it must never happen by accident.
            if ri == 0:
                print("=" * 66)
                print("DEMO variant: " + ("FULL published method (mixup ON)"
                                          if args.mixup else
                                          "'w/o Mix' ABLATION (mixup OFF) "
                                          "- not the published method"))
                print("=" * 66, flush=True)
            # DEMO's mixup needs the PPR matrix; without it anomaly_mixup()
            # raises TypeError on a None index. Computed once, then cached.
            ppr = get_ppr(args_cli.dataset, graph) if args.mixup else None
            res = train(split, labels_np, graph, args, info, logger, ppr)
        else:
            res = train_outpost_v4(split, labels_np, graph, args, info, logger)
        rotations.append(res)
        b = res["best"]
        print(f"  rotation {idx}: ROC {b['auroc_all']:.4f} PR {b['aupr_all']:.4f}")

    if not rotations:
        return
    agg = aggregate_rotations(rotations)
    aggv = aggregate_rotations(rotations, which="val_selected")
    print(f"\n[{args_cli.dataset}] AGGREGATE  best-over-epochs: "
          f"ROC {agg['auroc_all']:.4f} PR {agg['aupr_all']:.4f}")
    print(f"[{args_cli.dataset}] AGGREGATE  val-selected:     "
          f"ROC {aggv['auroc_all']:.4f} PR {aggv['aupr_all']:.4f}")
    logger.info(f"AGG best={agg} val_selected={aggv}")

    from results_writer import append_row
    append_row(args_cli.dataset, args_cli.method, args_cli.seed,
               args_cli.train_seed, agg, aggv, cfg,
               tag=args_cli.tag or "")
    print("row appended to results/results.csv")


if __name__ == "__main__":
    main()
