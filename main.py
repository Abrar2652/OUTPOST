"""OUTPOST entry point."""

import argparse
import json
import logging
import os
import socket
import time

import numpy as np
from addict import Dict

from utils import aggregate_rotations, ad_split_num, load_data, set_seed, split_fingerprint


def load_config(dataset, method="outpost", overrides=None):
    cfg_all = json.load(open("config.json"))
    cfg = dict(cfg_all.get("default", {}))
    ds = dict(cfg_all.get(dataset, {}))
    cfg.update(ds)
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return cfg


def parse_set(pairs):
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
    per = np.asarray(dset_info["class_per"])
    if dataset in ("yelp", "tfinance", "amazon"):
        return np.array([1])
    if dataset in ("photo", "computers", "cs"):
        return np.where((per <= 0.05) & (per >= 0.0))[0]
    if dataset == "ogbn-arxiv":
        return np.where((per <= 0.05) & (per >= 0.03))[0]
    if dataset == "ogbn-mag":
        return np.where((per <= 0.0003) & (per >= 0.0))[0]
    raise ValueError(dataset)


def rotation_path(args_cli, rotation_class):
    tag = (args_cli.tag or "untagged").replace("/", "_").replace(" ", "_")
    ts = "" if args_cli.train_seed is None else f"_t{args_cli.train_seed}"
    return (f"results/rotations/{args_cli.dataset}_{args_cli.method}"
            f"_s{args_cli.seed}{ts}_{tag}_rot{rotation_class}.json")


def write_rotation(args_cli, cfg, res):
    os.makedirs("results/rotations", exist_ok=True)
    path = rotation_path(args_cli, res["rotation_class"])
    import platform, socket
    try:
        import torch as _t
        _gpu = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu"
        _tv = _t.__version__
    except Exception:
        _gpu, _tv = "?", "?"
    payload = {"dataset": args_cli.dataset, "method": args_cli.method,
               "seed": args_cli.seed, "train_seed": args_cli.train_seed,
               "tag": args_cli.tag or "", "config": cfg,
               "host": socket.gethostname(), "gpu": _gpu, "torch": _tv,
               "python": platform.python_version(),
               "rotations": [{k: v for k, v in res.items() if k != "scores"}]}
    tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def save_scores(args_cli, res, split, labels_np, info, idx):
    from sklearn.metrics import roc_auc_score
    S = res["scores"]
    y = np.isin(labels_np, list(info["all_anomaly"])).astype(int)
    iv = np.asarray(split["idx_val"]); it = np.asarray(split["idx_test"]["all"])
    def auc(idx, s):
        yy = y[idx]
        return 0.5 if yy.min() == yy.max() else roc_auc_score(yy, s[idx])
    va = np.array([auc(iv, S[t]) for t in range(S.shape[0])])
    ta = np.array([auc(it, S[t]) for t in range(S.shape[0])])
    e_val, e_best = int(va.argmax()), int(ta.argmax())
    os.makedirs("results/scores", exist_ok=True)
    stem = os.path.basename(rotation_path(args_cli, idx))[:-5]
    np.save(f"results/scores/{stem}_best.npy", S[e_best].astype(np.float16))
    np.save(f"results/scores/{stem}_valsel.npy", S[e_val].astype(np.float16))
    json.dump({"best_epoch": e_best, "valsel_epoch": e_val, "n_nodes": int(S.shape[1]),
               "n_epochs_logged": int(S.shape[0]), "split_sha": res.get("split_sha")},
              open(f"results/scores/{stem}.json", "w"), indent=1)
    res["scores_saved"] = {"best_epoch": e_best, "valsel_epoch": e_val}
    del res["scores"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp",
                    choices=["photo", "computers", "cs",
                             "yelp", "amazon", "tfinance", "ogbn-arxiv", "ogbn-mag"])
    ap.add_argument("--method", default="outpost", choices=["outpost"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_seed", type=int, default=None,
                    help="vary training randomness while holding the split fixed")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override any config key, e.g. --set sim_topk_frac=1.0")
    ap.add_argument("--tag", default=None, help="label for the results row")
    ap.add_argument("--save-scores", action="store_true",
                    help="keep per-node scores for evaluation")
    ap.add_argument("--rotations", type=int, nargs="*", default=None,
                    metavar="CLASS",
                    help="run only specified seen-anomaly classes")
    args_cli = ap.parse_args()

    ov = parse_set(args_cli.set)
    if args_cli.epochs is not None:
        ov["num_epochs"] = args_cli.epochs
    if args_cli.save_scores:
        ov["record_scores"] = True

    cfg = load_config(args_cli.dataset, args_cli.method, ov)
    args = Dict(cfg)
    args.dataname = args_cli.dataset
    args.num_classes = 2
    args.train_normal_ratio = cfg.get("train_normal_ratio", 0.05)
    args.train_anormaly_num = cfg.get("train_anormaly_num", 50)
    args.val_normal_ratio = cfg.get("val_normal_ratio", 0.01)
    args.val_anormaly_num = cfg.get("val_anormaly_num", 30)

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
    acls_full = anomaly_classes(args_cli.dataset, dset_info)
    print(f"[{args_cli.dataset}] anomaly classes (rotations): {list(acls_full)}")
    acls = acls_full
    if args_cli.rotations:
        unknown = set(args_cli.rotations) - set(int(c) for c in acls_full)
        if unknown:
            raise SystemExit(f"--rotations {sorted(unknown)} are not anomaly "
                             f"classes of {args_cli.dataset}: {list(acls_full)}")
        acls = np.array([c for c in acls_full if int(c) in args_cli.rotations])
        print(f"[{args_cli.dataset}] running subset: {list(acls)}")

    from trainer import train_outpost_v4

    rotations = []
    for ri, idx in enumerate(acls):
        info = {"known_anomaly": idx,
                "unknown_anomaly": [i for i in acls_full if i != idx],
                "normal": [i for i in dset_info["class_idx"] if i not in acls_full],
                "all_anomaly": acls_full}
        set_seed(args_cli.seed)
        split = ad_split_num(labels_np, args, info)
        if args_cli.train_seed is not None:
            set_seed(args_cli.train_seed)
        t_rot = time.time()
        res = train_outpost_v4(split, labels_np, graph, args, info, logger)
        res["rotation_class"] = int(idx)
        res["split_sha"] = split_fingerprint(split)
        if args_cli.save_scores and "scores" in res:
            save_scores(args_cli, res, split, labels_np, info, int(idx))
        res["seconds"] = round(time.time() - t_rot, 1)
        try:
            import torch
            res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        except Exception:
            pass
        rotations.append(res)
        b = res["best"]
        print(f"  rotation {idx}: ROC {b['auroc_all']:.4f} PR {b['aupr_all']:.4f}")
        write_rotation(args_cli, cfg, res)

    if not rotations:
        return

    print(f"[{args_cli.dataset}] params {rotations[0].get('n_params')}, "
          f"peak GPU {max(r.get('peak_gpu_gb', 0) for r in rotations):.2f} GB, "
          f"{sum(r['seconds'] for r in rotations)/60:.1f} min total")

    if len(acls) < len(acls_full):
        print(f"[{args_cli.dataset}] ran {len(acls)}/{len(acls_full)} rotations; "
              f"no results.csv row written. Merge the shards with:\n"
              f"    python analysis/scripts/merge_rotations.py")
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
