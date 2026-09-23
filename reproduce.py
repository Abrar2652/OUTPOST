"""ONE COMMAND to reproduce the paper end to end.

    python reproduce.py

Runs OUTPOST on all six datasets, then builds both paper tables and every
figure. Nothing else needs to be typed - the environment is checked, the data is
unpacked, and results are written as each dataset finishes.

    python reproduce.py --quick          # 3 epochs/dataset, ~10 min: proves the
                                         # pipeline works before committing hours
    python reproduce.py --seeds 42 0 1   # 3 seeds each (paper-grade, slower)
    python reproduce.py --small-only     # Photo/Computers/CS only
    python reproduce.py --resume         # skip datasets already finished

Every dataset is run FRESH by default so the numbers are yours, not ours; our
measurements sit in results/reference_runs.csv and are printed side by side for
comparison. Interrupting is safe - finished datasets are already written to
results/results.csv, and --resume continues from there.

Wall-clock, one seed, on a >=16 GB GPU:
    photo ~20 min | computers ~30 min | cs 2-4 h
    yelp ~30 min  | ogbn-arxiv 2-4 h  | ogbn-mag 6-12 h
CS and ogbn-mag OOM on cards below ~16 GB; use --small-only or skip them there.
"""

import argparse
import os
import subprocess
import sys
import time
import zipfile

SMALL = ["photo", "computers", "cs"]
LARGE = ["yelp", "ogbn-arxiv", "ogbn-mag"]
NEEDS_BIG_GPU = {"cs", "ogbn-mag"}
EST = {"photo": "~20 min", "computers": "~30 min", "cs": "2-4 h",
       "yelp": "~30 min", "ogbn-arxiv": "2-4 h", "ogbn-mag": "6-12 h"}

BAR = "=" * 72


def step(n, total, msg):
    print(f"\n{BAR}\n[{n}/{total}] {msg}\n{BAR}", flush=True)


def check_env():
    """Fail early and clearly rather than deep inside a training loop."""
    problems, notes = [], []
    try:
        import torch
        notes.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            notes.append(f"GPU {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")
            if gb < 15:
                notes.append("WARNING: <16 GB - cs and ogbn-mag will likely OOM")
        else:
            notes.append("no CUDA - running on CPU (much slower)")
    except Exception as e:
        problems.append(f"torch missing/broken: {e}")
    for mod, pip in (("torch_geometric", "torch_geometric"), ("addict", "addict"),
                     ("sklearn", "scikit-learn"), ("pandas", "pandas"),
                     ("matplotlib", "matplotlib"), ("yaml", "pyyaml")):
        try:
            __import__(mod)
        except Exception:
            problems.append(f"missing {mod}  ->  pip install {pip}")
    try:
        __import__("ogb")
    except Exception:
        notes.append("ogb not installed - ogbn-* will be skipped "
                     "(pip install ogb to include them)")
    return problems, notes


def ensure_data(datasets):
    """Unpack dataset.zip if needed. ogbn-* are fetched by ogb on first use."""
    need = [d for d in datasets if d in ("photo", "computers", "cs", "yelp")]
    missing = []
    for d in need:
        f = f"data/{d}/{d}.npz" if d != "yelp" else "data/yelp/yelp.zip"
        if not os.path.exists(f):
            missing.append(d)
    if not missing:
        print("  data present")
        return True
    if not os.path.exists("dataset.zip"):
        print(f"  ERROR: data missing for {missing} and dataset.zip not found")
        return False
    print(f"  unpacking dataset.zip (needed for {missing}) ...")
    with zipfile.ZipFile("dataset.zip") as z:
        z.extractall(".")
    print("  done")
    return True


def run_one(dataset, seed, method, epochs, log):
    cmd = [sys.executable, "main.py", "--dataset", dataset, "--seed", str(seed),
           "--method", method, "--tag", "reproduce"]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    ok = r.returncode == 0
    log.write(f"{time.strftime('%F %T')} {dataset} seed={seed} method={method} "
              f"rc={r.returncode} {dt:.0f}s\n")
    log.flush()
    print(f"  -> {'OK' if ok else 'FAILED (rc=%d)' % r.returncode} in {dt/60:.1f} min",
          flush=True)
    return ok


def finished(dataset, method, min_epochs=100):
    import pandas as pd
    p = "results/results.csv"
    if not os.path.exists(p):
        return False
    d = pd.read_csv(p)
    d = d[(d.dataset == dataset) & (d.method == method) & (d.num_epochs >= min_epochs)]
    return not d.empty


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the OUTPOST paper: all datasets, tables, figures.")
    ap.add_argument("--quick", action="store_true",
                    help="3 epochs per dataset - pipeline check, NOT paper numbers")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42])
    ap.add_argument("--small-only", action="store_true", help="Table 1 datasets only")
    ap.add_argument("--large-only", action="store_true", help="Table 2 datasets only")
    ap.add_argument("--datasets", nargs="*", default=None, help="explicit list")
    ap.add_argument("--resume", action="store_true",
                    help="skip datasets already completed in results/results.csv")
    a = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)
    epochs = 3 if a.quick else None

    datasets = a.datasets or (SMALL if a.small_only else
                              LARGE if a.large_only else SMALL + LARGE)

    total = 4
    step(1, total, "Checking the environment")
    problems, notes = check_env()
    for n in notes:
        print(f"  {n}")
    if problems:
        print("\n  CANNOT CONTINUE:")
        for p in problems:
            print(f"    - {p}")
        print("\n  Fix with:  pip install -r requirements.txt")
        sys.exit(1)
    try:
        __import__("ogb")
    except Exception:
        before = len(datasets)
        datasets = [d for d in datasets if not d.startswith("ogbn")]
        if len(datasets) < before:
            print("  -> skipping ogbn-* (ogb not installed)")

    step(2, total, "Preparing data")
    if not ensure_data(datasets):
        sys.exit(1)

    methods = ["outpost"]
    jobs = [(d, s, m) for m in methods for d in datasets for s in a.seeds]
    if a.resume:
        jobs = [(d, s, m) for (d, s, m) in jobs if not finished(d, m)]

    step(3, total, f"Training: {len(jobs)} run(s)"
                   + ("  [QUICK MODE - 3 epochs, not paper numbers]" if a.quick else ""))
    if not a.quick:
        print("  estimated wall-clock per seed:")
        for d in datasets:
            flag = "   (needs >=16 GB GPU)" if d in NEEDS_BIG_GPU else ""
            print(f"    {d:<12} {EST[d]}{flag}")
        print()
    failed = []
    with open("results/reproduce.log", "a", encoding="utf-8") as log:
        log.write(f"\n=== reproduce.py {time.strftime('%F %T')} quick={a.quick} "
                  f"seeds={a.seeds} datasets={datasets} ===\n")
        for i, (d, s, m) in enumerate(jobs, 1):
            print(f"\n--- [{i}/{len(jobs)}] {m} on {d}, seed {s} "
                  f"({EST.get(d,'?')}) ---", flush=True)
            if not run_one(d, s, m, epochs, log):
                failed.append((d, s, m))

    print(f"\n{BAR}\nDONE\n{BAR}")
    print("  raw      results/results.csv   (one row per run)")
    if a.quick:
        print("\n  --quick used 3 epochs. Now do the real run:\n")
        print("      python reproduce.py --seeds 42 0 1")
    if failed:
        print("\n  FAILED runs (re-run individually to see the error):")
        for d, s, m in failed:
            print(f"    python main.py --dataset {d} --seed {s} --method {m}")


if __name__ == "__main__":
    main()
