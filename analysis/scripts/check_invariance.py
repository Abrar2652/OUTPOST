"""Checks that execution-level changes did not move any number.

Two things in this repository change HOW a run executes without being allowed to
change WHAT it computes:

  rotation sharding   `main.py --rotations` splits a dataset's rotations across
                      processes. Legitimate only if a rotation's split and
                      initialisation depend on (seed, rotation) alone.
  feature placement   `features_on_gpu` keeps the feature table on the device and
                      moves sampled ids to it instead of gathering on the host.
                      Legitimate only if it gathers the same rows in the same
                      order.

Both claims are cheap to test and expensive to get wrong, so they are tested
rather than asserted: each runs the same short configuration both ways and
requires the per-rotation metrics to match exactly.

    python analysis/scripts/check_invariance.py
    python analysis/scripts/check_invariance.py --dataset cs --epochs 6
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
KEYS = ("auroc_all", "aupr_all", "auroc_unknown", "aupr_unknown")


def run(dataset, epochs, tag, extra, method="outpost"):
    cmd = [sys.executable, "-u", "main.py", "--dataset", dataset,
           "--seed", "42", "--epochs", str(epochs), "--tag", tag,
           "--method", method] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(f"run failed: {' '.join(cmd)}")
    out = {}
    for p in os.listdir("results/rotations"):
        if f"_{tag}_rot" in p and p.startswith(f"{dataset}_{method}"):
            d = json.load(open(f"results/rotations/{p}"))
            for rot in d["rotations"]:
                out[int(rot["rotation_class"])] = {k: rot["best"][k] for k in KEYS}
    return out


def compare(name, a, b):
    ok = True
    if set(a) != set(b):
        print(f"  {name}: FAIL - different rotations {sorted(a)} vs {sorted(b)}")
        return False
    for c in sorted(a):
        for k in KEYS:
            if a[c][k] != b[c][k]:
                print(f"  {name}: FAIL - rotation {c} {k}: "
                      f"{a[c][k]!r} vs {b[c][k]!r}")
                ok = False
    print(f"  {name}: {'identical over ' + str(len(a)) + ' rotation(s)' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="photo")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--method", default="outpost", choices=["outpost", "demo"])
    a = ap.parse_args()

    print(f"invariance checks on {a.dataset}, {a.epochs} epochs, {a.method}")
    ok = True

    if a.method == "demo":
        # The DEMO port's weak view is computed under no_grad. It feeds only the
        # hard pseudo-label and the confidence mask, so its graph cannot affect
        # any gradient - but that is an argument, and the run is what settles it.
        grad = run(a.dataset, a.epochs, "_inv_dgrad",
                   ["--set", "energy_loss=false", "demo_weak_nograd=false"], "demo")
        nog = run(a.dataset, a.epochs, "_inv_dnograd",
                  ["--set", "energy_loss=false", "demo_weak_nograd=true"], "demo")
        ok &= compare("demo_weak_nograd", grad, nog)
        for p in os.listdir("results/rotations"):
            if "_inv_" in p:
                os.remove(f"results/rotations/{p}")
        print("\nPASS" if ok else
              "\nFAIL - the no-grad weak view changed a number")
        sys.exit(0 if ok else 1)

    # 1. feature placement
    host = run(a.dataset, a.epochs, "_inv_hostfeat", ["--set", "features_on_gpu=false"])
    dev = run(a.dataset, a.epochs, "_inv_devfeat", ["--set", "features_on_gpu=true"])
    ok &= compare("features_on_gpu", host, dev)

    # 1b. the same, with SimSample on, across MORE THAN ONE rotation.
    # This case is separate because it is the one that broke and the checks above
    # did not catch it: `features_to_device` used to mutate the caller's graph,
    # so rotation 2 inherited rotation 1's device-resident features and SimSample
    # built its similarity ordering on the wrong device. Single-rotation runs
    # (Yelp) and sim_topk_frac=0 runs (everything else) both miss it entirely.
    if a.dataset != "yelp":
        hs = run(a.dataset, a.epochs, "_inv_simhost",
                 ["--set", "sim_topk_frac=1.0", "features_on_gpu=false"])
        ds = run(a.dataset, a.epochs, "_inv_simdev",
                 ["--set", "sim_topk_frac=1.0", "features_on_gpu=true"])
        ok &= compare("features_on_gpu + SimSample, multi-rotation", hs, ds)

    # 2. rotation sharding: whole run vs one process per rotation
    full = run(a.dataset, a.epochs, "_inv_full", [])
    shards = {}
    for c in sorted(full):
        shards.update(run(a.dataset, a.epochs, "_inv_shard",
                          ["--rotations", str(c)]))
    ok &= compare("rotation sharding", full, shards)

    for p in os.listdir("results/rotations"):
        if "_inv_" in p:
            os.remove(f"results/rotations/{p}")
    print("\nPASS" if ok else "\nFAIL - an execution change moved a number")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
