#!/usr/bin/env python3
"""Move shards whose recorded budget is not the expected one out of the merge path.

Four DEMO-on-arxiv jobs were launched before their campaign set num_epochs=400 and
cannot be stopped; they will finish at 200 and write shards under B_demo_mix, the
MAIN TABLE tag. Nothing downstream filters that tag on budget, so a 200-epoch row
would silently sit beside 400-epoch ones.

Watches results/rotations/ and relocates any matching shard to results/wrong_budget/.

    python analysis/scripts/quarantine_wrong_budget.py <dataset> <method> <expected_epochs> [minutes]
"""
import glob, json, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
DEST = "results/wrong_budget"


def sweep(ds, method, want):
    moved = []
    os.makedirs(DEST, exist_ok=True)
    for f in glob.glob(f"results/rotations/{ds}_{method}_s*_*.json"):
        try:
            ep = json.load(open(f))["config"].get("num_epochs")
        except Exception:
            continue
        if ep is not None and int(ep) != want:
            shutil.move(f, os.path.join(DEST, os.path.basename(f)))
            moved.append((os.path.basename(f), ep))
    return moved


if __name__ == "__main__":
    ds, method, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
    minutes = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    deadline = time.time() + minutes * 60
    seen = 0
    while True:
        for name, ep in sweep(ds, method, want):
            seen += 1
            print(f"[{time.strftime('%H:%M')}] quarantined {name} (epochs={ep}, "
                  f"expected {want})", flush=True)
        if time.time() >= deadline:
            break
        time.sleep(60)
    print(f"done: {seen} shard(s) quarantined")
