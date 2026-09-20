"""Pick a configuration using the validation split, and only the validation split.

The rule is pre-registered in `analysis/tables/prediction_tuning.md`: maximise
mean `val_auc` - AUC-ROC on the validation nodes at the epoch where validation
AUC peaks - averaged over rotations and selection seeds. Ties within 0.002 go to
the configuration with fewer parameters, then to the repository default.

The safeguard that matters is structural rather than clerical. `read_val()`
loads `val_selected.val_auc` and the parameter count from each per-rotation
record and **discards every test metric before returning**, so no test number is
in scope while a winner is being chosen. Test metrics are read afterwards, by a
separate pass that runs only once a winner is fixed and only when `--reveal` is
passed. `valsel_auroc` is not usable here either: it is a test metric read at the
peak-validation epoch, which is a reporting column, not a selection signal.

    python analysis/scripts/select_hparams.py --dataset photo
    python analysis/scripts/select_hparams.py --dataset photo --reveal
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
TIE = 0.002


def read_val(dataset, tags):
    """{tag: {seed: (mean val_auc over rotations, n_params)}} - validation only.

    Everything else in the record, including every test metric, is dropped here
    and never returned.
    """
    out = defaultdict(dict)
    for p in sorted(glob.glob("results/rotations/*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r["dataset"] != dataset or r["method"] != "outpost":
            continue
        if tags is not None and r.get("tag") not in tags:
            continue
        vals, params = [], []
        for rot in r["rotations"]:
            v = rot.get("val_selected", {}).get("val_auc")
            if v is None:
                continue
            vals.append(float(v))
            params.append(rot.get("n_params"))
        if not vals:
            continue
        prev = out[r["tag"]].get(r["seed"], ([], None))
        out[r["tag"]][r["seed"]] = (prev[0] + vals, params[0])
    return {t: {s: (float(np.mean(v)), n) for s, (v, n) in d.items()}
            for t, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="photo")
    ap.add_argument("--default-tag", default="A_main")
    ap.add_argument("--expect-arms", type=int, default=0, help="refuse to pick a winner until this many sweep arms are complete")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="require exactly these selection seeds on every sweep arm before picking")
    ap.add_argument("--prefix", default="T_", help="sweep-arm tag prefix (E400_T_ for the 400-epoch sweep)")
    ap.add_argument("--out-suffix", default="", help="e.g. _400 -> hparam_selection_<ds>_400.json")
    ap.add_argument("--reveal", action="store_true",
                    help="after a winner is fixed, also print its test metrics")
    a = ap.parse_args()

    val = read_val(a.dataset, None)
    arms = {t: d for t, d in val.items()
            if t.startswith(a.prefix) or t == a.default_tag}
    if a.default_tag not in arms:
        raise SystemExit(f"no {a.default_tag} runs for {a.dataset} to compare against")

    # selection seeds: those the sweep arms were run at
    sweep = [t for t in arms if t.startswith(a.prefix)]
    if not sweep:
        raise SystemExit("no T_* sweep arms found yet")
    sel_seeds = sorted(set.intersection(*[set(arms[t]) for t in sweep]))
    if not sel_seeds:
        raise SystemExit("sweep arms share no common seed yet")
    if a.seeds is not None and set(sel_seeds) != set(a.seeds):
        raise SystemExit(f"selection seeds {sel_seeds} != required {sorted(a.seeds)}; some arm is incomplete, no winner written")

    rows = []
    for t in [a.default_tag] + sorted(sweep):
        seeds = [s for s in sel_seeds if s in arms[t]]
        if len(seeds) < len(sel_seeds):
            print(f"  [pending] {t}: {len(seeds)}/{len(sel_seeds)} seeds")
            continue
        v = [arms[t][s][0] for s in seeds]
        rows.append({"tag": t, "n": len(v), "val_auc": float(np.mean(v)),
                     "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                     "params": arms[t][seeds[0]][1]})
    if not rows:
        raise SystemExit("no complete arms yet")
    n_sweep_done = sum(1 for r in rows if r["tag"] != a.default_tag)
    if a.expect_arms and n_sweep_done < a.expect_arms:
        raise SystemExit(f"sweep incomplete: {n_sweep_done}/{a.expect_arms} arms have all selection seeds; no winner written")

    rows.sort(key=lambda r: -r["val_auc"])
    print("=" * 66)
    print(f"VALIDATION-ONLY SELECTION   {a.dataset}   seeds {sel_seeds}")
    print("no test metric has been read at this point")
    print("=" * 66)
    print(f"{'arm':18s} {'n':>2s} {'mean val_auc':>13s} {'sd':>8s} {'params':>9s}")
    for r in rows:
        mark = "  <- default" if r["tag"] == a.default_tag else ""
        print(f"{r['tag']:18s} {r['n']:>2d} {r['val_auc']:>13.4f} "
              f"{r['sd']:>8.4f} {r['params']:>9d}{mark}")

    best = rows[0]["val_auc"]
    tied = [r for r in rows if best - r["val_auc"] <= TIE]
    if len(tied) > 1:
        # pre-registered tie-break: fewer parameters, then the default
        if any(r["tag"] == a.default_tag for r in tied):
            winner = next(r for r in tied if r["tag"] == a.default_tag)
            why = f"tie within {TIE}; default retained"
        else:
            winner = min(tied, key=lambda r: r["params"])
            why = f"tie within {TIE}; fewest parameters"
    else:
        winner = rows[0]
        why = "highest mean val_auc"
    print(f"\nWINNER  {winner['tag']}   ({why})")
    print(f"  {len(tied)} arm(s) within the {TIE} tie band: "
          f"{[r['tag'] for r in tied]}")

    json.dump({"dataset": a.dataset, "selection_seeds": sel_seeds,
               "criterion": "mean val_auc, validation split only",
               "tie_band": TIE, "winner": winner, "arms": rows},
              open(f"analysis/tables/hparam_selection_{a.dataset}{a.out_suffix}.json", "w"),
              indent=1)
    print(f"-> analysis/tables/hparam_selection_{a.dataset}{a.out_suffix}.json")

    if not a.reveal:
        print("\nTest metrics deliberately not shown. Re-run with --reveal once "
              "the\nwinner above is final; that is the only read of the test "
              "split in this\nprocedure and it happens after selection, not "
              "during it.")
        return

    print("\n" + "=" * 66)
    print("TEST METRICS (read once, after selection)")
    print("=" * 66)
    import pandas as pd
    d = pd.read_csv("results/results.csv").drop_duplicates(
        subset=["dataset", "method", "seed", "train_seed", "tag"], keep="last")
    for t in dict.fromkeys([a.default_tag, winner["tag"]]):
        g = d[(d.dataset == a.dataset) & (d.tag == t) & (d.method == "outpost")]
        if g.empty:
            continue
        print(f"  {t:18s} n={len(g)}  ROC {g.best_auroc.mean():.4f}  "
              f"PR {g.best_aupr.mean():.4f}  "
              f"valsel ROC {g.valsel_auroc.mean():.4f}")


if __name__ == "__main__":
    main()
