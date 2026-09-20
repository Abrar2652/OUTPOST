#!/usr/bin/env python3
"""Score P36 and P37 from the DEMO tuning grid.

P36 (blind): under the identical validation-only rule, DEMO retains its published
configuration on all four datasets, and its test-selection bonus is below +0.02 on
each. Falsified if a non-default arm wins on Photo, Computers or Yelp.

P37 (blind): applying one rule to all three methods produces deltas that differ in
SIGN across methods on Amazon. Falsified if all three move the same way.

Selection is the rule every other method got: mean val_auc over seeds 0, 1 and 42,
validation split only, 0.002 tie band, ties to the published configuration. Test
metrics are read only after the winner is fixed, and every arm - the default
included - is scored on those same three seeds.

Scored at the REPORTED budget: Photo and Computers from the E400_DT_* arms at 400
epochs, Yelp and Amazon from DT_* (natively 400). The 200-epoch DT_* Photo and
Computers arms are a second budget, reported separately.

    python analysis/scripts/score_demo_tune.py
"""
import glob, json, statistics as st, sys
from collections import defaultdict

SEL = (0, 1, 42)
BAND = 0.002
# dataset -> (default tag, prefix of the swept arms at the reported budget)
# Every entry must pair a default and its challengers AT THE SAME BUDGET. The
# first version paired yelp/amazon's 400-epoch B_demo_mix against DT_* arms that
# ran at DEMO's default 200 - because DEMO never inherits a dataset's num_epochs
# (load_config takes only input_dim and eval_batch_mult from the dataset block).
# A default with twice the budget wins trivially, so that comparison measured the
# budget gap, not the selection rule.
SPEC = {"photo": ("E400_demo", "E400_DT_"), "computers": ("E400_demo", "E400_DT_"),
        "yelp": ("B_demo_mix", "E400_DT_"), "amazon": ("B_demo_mix", "E400_DT_")}


# Each seed must carry its COMPLETE rotation set. Counting seeds alone would have
# scored P36 while one Photo arm was missing rot0 of seed 42 - a selection seed -
# averaging that seed over one rotation instead of two and calling the arm ready.
ROTATIONS = {"photo": [0, 7], "computers": [0, 3, 5, 6, 9],
             "yelp": [1], "amazon": [1]}


def diverged(ds, tag):
    """True if a run of this arm died with a NaN rather than never being launched."""
    for lg in glob.glob(f"runs/*/{tag}__{ds}_demo_s*_rot*.log"):
        try:
            if "Input contains NaN" in open(lg, errors="replace").read()[-4000:]:
                return True
        except OSError:
            pass
    return False


def budgets(ds, prefix, default):
    """{tag: set of recorded num_epochs} - the check the tag names cannot give."""
    seen = defaultdict(set)
    for f in glob.glob(f"results/rotations/{ds}_demo_s*_*.json"):
        j = json.load(open(f))
        t = j["tag"]
        if t.startswith(prefix) or t == default:
            seen[t].add(j["config"].get("num_epochs"))
    return seen


def arms(ds, prefix, default):
    out = defaultdict(dict)
    for f in glob.glob(f"results/rotations/{ds}_demo_s*_*.json"):
        j = json.load(open(f))
        tag = j["tag"]
        if not (tag.startswith(prefix) or tag == default):
            continue
        if prefix == "DT_" and tag.startswith("E400_DT_"):
            continue                     # keep the budgets apart
        if j["seed"] not in SEL:
            continue
        for r in j["rotations"]:
            out[tag].setdefault(j["seed"], []).append(
                (r["val_selected"]["val_auc"], r["best"]["auroc_all"]))
    need = set(ROTATIONS[ds])
    rows = []
    for tag, seeds in out.items():
        if len(seeds) < len(SEL) or any(len(rs) < len(need) for rs in seeds.values()):
            # Distinguish "still running" from "this configuration does not train".
            # A run that diverged to NaN will do so again - set_seed() turns on
            # deterministic algorithms - so waiting for it is futile, and an arm
            # that cannot produce a validation score cannot be selected. That is a
            # result about the configuration, not a gap in the data.
            if diverged(ds, tag):
                rows.append({"tag": tag, "n": 0, "val": None, "test": None,
                             "diverged": True})
            continue
        v = [st.mean([x[0] for x in rs]) for rs in seeds.values()]
        t = [st.mean([x[1] for x in rs]) for rs in seeds.values()]
        rows.append({"tag": tag, "n": len(v), "val": st.mean(v), "test": st.mean(t)})
    # diverged arms carry no val score; keep them at the end rather than crashing
    return sorted(rows, key=lambda r: (r["val"] is None, -(r["val"] or 0)))


def main():
    print(f"{'dataset':11s} {'arms':>4s} {'winner':26s} {'default kept':>12s} "
          f"{'in band':>8s} {'bonus':>8s}")
    # 3 swept arms + the default. Scoring a grid that is short an arm is how a
    # selection gets declared on a subset - the defect select_hparams.py grew its
    # --expect-arms guard for, and the one this scorer hit on its first run when a
    # Photo arm was missing one rotation of a selection seed.
    EXPECT = 4
    results, incomplete, mismatched = {}, [], []
    for ds, (default, prefix) in SPEC.items():
        b = budgets(ds, prefix, default)
        eps = {e for v in b.values() for e in v if e is not None}
        if len(eps) > 1:
            mismatched.append(f"{ds}: arms span budgets {sorted(eps)} "
                              f"({ {k: sorted(v) for k, v in b.items()} })")
    if mismatched:
        print("BUDGET MISMATCH - refusing to score:")
        for m in mismatched:
            print(f"  {m}")
        return 1
    for ds, (default, prefix) in SPEC.items():
        rows_all = arms(ds, prefix, default)
        div = [r for r in rows_all if r.get("diverged")]
        rows = [r for r in rows_all if not r.get("diverged")]
        for r in div:
            print(f"{ds:11s} {'':4s} {r['tag']:26s} DIVERGED to NaN - cannot be "
                  f"selected, excluded")
        if len(rows_all) != EXPECT or not any(r["tag"] == default for r in rows):
            have = {r["tag"] for r in rows}
            incomplete.append(f"{ds}: {len(rows)}/{EXPECT} complete arms "
                              f"(have {sorted(have)})")
            continue
        top = rows[0]["val"]
        band = [r for r in rows if top - r["val"] <= BAND]
        win = next((r for r in band if r["tag"] == default), band[0])
        best_test = max(rows, key=lambda r: r["test"])
        bonus = best_test["test"] - win["test"]
        results[ds] = {"winner": win["tag"], "kept": win["tag"] == default,
                       "n_arms": len(rows), "in_band": len(band), "bonus": bonus,
                       "winner_test": win["test"], "best_arm": best_test["tag"]}
        print(f"{ds:11s} {len(rows):4d} {win['tag']:26s} "
              f"{str(win['tag'] == default):>12s} {len(band):>4d}/{len(rows):<3d} "
              f"{bonus:+8.4f}")
    if incomplete:
        print(f"\nNOT READY: {', '.join(incomplete)}")
        return 1

    kept = all(r["kept"] for r in results.values())
    bonus_ok = all(r["bonus"] < 0.02 for r in results.values())
    print(f"\nP36  default retained on all four: {kept};  every bonus < 0.02: {bonus_ok}")
    print(f"  VERDICT: {'CONFIRMED' if kept and bonus_ok else 'FALSIFIED'}")
    if not kept:
        for ds, r in results.items():
            if not r["kept"]:
                print(f"    {ds}: validation chose {r['winner']} over {SPEC[ds][0]}")

    # P37: do the three methods move the same way on Amazon?
    print("\nP37  Amazon: sign of each method's selection delta")
    print("  OUTPOST  -0.0040  (T_hidden16 vs A_sim0.0, P32)")
    print("  NSReg    +0.0054  (NT_lr0.003_wd0.0 vs E400_nsreg, P35)")
    a = results.get("amazon")
    if a:
        d = a["winner_test"] - next(r["test"] for r in arms("amazon", "DT_", "B_demo_mix")
                                    if r["tag"] == "B_demo_mix")
        print(f"  DEMO     {d:+.4f}  ({a['winner']} vs B_demo_mix)")
        # Compare the SIGNS of the three deltas. The first version built
        # {-0.0040 < 0, 0.0054 > 0, d > 0} - three booleans each asking "did this
        # match its own expectation", all True, which collapsed to one element and
        # read as "they agree in sign". It would have reported FALSIFIED for
        # (-0.0040, +0.0054, +0.0041), which plainly does not move one way.
        def sign(x):
            return 0 if abs(x) < 1e-9 else (1 if x > 0 else -1)
        signs = {sign(-0.0040), sign(0.0054), sign(d)}
        differ = len(signs) > 1
        print(f"  signs: OUTPOST {sign(-0.0040):+d}  NSReg {sign(0.0054):+d}  "
              f"DEMO {sign(d):+d}")
        print(f"  VERDICT: {'CONFIRMED' if differ else 'FALSIFIED'} "
              f"(deltas {'differ' if differ else 'agree'} in sign)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
