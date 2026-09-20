#!/usr/bin/env python3
"""Test the oracle-inflation ordering: semi-synthetic > OGB > real.

The paper quotes three ranges. This checks whether the ordering is a fact about
the data or an artefact of how the ranges were drawn, and writes the numbers so no
one has to retype them.

The claim that carries is COMPLETE SEPARATION at the dataset level - every graph
of one kind inflating more than every graph of the next. The pooled seed-level
Mann-Whitney p-values are also reported, with the caveat that they treat seeds as
independent when seeds within a graph share a split, so they overstate the
evidence and are descriptive only.

    python analysis/scripts/inflation_ordering.py -> analysis/tables/inflation_ordering.json
"""
import glob, json, os, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

KIND = {"photo": "semi-synthetic", "computers": "semi-synthetic", "cs": "semi-synthetic",
        "ogbn-arxiv": "OGB", "ogbn-mag": "OGB",
        "yelp": "real", "amazon": "real", "tfinance": "real"}
ORDER = ["semi-synthetic", "OGB", "real"]


def main_tag(ds):
    base = {"photo": "E400_outpost", "computers": "E400_outpost", "cs": "E400_outpost",
            "yelp": "A_main", "ogbn-arxiv": "A_main", "ogbn-mag": "A_main"}
    if ds in base:
        return base[ds]
    try:
        sel = json.load(open("analysis/tables/arm_selection.json")).get(ds)
        return sel["arm"] if isinstance(sel, dict) else sel or "A_sim0.0"
    except Exception:
        return "A_sim0.0"


def main():
    rows, pooled = [], defaultdict(list)
    for ds, kind in KIND.items():
        per = defaultdict(list)
        for f in glob.glob(f"results/rotations/{ds}_outpost_s*_{main_tag(ds)}_rot*.json"):
            j = json.load(open(f))
            for r in j["rotations"]:
                per[j["seed"]].append(r["best"]["auroc_all"]
                                      - r["val_selected"]["auroc_all"])
        g = [st.mean(v) for v in per.values()]
        if not g:
            continue
        rows.append({"dataset": ds, "kind": kind, "n": len(g),
                     "inflation": round(st.mean(g), 4),
                     "sd": round(st.stdev(g), 4) if len(g) > 1 else None,
                     "tag": main_tag(ds)})
        pooled[kind] += g
    rows.sort(key=lambda r: -r["inflation"])

    # complete separation: does every graph of kind A exceed every graph of kind B?
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["inflation"])
    sep = []
    for i in range(len(ORDER) - 1):
        a, b = ORDER[i], ORDER[i + 1]
        if by_kind[a] and by_kind[b]:
            sep.append({"higher": a, "lower": b,
                        "complete_separation": min(by_kind[a]) > max(by_kind[b]),
                        "min_higher": round(min(by_kind[a]), 4),
                        "max_lower": round(max(by_kind[b]), 4)})
    out = {"datasets": rows, "separation": sep,
           "all_pairs_separated": all(s["complete_separation"] for s in sep)}

    try:
        from scipy.stats import mannwhitneyu
        tests = []
        for i in range(len(ORDER)):
            for j in range(i + 1, len(ORDER)):
                a, b = ORDER[i], ORDER[j]
                if pooled[a] and pooled[b]:
                    p = mannwhitneyu(pooled[a], pooled[b], alternative="greater").pvalue
                    tests.append({"higher": a, "lower": b, "n_higher": len(pooled[a]),
                                  "n_lower": len(pooled[b]), "p": float(p)})
        out["pooled_seed_tests"] = tests
        out["pooled_caveat"] = ("seeds within a graph share a split, so these treat "
                                "non-independent observations as independent and "
                                "overstate the evidence; descriptive only")
    except Exception:
        pass

    json.dump(out, open("analysis/tables/inflation_ordering.json", "w"), indent=1)
    print(f"{'graph':11s} {'kind':15s} {'n':>3s} {'inflation':>10s}")
    for r in rows:
        print(f"{r['dataset']:11s} {r['kind']:15s} {r['n']:3d} {r['inflation']:10.4f}"
              + (f" ± {r['sd']:.4f}" if r["sd"] else ""))
    print()
    for s in sep:
        print(f"  every {s['higher']} ({s['min_higher']:.4f} lowest) > every "
              f"{s['lower']} ({s['max_lower']:.4f} highest): "
              f"{'YES' if s['complete_separation'] else 'NO'}")
    for t in out.get("pooled_seed_tests", []):
        print(f"  pooled {t['higher']} > {t['lower']}: p={t['p']:.1e} "
              f"(n={t['n_higher']},{t['n_lower']}) -- descriptive only")
    print("\n-> analysis/tables/inflation_ordering.json")


if __name__ == "__main__":
    main()
