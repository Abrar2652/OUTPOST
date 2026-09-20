#!/usr/bin/env python3
"""Is the selection tie band above or below the noise of the statistic it thresholds?

The rule this study uses - and the one the surrounding literature uses - picks a
configuration when its mean validation AUC over three seeds beats the default's by
more than 0.002. That threshold is only meaningful if 0.002 is large relative to
the sampling error of that COMPARISON.

The quantity that matters is the standard error of the PAIRED DIFFERENCE between
two arms on their shared seeds, not the spread of either arm on its own. Arms are
evaluated on the same seeds, so part of each arm's seed-to-seed variation is
common to both and cancels. Using the marginal spread instead inflates the answer
badly where arms are correlated: it made Yelp look like it needed 379 seeds when
the paired figure is 42.

Measured over every pair of arms of the same method on the same dataset that share
at least three seeds - thousands of pairs, not the handful where a selection fired.

    python analysis/scripts/selection_noise.py  ->  analysis/tables/selection_noise.json
"""
import itertools, json, glob, os, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
BAND = 0.002
MIN_SEEDS = 5
DS = ["photo", "computers", "cs", "yelp", "amazon", "tfinance"]


def main():
    out = {"tie_band": BAND, "min_shared_seeds": 3,
           "statistic": "sd of the paired per-seed difference between two arms of "
                        "the same method, on their shared seeds", "datasets": []}
    for ds in DS:
        per = defaultdict(dict)
        for f in glob.glob(f"results/rotations/{ds}_*_s*_*.json"):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            vals = [r["val_selected"]["val_auc"] for r in j.get("rotations", [])
                    if r.get("val_selected", {}).get("val_auc") is not None]
            if vals:
                per[(j["method"], j["tag"])][j["seed"]] = st.mean(vals)
        diffs, widest = [], None
        for (ma, ta), (mb, tb) in itertools.combinations(sorted(per), 2):
            if ma != mb:
                continue                      # selection compares arms of one method
            a, b = per[(ma, ta)], per[(mb, tb)]
            ks = sorted(set(a) & set(b))
            if len(ks) < 3:
                continue
            d = [a[k] - b[k] for k in ks]
            diffs.append(st.stdev(d))
        if not diffs:
            continue
        for k, v in per.items():
            if len(v) >= MIN_SEEDS:
                rng = max(v.values()) - min(v.values())
                if widest is None or rng > widest[0]:
                    widest = (rng, k, v)
        sd = st.median(diffs)
        rec = {"dataset": ds, "arm_pairs": len(diffs),
               "paired_diff_sd": round(sd, 4),
               "se_3_seeds": round(sd / 3 ** 0.5, 4),
               "se_10_seeds": round(sd / 10 ** 0.5, 4),
               "band_below_noise_at_3": sd / 3 ** 0.5 > BAND,
               "band_below_noise_at_10": sd / 10 ** 0.5 > BAND,
               "seeds_needed_for_band": int((sd / BAND) ** 2 + 0.999)}
        if widest:
            rec["widest_arm"] = {"tag": widest[1][1], "method": widest[1][0],
                                 "n": len(widest[2]),
                                 "min": round(min(widest[2].values()), 4),
                                 "max": round(max(widest[2].values()), 4),
                                 "range": round(widest[0], 4)}
        out["datasets"].append(rec)
    n = len(out["datasets"])
    out["summary"] = {
        "n_datasets": n,
        "band_below_noise_at_3": sum(d["band_below_noise_at_3"] for d in out["datasets"]),
        "band_below_noise_at_10": sum(d["band_below_noise_at_10"] for d in out["datasets"]),
        "max_seeds_needed": max((d["seeds_needed_for_band"] for d in out["datasets"]),
                                default=None),
    }
    json.dump(out, open("analysis/tables/selection_noise.json", "w"), indent=1)
    print(f"{'dataset':11s} {'pairs':>6s} {'paired-diff sd':>15s} {'SE(3)':>8s} "
          f"{'SE(10)':>8s} {'seeds needed':>13s}")
    for d in out["datasets"]:
        print(f"{d['dataset']:11s} {d['arm_pairs']:6d} {d['paired_diff_sd']:15.4f} "
              f"{d['se_3_seeds']:8.4f} {d['se_10_seeds']:8.4f} "
              f"{d['seeds_needed_for_band']:13d}")
    s_ = out["summary"]
    print(f"\n{s_['band_below_noise_at_3']}/{n} datasets: the {BAND} band is BELOW the "
          f"standard error of the 3-seed paired difference")
    print(f"{s_['band_below_noise_at_10']}/{n} datasets: still below it at 10 seeds")
    print(f"seeds needed for the band to exceed one standard error: up to "
          f"{s_['max_seeds_needed']}")
    print("\n-> analysis/tables/selection_noise.json")


if __name__ == "__main__":
    main()
