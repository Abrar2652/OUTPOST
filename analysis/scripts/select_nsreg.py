#!/usr/bin/env python3
"""Validation-only selection over the NSReg lr x weight_decay grid.

The rule was fixed before the campaign ran, in the `_comment` of
campaigns/nsreg_tune.json: "the wrapper records val_auc on OUTPOST's validation
split, and selection is on that alone."  Same tie band (0.002 mean val_auc) and
same tie-break (retain the released configuration) as select_hparams.py uses for
OUTPOST, so the two methods are tuned under one procedure.

Test metrics are read only after the winner is fixed, and are never an input to
the choice.
"""
import json, glob, statistics as st, sys
from collections import defaultdict

BAND = 0.002
DEFAULT = "E400_nsreg"          # NSReg's own released configuration
SEEDS = (0, 1, 42)              # the three selection seeds, as for OUTPOST


def arms(ds):
    out = defaultdict(dict)
    for f in glob.glob(f"results/rotations/{ds}_nsreg_s*.json"):
        j = json.load(open(f))
        tag = j["tag"]
        if not (tag.startswith("NT_") or tag == DEFAULT):
            continue
        if j["seed"] not in SEEDS:
            continue
        for r in j["rotations"]:
            out[tag].setdefault(j["seed"], []).append(
                (r["val_selected"]["val_auc"], r["best"]["auroc_all"],
                 r["val_selected"]["auroc_all"]))
    rows = []
    for tag, seeds in out.items():
        v = [st.mean([x[0] for x in rs]) for rs in seeds.values()]
        t = [st.mean([x[1] for x in rs]) for rs in seeds.values()]
        vt = [st.mean([x[2] for x in rs]) for rs in seeds.values()]
        rows.append({"tag": tag, "n": len(v), "val_auc": st.mean(v),
                     "sd": st.stdev(v) if len(v) > 1 else 0.0,
                     "_oracle": st.mean(t), "_valsel": st.mean(vt)})
    return sorted(rows, key=lambda r: -r["val_auc"])


def select(rows, expect):
    if expect and len(rows) != expect:
        return None, f"expected {expect} arms, found {len(rows)}"
    if not rows:
        return None, "no arms"
    incomplete = [r["tag"] for r in rows if r["n"] < len(SEEDS)]
    if incomplete:
        return None, f"incomplete arms: {incomplete}"
    top = rows[0]["val_auc"]
    band = [r for r in rows if top - r["val_auc"] <= BAND]
    default = [r for r in band if r["tag"] == DEFAULT]
    return (default[0] if default else band[0]), None


def main():
    ds_list = sys.argv[1:] or ["photo", "computers", "yelp", "amazon"]
    out = {}
    for ds in ds_list:
        rows = arms(ds)
        win, err = select(rows, None)
        if err:
            print(f"{ds}: SKIP ({err})")
            continue
        top = rows[0]["val_auc"]
        band = [r for r in rows if top - r["val_auc"] <= BAND]
        out[ds] = {
            "winner": win["tag"],
            "retained_default": win["tag"] == DEFAULT,
            "criterion": "mean val_auc over seeds 0/1/42, validation split only",
            "tie_band": BAND,
            "n_arms": len(rows),
            "n_in_band": len(band),
            "val_spread": round(top - rows[-1]["val_auc"], 6),
            "arms": [{k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in rows],
        }
        print(f"{ds:10s} winner {win['tag']:22s} "
              f"(default retained: {win['tag'] == DEFAULT})  "
              f"{len(band)}/{len(rows)} in band  spread {top - rows[-1]['val_auc']:.4f}")
        for r in rows:
            mark = "*" if r["tag"] == win["tag"] else " "
            print(f"   {mark} {r['tag']:22s} val {r['val_auc']:.4f}  "
                  f"oracle {r['_oracle']:.4f}  valsel {r['_valsel']:.4f}")
    json.dump(out, open("analysis/tables/nsreg_selection.json", "w"), indent=1)
    print("\nwrote analysis/tables/nsreg_selection.json")


if __name__ == "__main__":
    main()
