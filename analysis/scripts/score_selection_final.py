#!/usr/bin/env python3
"""Score P32-P35 and P40 from the shards, once campaigns/sel_final.json has landed.

Written before the runs finished so the arithmetic behind each verdict is fixed
in advance and not chosen after seeing the numbers. Prints "NOT READY" and the
missing seeds rather than scoring a partial campaign - a three-seed answer
quoted as a ten-seed one is exactly the failure this file exists to prevent.
"""
import json, glob, sys, statistics as st
from collections import defaultdict
from scipy.stats import wilcoxon

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
SEL_SEEDS = [0, 1, 42]


def load(ds, method, tag):
    """{seed: (val_auc, oracle_roc, valsel_roc)} averaged over the rotation set."""
    per = defaultdict(lambda: ([], [], []))
    for f in glob.glob(f"results/rotations/{ds}_{method}_s*_{tag}_rot*.json"):
        j = json.load(open(f))
        for r in j["rotations"]:
            a, b, c = per[j["seed"]]
            a.append(r["val_selected"]["val_auc"])
            b.append(r["best"]["auroc_all"])
            c.append(r["val_selected"]["auroc_all"])
    return {s: (st.mean(v[0]), st.mean(v[1]), st.mean(v[2])) for s, v in per.items()}


def paired(a, b, idx):
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    x = [a[k][idx] for k in ks]
    y = [b[k][idx] for k in ks]
    dd = [p - q for p, q in zip(x, y)]
    p = 1.0 if all(abs(v) < 1e-12 for v in dd) else float(wilcoxon(x, y).pvalue)
    return {"n": len(ks), "delta": st.mean(dd),
            "wins": sum(1 for v in dd if v > 0), "p": p}


def need(name, d, seeds=SEEDS):
    missing = [s for s in seeds if s not in d]
    if missing:
        print(f"NOT READY: {name} missing seeds {missing}")
        return True
    return False


def main():
    hid = load("amazon", "outpost", "T_hidden16")
    dfl = load("amazon", "outpost", "A_sim0.0")
    nopl = load("amazon", "outpost", "C_nopl_hidden16")
    demo = load("amazon", "demo", "B_demo_mix")
    ns_ph_new = load("photo", "nsreg", "NT_lr0.003_wd0.0")
    ns_ph_old = load("photo", "nsreg", "E400_nsreg")
    ns_am_new = load("amazon", "nsreg", "NT_lr0.003_wd0.0")
    ns_am_old = load("amazon", "nsreg", "E400_nsreg")

    # Readiness is per prediction, not per campaign. Refusing a partial answer is
    # about one prediction's n being wrong, so blocking P32 because the NSReg arms
    # are still running would leave a scorable result unscored for no reason.
    ok_hid = not need("amazon T_hidden16", hid) and not need("amazon A_sim0.0", dfl)
    ok_nopl = not need("amazon C_nopl_hidden16", nopl)
    ok_ns = (not need("photo nsreg NT_lr0.003_wd0.0", ns_ph_new)
             and not need("amazon nsreg NT_lr0.003_wd0.0", ns_am_new))
    ok_demo = not need("amazon DEMO", demo)
    if not (ok_hid or ok_nopl or ok_ns):
        print("\nNothing scorable yet. Re-run as arms complete.")
        return 1

    print("=" * 72)

    if ok_hid:
        print("P32  Amazon at the validation-selected config vs the inherited default")
        r = paired(hid, dfl, 1)
        band = -0.012 <= r["delta"] <= -0.003
        signs = (10 - r["wins"]) >= 7
        print(f"  oracle delta {r['delta']:+.4f}  (registered band [-0.012, -0.003]: "
              f"{'IN' if band else 'OUT'})")
        print(f"  seeds negative {10 - r['wins']}/10 (registered >= 7: "
              f"{'yes' if signs else 'no'})   p={r['p']:.4f}")
        print(f"  VERDICT: {'CONFIRMED' if band and signs else 'FALSIFIED'}")
    else:
        print("P32  NOT READY")

    if ok_hid and ok_demo:
        print("\nP33  the head-to-head deficit against full DEMO, at the honest config")
        was = paired(dfl, demo, 1)
        now = paired(hid, demo, 1)
        band = -0.020 <= now["delta"] <= -0.006
        nowv = paired(hid, demo, 2)
        print(f"  was (A_sim0.0)   {was['delta']:+.4f}")
        print(f"  now (T_hidden16) {now['delta']:+.4f}  (registered [-0.020, -0.006]: "
              f"{'IN' if band else 'OUT'})  p={now['p']:.4f}")
        print(f"  val-selected     {nowv['delta']:+.4f} (stays a loss: "
              f"{'yes' if nowv['delta'] < 0 else 'NO'})")
        print(f"  VERDICT: {'CONFIRMED' if band and nowv['delta'] < 0 else 'FALSIFIED'}")
    else:
        print("\nP33  NOT READY")

    if ok_nopl and ok_hid:
        print("\nP34  pseudo-labelling at the new width")
        r = paired(nopl, hid, 1)
        good = abs(r["delta"]) <= 0.010
        print(f"  C_nopl - full at hidden_dim 16: {r['delta']:+.4f} "
              f"(registered within +/-0.010: {'yes' if good else 'no'})  "
              f"wins {r['wins']}/{r['n']}  p={r['p']:.4f}")
        print(f"  VERDICT: {'CONFIRMED' if good else 'FALSIFIED'}")
    else:
        print("\nP34  NOT READY")

    if ok_ns:
        print("\nP35  NSReg's tuning gain at ten seeds")
        ph = paired(ns_ph_new, ns_ph_old, 1)
        am = paired(ns_am_new, ns_am_old, 1)
        good = ph["delta"] >= 0.005 and am["delta"] >= 0.002
        print(f"  Photo  {ph['delta']:+.4f} (registered >= +0.005)  "
              f"wins {ph['wins']}/{ph['n']}  p={ph['p']:.4f}")
        print(f"  Amazon {am['delta']:+.4f} (registered >= +0.002)  "
              f"wins {am['wins']}/{am['n']}  p={am['p']:.4f}")
        print(f"  VERDICT: {'CONFIRMED' if good else 'FALSIFIED'}")
    else:
        print("\nP35  NOT READY")

    if ok_hid:
        print("\nP40  does Amazon's validation preference survive ten seeds?")
        v3h = st.mean([hid[s][0] for s in SEL_SEEDS])
        v3d = st.mean([dfl[s][0] for s in SEL_SEEDS])
        vAh = st.mean([hid[s][0] for s in SEEDS])
        vAd = st.mean([dfl[s][0] for s in SEEDS])
        print(f"  3 selection seeds: T_hidden16 {v3h:.4f} vs A_sim0.0 {v3d:.4f} "
              f"-> margin {v3h - v3d:+.4f}")
        print(f"  all ten seeds:     T_hidden16 {vAh:.4f} vs A_sim0.0 {vAd:.4f} "
              f"-> margin {vAh - vAd:+.4f}")
        holds = (vAh - vAd) > 0.002
        print(f"  VERDICT: {'CONFIRMED' if holds else 'FALSIFIED'} "
              f"(margin {'above' if holds else 'inside/below'} the 0.002 tie band)")
        print("\n  Reported selection is unchanged either way: the pre-registered "
              "rule uses the three selection seeds.")
    else:
        print("\nP40  NOT READY")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
