"""Every table behind the paper's claims, regenerated from the run record.

One file so the appendix cannot drift from results.csv: each section states the
tag it selects and the n it found, and a dataset with no runs at a given setting
prints as absent rather than inheriting a neighbour's number.

    python analysis/scripts/build_appendix.py            # -> analysis/appendix.json + .md
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

# The uniform 400-epoch protocol. photo/computers/cs were run at 200 in A_main
# and re-run at 400 under E400_*; yelp, arxiv and mag are natively 400 epochs
# (config.json), so A_main already is their 400-epoch arm.
DS = [("photo", "Photo"), ("computers", "Computers"), ("cs", "CS"),
      ("yelp", "Yelp"), ("amazon", "Amazon"), ("tfinance", "T-Finance"),
      ("ogbn-arxiv", "ogbn-arxiv"), ("ogbn-mag", "ogbn-mag")]
E400 = {"photo": ("E400_outpost", "E400_demo"),
        "computers": ("E400_outpost", "E400_demo"),
        "cs": ("E400_outpost", "E400_demo"),
        "yelp": ("A_main", "B_demo_mix"),
        # amazon / tfinance: the SimSample arm is chosen on VALIDATION from the
        # shards, once, by select_arm(); written to arm_selection.json for the
        # other scripts. Test metrics are never consulted.
        "amazon": ("__select__", "B_demo_mix"),
        "tfinance": ("__select__", "B_demo_mix"),
        "ogbn-arxiv": ("A_main", "B_demo_mix"),   # DEMO now runs here (METHODOLOGY 11.2)
        "ogbn-mag": ("A_main", None)}
M = [("best_auroc", "oracle ROC"), ("best_aupr", "oracle PR"),
     ("valsel_auroc", "valsel ROC"), ("valsel_aupr", "valsel PR")]


def load():
    d = pd.read_csv("results/results.csv").drop_duplicates(
        subset=["dataset", "method", "seed", "train_seed", "tag"], keep="last")
    # the same smoke-run guard every other consumer applies; without it a
    # DEMO cell here disagreed with the LaTeX table in the last digit
    return d[d.num_epochs >= 100]


def load_shards():
    """Per-rotation metrics at FULL precision, from results/rotations/*.json.

    results.csv stores four decimals. On Amazon that rounded two different
    methods' val-selected AUC-ROC to the same 4-dp value for one seed, which a
    paired test read as a tie: the appendix reported p=0.0077 (0/1/9) while
    stats.py, which reads the shards, reported p=0.0020 (0/0/10) for the same
    comparison. Two p-values for one test is a reviewer's catch. Paired tests
    now come from here.
    """
    import glob
    rows = []
    for p in glob.glob("results/rotations/*.json"):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        for rot in r.get("rotations", []):
            b, v = rot.get("best", {}), rot.get("val_selected", {})
            rows.append({"dataset": r["dataset"], "method": r["method"],
                         "seed": int(r["seed"]), "train_seed": r.get("train_seed"),
                         "tag": r.get("tag", ""), "rotation": int(rot["rotation_class"]),
                         "best_auroc": b.get("auroc_all"), "best_aupr": b.get("aupr_all"),
                         "valsel_auroc": v.get("auroc_all"), "valsel_aupr": v.get("aupr_all"),
                         # the open-set metric proper: the classes never labelled
                         "best_auroc_unseen": b.get("auroc_unknown"), "best_aupr_unseen": b.get("aupr_unknown"),
                         "valsel_auroc_unseen": v.get("auroc_unknown"), "valsel_aupr_unseen": v.get("aupr_unknown")})
    sh = pd.DataFrame(rows)
    # a re-run overwrites its shard file, so there is at most one per key
    return sh.drop_duplicates(subset=["dataset", "method", "seed", "train_seed",
                                      "tag", "rotation"], keep="last")


SH = None


def select_arm(ds, arms=("A_sim1.0", "A_sim0.0")):
    """The SimSample arm with the higher mean validation AUC in the shards."""
    import glob as _g
    best, best_v = None, -1
    for tag in arms:
        vals = []
        for p in _g.glob(f"results/rotations/{ds}_outpost_s*_{tag}_rot*.json"):
            for rot in json.load(open(p))["rotations"]:
                v = rot.get("val_selected", {}).get("val_auc")
                if v is not None:
                    vals.append(v)
        if vals and sum(vals) / len(vals) > best_v:
            best, best_v = tag, sum(vals) / len(vals)
    return best
sys_path_dir = os.path.dirname(os.path.abspath(__file__))
import sys
if sys_path_dir not in sys.path:
    sys.path.insert(0, sys_path_dir)
from merge_rotations import ROTATIONS as _ROT   # the complete rotation set per dataset


def _complete_seeds(df, ds):
    """Seeds whose rotation set is complete for `ds`. A seed with 2 of 8 CS
    rotations finished must not enter a per-seed mean next to seeds with 8."""
    req = set(int(c) for c in _ROT.get(ds, []))
    if not req:
        return set(df.seed.unique())
    have = df.groupby("seed")["rotation"].apply(lambda r: set(int(x) for x in r))
    return set(have[have.apply(lambda s: req <= s)].index)


def paired2(ds, ma, ta, mb, tb, m):
    """Paired A-minus-B from the shards, at both units.

    seed level  : mean over rotations per seed, then paired over seeds
                  (conservative; what the appendix reports as n)
    unit level  : paired over (seed, rotation), the unit stats.py uses and
                  the paper states - legitimate because both arms see the
                  same split for the same rotation
    """
    global SH
    if SH is None:
        SH = load_shards()
    A = SH[(SH.dataset == ds) & (SH.method == ma) & (SH.tag == ta)]
    B = SH[(SH.dataset == ds) & (SH.method == mb) & (SH.tag == tb)]
    if A.empty or B.empty:
        return None
    A = A.dropna(subset=[m]); B = B.dropna(subset=[m])
    # a seed counts only when BOTH arms have its complete rotation set
    ok = _complete_seeds(A, ds) & _complete_seeds(B, ds)
    A, B = A[A.seed.isin(ok)], B[B.seed.isin(ok)]
    j = A.merge(B, on=["seed", "rotation"], suffixes=("_a", "_b"))
    if j.empty:
        return None
    du = (j[f"{m}_a"] - j[f"{m}_b"]).astype(float)
    ds_ = j.groupby("seed").apply(lambda g: (g[f"{m}_a"] - g[f"{m}_b"]).mean()).astype(float)
    if len(ds_) < 3:
        return None
    w = int((ds_ > 1e-9).sum()); l = int((ds_ < -1e-9).sum())
    # Two arms can be numerically identical on every seed - the same tag compared
    # with itself, or a component whose removal changes nothing at all. scipy
    # raises on an all-zero difference, and an unguarded raise here takes down
    # the whole appendix build and leaves a stale appendix.json behind, which is
    # exactly what happened between 13:59 and 22:30 on 2026-09-07. A tie on every
    # seed is p = 1 by definition, so say so instead of dying.
    _identical = bool((ds_.abs() < 1e-12).all())
    r = {"n": int(len(ds_)), "delta": round(ds_.mean(), 4), "W": w, "T": int(len(ds_)) - w - l, "L": l,
         "identical": _identical,
         "p_wilcoxon": 1.0 if _identical else round(float(wilcoxon(ds_).pvalue), 4),
         "p_ttest": 1.0 if _identical else round(float(ttest_rel(
             j.groupby("seed")[f"{m}_a"].mean(),
             j.groupby("seed")[f"{m}_b"].mean()).pvalue), 4),
         "wilcoxon_floor": round(2.0 ** -(len(ds_) - 1), 4)}
    # per-seed means rounded to 4 dp first: the path results.csv takes, so the paper tables agree to the last digit
    ga = j.groupby("seed")[f"{m}_a"].mean().round(4); gb = j.groupby("seed")[f"{m}_b"].mean().round(4)
    r.update(mean_a=float(f"{ga.mean():.4f}"), sd_a=float(f"{ga.std():.4f}"),      # per-arm mean±sd over the paired seeds,
             mean_b=float(f"{gb.mean():.4f}"), sd_b=float(f"{gb.std():.4f}"))      # so tex tables of either arm can be checked
    r["at_floor"] = r["p_wilcoxon"] <= r["wilcoxon_floor"] + 1e-9
    # the (seed, rotation) unit, as stats.py reports it
    wu = int((du > 1e-9).sum()); lu = int((du < -1e-9).sum())
    _ident_u = bool((du.abs() < 1e-12).all())
    r["units"] = {"n": int(len(du)), "delta": round(du.mean(), 4), "W": wu,
                  "T": int(len(du)) - wu - lu, "L": lu,
                  "p_wilcoxon": (1.0 if _ident_u else float(wilcoxon(du).pvalue))
                  if len(du) >= 5 else None}
    return r


def arm(d, ds, method, tag):
    return d[(d.dataset == ds) & (d.method == method) & (d.tag == tag)].set_index("seed")


def msd(g, m):
    v = g[m]
    # format-then-parse, exactly as make_paper_tables.py prints its cells, so the
    # two cannot disagree in the last digit through different rounding paths
    return {"mean": float(f"{v.mean():.4f}"),
            "sd": float(f"{v.std(ddof=1):.4f}") if len(v) > 1 else None, "n": len(v)}


def paired(a, b, m):
    """Paired comparison on the seeds both arms share."""
    k = sorted(set(a.index) & set(b.index))
    if len(k) < 3:
        return None
    dl = (a.loc[k, m] - b.loc[k, m]).astype(float)
    w = int((dl > 1e-6).sum()); l = int((dl < -1e-6).sum())
    r = {"n": len(k), "delta": round(dl.mean(), 4), "W": w, "T": len(k) - w - l, "L": l,
         "p_wilcoxon": round(float(wilcoxon(dl).pvalue), 4),
         "p_ttest": round(float(ttest_rel(a.loc[k, m], b.loc[k, m]).pvalue), 4)}
    # At n pairs the two-sided Wilcoxon cannot go below 2^-(n-1); report it so a
    # p at the floor is not read as a marginal effect.
    r["wilcoxon_floor"] = round(2.0 ** -(len(k) - 1), 4)
    r["at_floor"] = r["p_wilcoxon"] <= r["wilcoxon_floor"] + 1e-9
    return r


def main():
    global SH
    d = load()
    sel = {}
    for ds, (ot, dt) in list(E400.items()):
        if ot == "__select__":
            chosen = select_arm(ds)
            if chosen:
                E400[ds] = (chosen, dt); sel[ds] = chosen
            else:
                del E400[ds]          # not run yet: dataset simply absent
    json.dump(sel, open("analysis/tables/arm_selection.json", "w"), indent=1)
    out = {"protocol": "uniform 400 epochs; per-dataset tags listed per row",
           "generated_from": "results/results.csv"}

    # --- 1 + 2. main table, oracle and val-selected together -----------------
    main_tbl = []
    # NSReg at the configuration its own validation-only sweep selected, wherever
    # that selection survives ten seeds (P35): Amazon's does (+0.0054, 9/10,
    # p=0.006) and is adopted; Photo's does not (+0.0106 at three seeds, -0.0009
    # at ten) and keeps the released configuration.
    NS_MAIN = {"photo": "E400_nsreg", "computers": "E400_nsreg", "cs": "E400_nsreg",
               "yelp": "E400_nsreg", "amazon": "NT_lr0.003_wd0.0",
               "tfinance": "E400_nsreg", "ogbn-arxiv": "E400_nsreg"}
    for ds, label in DS:
        if ds not in E400:
            continue
        ot, dt = E400[ds]
        for meth, tag in (("outpost", ot), ("demo", dt), ("nsreg", NS_MAIN.get(ds))):
            if tag is None:
                continue
            g = arm(d, ds, meth, tag)
            if not len(g):
                continue
            row = {"dataset": label, "method": meth.upper(), "tag": tag}
            for m, lab in M:
                row[lab] = msd(g, m)
            main_tbl.append(row)
    out["main_table"] = main_tbl

    # --- 2b. baseline hyperparameter selection, same rule as OUTPOST's -------
    # NSReg and DEMO are swept under select_hparams.py's rule (validation only,
    # 0.002 tie band, ties to the published configuration) so a referee can see
    # that the sweep was not a courtesy extended only to our own method. The
    # records are written by select_nsreg.py before this runs.
    try:
        _ns = json.load(open("analysis/tables/nsreg_selection.json"))
    except Exception:
        _ns = {}
    ns_sel = []
    for ds, label in DS:
        if ds not in _ns:
            continue
        e = _ns[ds]
        rel = next((a for a in e["arms"] if a["tag"] == "E400_nsreg"), None)
        win = next((a for a in e["arms"] if a["tag"] == e["winner"]), None)
        ns_sel.append({
            "dataset": label, "method": "NSREG",
            "winner": e["winner"], "retained_default": e["retained_default"],
            "n_arms": e["n_arms"], "n_in_band": e["n_in_band"],
            "tie_band": e["tie_band"], "val_spread": e["val_spread"],
            "released_oracle": rel and round(rel["_oracle"], 4),
            "selected_oracle": win and round(win["_oracle"], 4),
            "delta_oracle": (None if not (rel and win)
                             else round(win["_oracle"] - rel["_oracle"], 4)),
            "delta_valsel": (None if not (rel and win)
                             else round(win["_valsel"] - rel["_valsel"], 4)),
            "n_seeds": win and win["n"],
        })
    out["baseline_selection"] = ns_sel

    # --- 3. paired OUTPOST vs DEMO ------------------------------------------
    pairs = []
    for ds, label in DS:
        if ds not in E400:
            continue
        ot, dt = E400[ds]
        if dt is None:
            pairs.append({"dataset": label, "status":
                          "no DEMO arm: mixup needs a dense NxN PPR "
                          "(115 GB on arxiv, 2.2 TB on mag). NOTE 2026-09-14: "
                          "only the mag figure is a real obstacle. The arxiv "
                          "matrix was built in 8 minutes and DEMO runs there; "
                          "NSReg runs there too, in 2.17 GB. These cells are "
                          "empty for COST, not feasibility."})
            continue
        r = {"dataset": label}
        for m, lab in M:
            r[lab] = paired2(ds, "outpost", ot, "demo", dt, m)
        pairs.append(r)
    out["paired_vs_demo"] = pairs

    # --- 4. ablation arms ----------------------------------------------------
    # The mechanism campaign (2026-09-03) added the single-flag suite and the
    # gate+PL pair to the two datasets that carry the wins; they were missing
    # here, so the decomposition the paper's component story rests on was not
    # in the appendix at all.
    ABL = {"photo": ["C_nogate", "C_nopl", "C_noconformal", "L_lean",
                     "C_nosynth", "C_sim"],
           "computers": ["C_nogate", "C_nopl", "C_noconformal", "L_lean", "C_sim"],
           "cs": ["C_nogate", "C_nopl", "C_noconformal", "L_lean", "C_sim"],
           "ogbn-arxiv": ["C_sim"],
           "yelp": ["C_nogate", "C_nopl", "C_noconformal", "L_lean",
                    "C_nosim", "C_simplacebo", "C_fview", "C_hopmix", "C_synth"],
           "amazon": ["C_nogate", "C_noconformal", "C_nopl", "L_lean", "C_simplacebo", "A_sim1.0"],
           "tfinance": ["C_nopl", "A_sim1.0", "A_sim0.0"]}
    # Amazon and T-Finance name their arms after the SimSample setting or the
    # width they were selected at, so the ablation of the selected arm is
    # "C_nopl_sim0.0" or "C_nopl_hidden16", never a bare "C_nopl". Listing the
    # bare tag silently dropped T-Finance's pseudo-labelling row - the single
    # most important result on that graph (P28) - from this table. Resolve the
    # suffix from whatever arm validation actually selected.
    for _ds in ("amazon", "tfinance"):
        if _ds not in E400:
            continue
        _base = E400[_ds][0]
        _suffix = ("_" + _base.split("_", 1)[1]) if "_" in _base else ""
        # try the suffixed name first, then the bare one: T-Finance's PL arm is
        # C_nopl_sim0.0 but its gate and conformal arms are plain C_nogate /
        # C_noconformal, and only checking the suffixed form dropped P38 and P39's
        # evidence out of the appendix entirely
        _want = []
        for _stem in ("C_nopl", "C_nogate", "C_noconformal"):
            for _cand in (f"{_stem}{_suffix}", _stem):
                if len(arm(d, _ds, "outpost", _cand)):
                    _want.append(_cand)
                    break
        ABL[_ds] = _want + [t for t in ABL.get(_ds, []) if t not in _want]
    abl = []
    for ds, label in DS:
        base = arm(d, ds, "outpost", E400.get(ds, ("A_main",))[0] if ds in ("amazon", "tfinance") else "A_main")
        base_tag = E400.get(ds, ("A_main",))[0] if ds in ("amazon", "tfinance") else "A_main"
        for tag in ABL.get(ds, []):
            if tag == base_tag:
                continue          # the base is not an ablation of itself
            g = arm(d, ds, "outpost", tag)
            if not len(g):
                continue
            row = {"dataset": label, "arm": tag,
                   "arm_roc": msd(g, "best_auroc"), "base_roc": msd(base, "best_auroc")}
            row["delta_roc"] = paired2(ds, "outpost", tag, "outpost", base_tag, "best_auroc")
            row["delta_pr"] = paired2(ds, "outpost", tag, "outpost", base_tag, "best_aupr")
            abl.append(row)
    out["ablations"] = abl

    # --- 5a. oracle inflation: best-over-epochs minus val-selected -----------
    infl = []
    for ds, label in DS:
        if ds not in E400:
            continue
        ot, _ = E400[ds]
        g = arm(d, ds, "outpost", ot)
        if not len(g):
            continue
        for m1, m2, lab in [("best_auroc", "valsel_auroc", "ROC"),
                            ("best_aupr", "valsel_aupr", "PR")]:
            gap = (g[m1] - g[m2]).astype(float)
            infl.append({"dataset": label, "metric": lab, "n": len(g),
                         "oracle": round(g[m1].mean(), 4),
                         "valsel": round(g[m2].mean(), 4),
                         "inflation": round(gap.mean(), 4),
                         "sd": round(gap.std(ddof=1), 4) if len(g) > 1 else None})
    # the same gap on the UNSEEN classes, from the shards (results.csv has no
    # val-selected unseen columns). Multi-class graphs only.
    if SH is None:
        SH = load_shards()
    for ds, label in DS:
        if ds in ("yelp", "amazon", "tfinance") or ds not in E400:
            continue
        ot, _ = E400[ds]
        g = SH[(SH.dataset == ds) & (SH.method == "outpost") & (SH.tag == ot)]
        for m1, m2, lab in [("best_auroc_unseen", "valsel_auroc_unseen", "ROC (unseen)"),
                            ("best_aupr_unseen", "valsel_aupr_unseen", "PR (unseen)")]:
            gg = g.dropna(subset=[m1, m2])
            if gg.empty:
                continue
            per_seed = gg.groupby("seed")[[m1, m2]].mean()
            gap = (per_seed[m1] - per_seed[m2]).astype(float)
            infl.append({"dataset": label, "metric": lab, "n": int(len(per_seed)),
                         "oracle": round(per_seed[m1].mean(), 4),
                         "valsel": round(per_seed[m2].mean(), 4),
                         "inflation": round(gap.mean(), 4),
                         "sd": round(gap.std(ddof=1), 4) if len(gap) > 1 else None})
    out["oracle_inflation"] = infl

    # --- 5b. hyperparameter selection gap ------------------------------------
    hp = []
    NATIVE400 = {"yelp", "amazon", "tfinance", "ogbn-arxiv", "ogbn-mag"}   # their T_* sweep already ran at 400
    for ds, label in DS:
      for suffix, epochs in (("", 400 if ds in NATIVE400 else 200), ("_400", 400)):
        p = f"analysis/tables/hparam_selection_{ds}{suffix}.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        win = j["winner"]
        # Every arm must be scored on the SAME seeds or the gap is an artefact of
        # n. The swept arms have the three selection seeds; the default usually has
        # ten, and averaging it over all ten while the challengers get three made
        # Computers' 400-epoch gap read 0.0016 against a seed-matched 0.0041. Worse,
        # once validation stopped picking the default (Amazon), "gap from the
        # default" and "gap from what validation picked" stopped being the same
        # number - and the registered quantity is the latter.
        sel_seeds = set(j["selection_seeds"])
        def _on_sel(tag):
            g = arm(d, ds, "outpost", tag)
            g = g[g.index.isin(sel_seeds)] if len(g) else g
            return g
        best_test, best_tag = -1, None
        for a in [win] + j["arms"]:
            g = _on_sel(a["tag"])
            if len(g) == len(sel_seeds) and g.best_auroc.mean() > best_test:
                best_test, best_tag = g.best_auroc.mean(), a["tag"]
        gw = _on_sel(win["tag"])
        hp.append({"dataset": label, "epochs": epochs, "n_arms": len(j["arms"]) + 1,
                   "selection_seeds": j["selection_seeds"],
                   "seed_matched": True,
                   "val_winner": win["tag"],
                   "retained_default": bool(j.get("retained_default",
                                                  win["tag"].endswith("outpost")
                                                  or win["tag"] in ("A_main", "A_sim0.0"))),
                   "val_winner_test_roc": round(gw.best_auroc.mean(), 4) if len(gw) else None,
                   "best_test_arm": best_tag,
                   "best_test_roc": round(best_test, 4) if best_tag else None,
                   "selection_gap": (round(best_test - gw.best_auroc.mean(), 4)
                                     if len(gw) and best_tag else None)})
    out["hyperparameter_selection"] = hp

    # --- 5c. epoch budget ----------------------------------------------------
    if os.path.exists("analysis/tables/epoch_budget.csv"):
        out["epoch_budget"] = pd.read_csv("analysis/tables/epoch_budget.csv").to_dict("records")
    if os.path.exists("analysis/tables/demo_gap_by_budget.csv"):
        out["demo_gap_by_budget"] = pd.read_csv(
            "analysis/tables/demo_gap_by_budget.csv").to_dict("records")

    # --- 6. detectability law ------------------------------------------------
    if os.path.exists("analysis/tables/detectability_law.json"):
        j = json.load(open("analysis/tables/detectability_law.json"))
        out["detectability_law"] = {
            "spearman_rho": round(j["spearman_rho"], 4),
            "n_classes": j["n_classes"], "n_datasets": j["n_datasets"],
            "p_asymptotic": j["p_asymptotic"],
            "p_permutation_within_dataset": j["p_permutation_within_dataset"],
            "cluster_bootstrap_ci95": [round(x, 4) for x in j["cluster_bootstrap_ci95"]],
            "leave_one_dataset_out": j["leave_one_dataset_out"],
            "classes": j["classes"]}

    # --- 7. pre-registration record -----------------------------------------
    mag = arm(d, "ogbn-mag", "outpost", "A_main")
    pb = pd.read_csv("analysis/tables/published_baselines.csv").set_index("Method")
    field = pb["ogbn-mag_AUC-ROC"].dropna()
    sim = {}
    for ds in ["photo", "computers", "cs", "ogbn-arxiv"]:
        r = paired2(ds, "outpost", "C_sim", "outpost", "A_main", "best_auroc")
        if r:
            sim[ds] = r
    ys = paired2("yelp", "outpost", "A_main", "outpost", "C_nosim", "best_auroc")
    if ys:
        sim["yelp"] = ys
    photo_curve = []
    for b in (5, 10, 50, 100):
        r = paired2("photo", "outpost", f"G_b{b}_sim1.0", "outpost", f"G_b{b}_sim0.0", "best_auroc")
        if r:
            photo_curve.append({"budget": b, **r})
    out["preregistration"] = {
        "P5": {
            "text": "OUTPOST's ogbn-mag AUC-ROC will fall below 0.60, i.e. it "
                    "will not meaningfully beat the published field. AUC-PR will "
                    "stay near the 0.004-0.006 band. A large OUTPOST gain on "
                    "ogbn-mag would falsify the law.",
            "roc": msd(mag, "best_auroc"), "pr": msd(mag, "best_aupr"),
            "published_field_roc": {k: float(v) for k, v in field.items()},
            "best_published_roc": float(field.max()),
            "margin_over_best_published": round(mag.best_auroc.mean() - field.max(), 4),
            "seeds_above_best_published": int((mag.best_auroc > field.max()).sum()),
            "pr_band_predicted": [0.004, 0.006],
            "verdict": "SPLIT - threshold held, claim did not",
            "detail": "AUC-ROC 0.5923 is below 0.60 as predicted, but the "
                      "prediction equated that with 'will not meaningfully beat "
                      "the published field' and we beat it by +0.0956 on 5/5 "
                      "seeds, with AUC-PR 0.0101 against a predicted 0.004-0.006 "
                      "band (1.87x the best published). By P5's own stated "
                      "falsification condition this is the 'large gain on mag' "
                      "that puts pressure on the detectability law."},
        "P14": {
            "text": "If the subsampling-removal mechanism is correct, CS shows "
                    "the largest SimSample harm of any dataset - more negative "
                    "than Photo's -0.0318 - because 89.0% of its nodes sit in "
                    "the band where subsampling can be removed.",
            "simsample_delta_roc_by_dataset": sim,
            "verdict": "FALSIFIED",
            "detail": "CS gives +0.0005 (4/5 seeds, p=0.125), near zero rather "
                      "than the largest harm. This matches ogbn-arxiv (-0.0004) "
                      "and kills the subsampling-removal mechanism. NOTE: "
                      "prediction_budget.md still records P14 as 'CANNOT BE "
                      "TESTED ON THIS HARDWARE' - that was true on the 24 GB "
                      "card and stopped being true after the A6000 migration; "
                      "the arm has since run."},
        "P17": {
            "text": "Delta(b) on Photo will be positive or near-zero at b=5 and "
                    "b=10, most negative near b=25, and return toward zero at "
                    "b=100.",
            "photo_budget_curve": photo_curve,
            "verdict": "FALSIFIED",
            "detail": "Delta is negative at every budget including b=5, where "
                      "85.1% of nodes are above budget and the prediction "
                      "required positive or near-zero. The non-monotone shape "
                      "does not reproduce within one graph, so the cross-dataset "
                      "ordering has no within-graph support."}}

    # --- 9. second baseline: NSReg -------------------------------------------
    # same rule as NS_MAIN above: the paired comparison and the three-method
    # ranking must face the same NSReg the main table reports
    NS = dict(NS_MAIN)
    vs_ns, three, repro = [], [], []
    for ds, label in DS:
        if ds not in NS or ds not in E400:
            continue
        ot, dt = E400[ds]
        o, n_, dm = arm(d, ds, "outpost", ot), arm(d, ds, "nsreg", NS[ds]), \
                    (arm(d, ds, "demo", dt) if dt else None)
        if not len(n_):
            continue
        r = {"dataset": label, "n_nsreg": int(len(n_))}
        for m, lab in M:
            r[lab] = paired2(ds, "outpost", ot, "nsreg", NS[ds], m)
        vs_ns.append(r)
        # three methods, two selection rules: does the ranking change?
        row = {"dataset": label}
        for m, lab in (("best_auroc", "oracle"), ("valsel_auroc", "valsel")):
            vals = {"OUTPOST": float(o[m].mean()) if len(o) else None,
                    "DEMO": float(dm[m].mean()) if dm is not None and len(dm) else None,
                    "NSReg": float(n_[m].mean())}
            present = {k: v for k, v in vals.items() if v is not None}
            order = sorted(present, key=lambda k: -present[k])
            row[lab] = {"means": {k: round(v, 4) for k, v in present.items()},
                        "ranking": order}
        row["ranking_changes"] = row["oracle"]["ranking"] != row["valsel"]["ranking"]
        three.append(row)
    out["paired_vs_nsreg"] = vs_ns
    out["three_method_ranking"] = three
    # reproduction check: NSReg's own 201-epoch config vs its published numbers
    pb = pd.read_csv("analysis/tables/published_baselines.csv").set_index("Method")
    for ds, col in [("photo", "Photo"), ("computers", "Computers"),
                    ("cs", "CS"), ("yelp", "Yelp")]:
        n201 = arm(d, ds, "nsreg", "N201_nsreg")
        n400 = arm(d, ds, "nsreg", "E400_nsreg")
        if not len(n201) and not len(n400):
            continue
        pubv = float(pb.loc["NSReg", f"{col}_AUC-ROC"]) if "NSReg" in pb.index else None
        repro.append({"dataset": ds, "published": pubv,
                      "ours_201": round(float(n201.best_auroc.mean()), 4) if len(n201) else None,
                      "n_201": int(len(n201)),
                      "ours_400": round(float(n400.best_auroc.mean()), 4) if len(n400) else None,
                      "n_400": int(len(n400)),
                      "gap_201": round(pubv - float(n201.best_auroc.mean()), 4)
                                 if (pubv is not None and len(n201)) else None})
    out["nsreg_reproduction"] = repro

    # --- 10. the open-set metric: unseen anomaly classes ---------------------
    # Every table above scores ALL anomalies, which blends the labelled (seen)
    # class with the classes the model never saw. The paper's title is about
    # the latter. Binary graphs (yelp, amazon) have no unseen class and are
    # excluded; DEMO's trainer also writes 0.0 rather than 0.5 there.
    MU = [("best_auroc_unseen", "oracle ROC"), ("best_aupr_unseen", "oracle PR"),
          ("valsel_auroc_unseen", "valsel ROC"), ("valsel_aupr_unseen", "valsel PR")]
    unseen, unseen_three = [], []
    for ds, label in DS:
        if ds in ("yelp", "amazon", "tfinance") or ds not in E400:
            continue
        ot, dt = E400[ds]
        row = {"dataset": label}
        if dt:
            row["vs_DEMO"] = {lab: paired2(ds, "outpost", ot, "demo", dt, m) for m, lab in MU}
        if ds in NS:
            row["vs_NSReg"] = {lab: paired2(ds, "outpost", ot, "nsreg", NS[ds], m) for m, lab in MU}
        unseen.append(row)
        if SH is None:
            SH = load_shards()
        three = {"dataset": label}
        for m, lab in (("best_auroc_unseen", "oracle"), ("valsel_auroc_unseen", "valsel")):
            means = {}
            for meth, tag in (("OUTPOST", ot), ("DEMO", dt), ("NSReg", NS.get(ds))):
                if not tag:
                    continue
                g = SH[(SH.dataset == ds) & (SH.method == meth.lower()) & (SH.tag == tag)].dropna(subset=[m])
                g = g[g.seed.isin(_complete_seeds(g, ds))]
                if len(g):
                    means[meth] = round(float(g.groupby("seed")[m].mean().mean()), 4)
            three[lab] = {"means": means, "ranking": sorted(means, key=lambda k: -means[k])}
        three["ranking_changes"] = three["oracle"]["ranking"] != three["valsel"]["ranking"]
        unseen_three.append(three)
    out["paired_unseen"] = unseen
    out["three_method_ranking_unseen"] = unseen_three

    # --- 11. the gate, removed, at the paper's protocol ---------------------
    # Gate-off minus gate-on, OUTPOST, paired on seed. The atlas gate is a
    # NON-PARAMETRIC quantile veto (trainer.py applies torch.quantile at the
    # warmup boundary), so removing it changes no parameter count: the shards
    # record the identical n_params for E400_outpost and E400_nogate on every
    # dataset. An earlier comment here claimed 0.157x vs 0.289x DEMO; that
    # contrast belongs to the SPECTRAL fview gate (OUTPOST_V4SG), which is
    # disabled in every reported arm. If every entry is null the honest reading
    # is that the gate does nothing for accuracy either.
    GATE_OFF = {"photo": ("E400_outpost", "E400_nogate"), "computers": ("E400_outpost", "E400_nogate"),
                "cs": ("E400_outpost", "E400_nogate"), "yelp": ("A_main", "C_nogate"),
                "amazon": (E400.get("amazon", ("A_sim0.0",))[0], "C_nogate"),
                "tfinance": (E400.get("tfinance", ("A_sim0.0",))[0], "C_nogate"),
                "ogbn-arxiv": ("A_main", "A_nogate400")}
    gate = []
    for ds, label in DS:
        if ds not in GATE_OFF:
            continue
        on, off = GATE_OFF[ds]
        r = {"dataset": label, "on_tag": on, "off_tag": off}
        for m, lab in M:
            r[lab] = paired2(ds, "outpost", off, "outpost", on, m)   # off minus on
        if r["oracle ROC"] is not None:
            gate.append(r)
    out["gate_off"] = gate

    # --- 8. parameter counts -------------------------------------------------
    if os.path.exists("analysis/tables/table_complexity.csv"):
        out["parameter_counts"] = pd.read_csv(
            "analysis/tables/table_complexity.csv").to_dict("records")

    # 8b. the efficiency claim across every dataset and all three methods, taken
    # straight from the shards (each records n_params). table_complexity.csv only
    # ever covered Photo and Yelp, so "OUTPOST is a fraction of DEMO" rested on
    # two graphs while the main table reported eight.
    # load_shards() keeps only the metric columns, so read n_params from the raw
    # shard for the one seed we need rather than widening that frame for everyone.
    import glob as _glob

    def _params(ds, method, tag):
        for f in sorted(_glob.glob(
                f"results/rotations/{ds}_{method}_s*_{tag}_rot*.json")):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            for r in j.get("rotations", []):
                if r.get("n_params"):
                    return int(r["n_params"])
        return None

    eff = []
    for ds, label in DS:
        if ds not in E400:
            continue
        tags = {"OUTPOST": E400[ds][0], "DEMO": E400[ds][1],
                "NSREG": NS_MAIN.get(ds)}
        got = {}
        for meth_lab, tag in tags.items():
            if not tag:
                continue
            n = _params(ds, meth_lab.lower(), tag)
            if n:
                got[meth_lab] = n
        if "OUTPOST" not in got:
            continue
        demo = got.get("DEMO")
        eff.append({"dataset": label,
                    "outpost_params": got["OUTPOST"],
                    "demo_params": demo,
                    "nsreg_params": got.get("NSREG"),
                    "outpost_over_demo": round(got["OUTPOST"] / demo, 4) if demo else None,
                    "outpost_over_nsreg": (round(got["OUTPOST"] / got["NSREG"], 4)
                                           if got.get("NSREG") else None)})
    out["parameter_counts_all"] = eff

    # --- 8c. the selection re-runs (P32-P34, P40) ----------------------------
    # Amazon's validation-only sweep picked a non-default arm; these are the
    # paired consequences at n=10, plus the robustness check that decides whether
    # the preference behind the pick was real.
    selfin = {}
    hid, dfl = "T_hidden16", E400.get("amazon", ("A_sim0.0",))[0]
    if len(arm(d, "amazon", "outpost", hid)):
        selfin["arm"] = hid
        selfin["default"] = dfl
        selfin["vs_default"] = paired2("amazon", "outpost", hid, "outpost", dfl, "best_auroc")
        selfin["vs_demo_at_selected"] = paired2("amazon", "outpost", hid, "demo",
                                                "B_demo_mix", "best_auroc")
        selfin["vs_demo_at_default"] = paired2("amazon", "outpost", dfl, "demo",
                                               "B_demo_mix", "best_auroc")
        selfin["vs_demo_valsel"] = paired2("amazon", "outpost", hid, "demo",
                                           "B_demo_mix", "valsel_auroc")
        if len(arm(d, "amazon", "outpost", "C_nopl_hidden16")):
            selfin["pl_at_selected"] = paired2("amazon", "outpost", "C_nopl_hidden16",
                                               "outpost", hid, "best_auroc")
        # P40: the same validation comparison at three seeds and at ten
        import glob as _g

        def _val(tag, seeds):
            vals = []
            for sd in seeds:
                fs = _g.glob(f"results/rotations/amazon_outpost_s{sd}_{tag}_rot*.json")
                per = [r["val_selected"]["val_auc"]
                       for f in fs for r in json.load(open(f))["rotations"]]
                if per:
                    vals.append(sum(per) / len(per))
            return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

        SEL3, ALL10 = [0, 1, 42], [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
        v3h, n3 = _val(hid, SEL3); v3d, _ = _val(dfl, SEL3)
        vAh, nA = _val(hid, ALL10); vAd, _ = _val(dfl, ALL10)
        if None not in (v3h, v3d, vAh, vAd):
            selfin["val_margin"] = {
                "seeds_3": round(v3h - v3d, 4), "n3": n3,
                "seeds_10": round(vAh - vAd, 4), "n10": nA,
                "tie_band": 0.002,
                "reverses": (v3h - v3d) > 0.002 and (vAh - vAd) <= 0.002,
            }
    # the selected arm's own main-table cells, so a *_selarm.tex table can be
    # checked against the arm it actually reports instead of against the default
    if selfin.get("arm"):
        g = arm(d, "amazon", "outpost", selfin["arm"])
        if len(g):
            selfin["main_row"] = {lab: msd(g, m) for m, lab in M}
    out["selection_final"] = selfin

    # --- 8e. does the law predict a graph it was not fitted on? -------------
    try:
        out["law_crossval"] = json.load(open("analysis/tables/law_crossval.json"))
    except Exception:
        pass

    # --- 8d. is the tie band above the noise it thresholds? ------------------
    try:
        out["selection_noise"] = json.load(open("analysis/tables/selection_noise.json"))
    except Exception:
        pass

    json.dump(out, open("analysis/appendix.json", "w"), indent=1, default=float)
    print("wrote analysis/appendix.json")
    for k, v in out.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} rows")
        elif isinstance(v, dict):
            print(f"  {k}: {len(v)} keys")
    return out


if __name__ == "__main__":
    main()
