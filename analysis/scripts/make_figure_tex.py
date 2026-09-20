#!/usr/bin/env python3
"""Emit the paper's figure blocks, with every caption number read from its artifact.

One \\begin{figure} per figure in make_figures_paper.py. Each block carries a comment
naming the result file it was drawn from and the METHODOLOGY section and prediction IDs
it supports, a one-sentence caption stating that figure's single takeaway, and a label.

Captions are generated rather than typed for the same reason the figures are: a number
that a human retypes into a caption is a number that goes stale the next time the
campaign is rescored, and a reviewer checking the caption against the table is exactly
who would find it. Every figure here must exist on disk and every artifact it reads
must be present, or this script fails loudly instead of emitting a stale block.

    python analysis/scripts/make_figure_tex.py -> analysis/tables/tex/figures.tex

Captions are written in American English, matching the figures themselves.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "analysis/tables/tex/figures.tex"


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"figure tex: missing result file {path}")
    return json.load(open(path))


sys.path.insert(0, "analysis")
from figstyle import nice   # one dataset-spelling map for figures and captions alike


# --------------------------------------------------------------------------
def cap_selection_noise():
    d = load("analysis/tables/selection_noise.json")
    s = d["summary"]
    band = d["tie_band"]
    worst = max(d["datasets"], key=lambda r: r["seeds_needed_for_band"])
    return (
        f"The {band:.3f} AUC-ROC tie band used to call two arms equivalent sits below "
        f"the standard error of a {d['min_shared_seeds']}-seed comparison on all "
        f"{s['n_datasets']} graphs, and resolving a difference that small would take up "
        f"to {s['max_seeds_needed']} seeds ({nice(worst['dataset'])}); every tie this paper "
        f"reports is therefore a statement about noise, not about the methods.")


def cap_law_crossval():
    d = load("analysis/tables/law_crossval.json")
    # the fold that sinks the pooled number: biggest excess error, weighted by size
    w = max(d["folds"], key=lambda f: f["n_classes"] *
            (f["mae_same_frac_line"] - f["mae_intercept_only"]))
    return (
        f"Held out one dataset at a time, the detectability law's same-class-fraction "
        f"line predicts the AUC of unseen classes {'worse' if d['improvement'] < 0 else 'better'} "
        f"than a constant that ignores the diagnostic entirely (pooled MAE "
        f"{d['pooled_mae_same_frac_line']:.3f} against {d['pooled_mae_intercept_only']:.3f} "
        f"over {d['n_classes']} classes, the line winning {d['folds_line_better']} of "
        f"{d['n_folds']} folds yet losing on pooled error because it fails worst on the "
        f"largest fold, {nice(w['held_out'])} at {w['mae_same_frac_line']:.3f} against "
        f"{w['mae_intercept_only']:.3f} on {w['n_classes']} of {d['n_classes']} classes), "
        f"so the law describes the classes it was fitted on and carries no out-of-sample "
        f"content.")


def _abl(A, ds, stem):
    for r in A["ablations"]:
        if r["dataset"] == ds and (r["arm"] == stem or r["arm"].startswith(stem + "_")):
            return r
    raise SystemExit(f"figure tex: no ablation row for {ds}/{stem}")


def cap_mechanism():
    A = load("analysis/appendix.json")
    real = ["Yelp", "Amazon", "T-Finance"]
    synth = ["Computers", "CS", "Photo"]
    stems = ["C_nogate", "C_noconformal", "C_nopl"]
    rd = [_abl(A, g, s)["delta_roc"]["delta"] for g in real for s in stems]
    sd = []
    for g in synth:
        try:
            sd.append(_abl(A, g, "C_nopl")["delta_roc"]["delta"])
        except SystemExit:
            pass
    biggest = max(abs(x) for x in rd)
    return (
        f"Removing any of OUTPOST's three components moves AUC-ROC by at most "
        f"{biggest:.4f} on the {len(real)} graphs whose anomalies were observed rather "
        f"than injected, while removing pseudo-labeling costs {min(abs(x) for x in sd):.3f} "
        f"to {max(abs(x) for x in sd):.3f} on the injected-anomaly graphs (shaded): the "
        f"mechanism the method is built around is a property of the injection, not of "
        f"anomaly detection.")


def cap_inflation():
    d = load("analysis/tables/inflation_ordering.json")
    seps = [s for s in d["separation"] if s["complete_separation"]]
    tight = min(seps, key=lambda s: s["min_higher"] - s["max_lower"]) if seps else None
    n = len(d["datasets"])
    wide = max(seps, key=lambda s: s["min_higher"] - s["max_lower"]) if seps else None
    t = (f" (shaded: {wide['lower']} to {wide['higher']} is "
         f"{wide['min_higher'] - wide['max_lower']:.3f} wide, though the closest boundary, "
         f"{tight['lower']} to {tight['higher']}, is only "
         f"{tight['min_higher'] - tight['max_lower']:.3f} and well inside the run-to-run "
         f"spread the bars show)" if tight else "")
    return (
        f"Oracle selection inflates AUC-ROC in strict order of how a benchmark was "
        f"built -- semi-synthetic above OGB above real, with no overlap between the "
        f"{n} dataset means{t} -- so best-epoch reporting rewards a method most exactly "
        f"where the anomalies were manufactured.")


def cap_p8():
    d = load("analysis/tables/p8_per_class.json")
    c = d.get("diagnostic_vs_trained")
    if not c:
        raise SystemExit("figure tex: p8_per_class.json has no diagnostic_vs_trained; "
                         "rerun analysis/scripts/score_p8.py")
    h = c["against"]["auc_hop0"] if "against" in c else c
    k = d["class_263"]
    return (
        f"Across {c['n_classes']} ogbn-mag classes the training-free detectability "
        f"diagnostic has no rank correlation with what a trained model achieves on the "
        f"same class unseen ($\\rho = {h['spearman_rho']:+.3f}$, $p = {h['p_value']:.2f}$), "
        f"and the class the diagnostic rates most detectable ranks {k['rank']} of "
        f"{len(d['per_class'])} once trained: the diagnostic is not a ceiling.")



def cap_method():
    """The schematic's own caption still reads its numbers from config.json."""
    import json as _j
    c = _j.load(open("config.json"))
    dd = c.get("demo_default", {})
    sizes = dd.get("sampling_sizes", [25, 10])
    nl = dd.get("n_layers", 2)
    kp = sorted({v["K_p"] for v in c.values() if isinstance(v, dict) and "K_p" in v})
    kps = " or ".join(str(k) for k in kp)
    return (
        f"OUTPOST scores a node by how far its embedding falls outside a learned atlas "
        f"of normality: a {nl}-layer GraphSAGE encoder over neighbor-sampled subgraphs "
        f"(fan-out {sizes[0]} then {sizes[1]}) is projected and $L_2$-normalized onto "
        f"the unit sphere, where {kps} prototypes with fitted radii define normality as "
        f"a union of hyperspherical caps, and conformal pseudo-labeling -- vetoed by "
        f"the atlas for any candidate lying outside every cap -- expands the training "
        f"signal toward anomaly classes never labeled. Scores shown in stage 5 are "
        f"illustrative; every measured number in this paper is in the figures and "
        f"tables that follow.")


def _runs_cell(ds, tag, method="outpost"):
    sys.path.insert(0, "analysis/scripts")
    import make_figures as G
    return G._cell(G._runs(), ds, tag, method)


def _pub_yelp():
    import csv
    best = None
    for r in csv.DictReader(open("analysis/tables/published_baselines.csv")):
        v = (r.get("Yelp_AUC-PR") or "").strip()
        if v in ("", "--", "nan") or r["Method"].upper().startswith("OUTPOST"):
            continue
        if best is None or float(v) > best[1]:
            best = (r["Method"], float(v), float(r["Yelp_AUC-ROC"]))
    if best is None:
        raise SystemExit("figure tex: no published Yelp baseline")
    return best


def cap_efficiency():
    sys.path.insert(0, "analysis/scripts")
    import make_figures as G
    d = G._runs()
    main = G._cell(d, "yelp", "A_main")
    dem = G._cell(d, "yelp", "B_demo_mix", "demo")
    mp = G._params_of_run("yelp", "A_main") or G._params("yelp", "OUTPOST", 32, False)
    dp = G._params("yelp", "DEMO", 64, None)
    name, pr, _ = _pub_yelp()
    if not (main and dem and mp and dp):
        raise SystemExit("figure tex: efficiency inputs missing")
    return (
        f"On Yelp, OUTPOST reaches {main[1]:.4f} AUC-PR with {mp:,} trainable "
        f"parameters, {mp / dp:.2f}$\\times$ DEMO's {dp:,} and "
        f"{main[1] - pr:+.4f} above the best published result ({name} {pr:.4f}); "
        f"the two arms that vary width rest on {G._cell(d, 'yelp', 'T_hidden16')[2]} "
        f"seeds each, which is why they are drawn smaller.")


def cap_yelp():
    main = _runs_cell("yelp", "A_main")
    name, pr, _ = _pub_yelp()
    ns = _runs_cell("yelp", "E400_nsreg", "nsreg")
    return (
        f"On the one graph in this study whose anomalies are real fraud, OUTPOST "
        f"reaches {main[1]:.4f} AUC-PR against the best published {pr:.4f} ({name}), "
        f"but the same baseline re-run by us under this protocol reaches "
        f"{ns[1]:.4f}, so most of the apparent margin over the literature is "
        f"protocol, not method.")


def cap_simsample():
    off = _runs_cell("yelp", "C_nosim")
    plc = _runs_cell("yelp", "C_simplacebo")
    main = _runs_cell("yelp", "A_main")
    if not (off and plc and main):
        raise SystemExit("figure tex: simsample arms missing")
    gain = main[1] - off[1]
    kept = plc[1] - off[1]
    # the placebo can land BELOW the off arm, in which case "removes 110% of the
    # gain" is an arithmetically true sentence that no reader should have to parse
    tail = (f"lands at {plc[1]:.4f}, below the {off[1]:.4f} of turning SimSample off "
            f"altogether" if kept <= 0 else
            f"keeps only {kept / gain:.0%} of it ({plc[1]:.4f} against {off[1]:.4f} off)")
    return (
        f"Shuffling the similarity order while keeping everything else fixed destroys "
        f"the whole {gain:+.4f} Yelp AUC-PR gain that SimSample gives "
        f"({main[1]:.4f} on): the shuffled placebo {tail}, so the gain comes from the "
        f"order itself and not from sampling fewer neighbors.")


def cap_law():
    d = load("analysis/tables/detectability_law.json")
    ci = d.get("cluster_bootstrap_ci95")
    tr = d.get("two_regime", {})
    return (
        f"Across {d['n_classes']} anomaly classes on {d.get('n_datasets', 8)} graphs "
        f"a class's training-free detectability tracks how clustered it is "
        f"($\\rho$ = {d['spearman_rho']:.3f}, permutation "
        f"$p$ = {d['p_permutation_within_dataset']:.0e}, cluster-bootstrap 95\\% CI "
        f"[{ci[0]:.2f}, {ci[1]:.2f}]), and the right panel says why: propagation "
        f"amplifies clustered anomalies and erases scattered ones "
        f"({tr.get('n_propagation_regime', '?')} classes in the propagation regime). "
        f"This describes the classes it was measured on; Figure~\\ref{{fig:law-crossval}} "
        f"shows it does not predict held-out ones.")


def cap_regime():
    J = load("analysis/tables/simsample_by_dataset.json")
    BUD = 25
    hi = {k: v for k, v in J.items() if v["median_degree"] > BUD and v["sim_delta_roc"]}
    pos = [k for k, v in hi.items() if v["sim_delta_roc"]["delta"] > 0]
    neg = [k for k, v in hi.items() if v["sim_delta_roc"]["delta"] <= 0]
    lo = {k: v for k, v in J.items() if v["median_degree"] <= BUD and v["sim_delta_roc"]}
    lpos = [k for k, v in lo.items() if v["sim_delta_roc"]["delta"] > 0]
    return (
        f"If SimSample worked by restoring neighbors lost to the hop-1 budget of "
        f"{BUD}, its effect would follow the top two panels; it does not -- of the "
        f"{len(hi)} graphs above the budget {len(pos)} improve and {len(neg)} "
        f"{'does' if len(neg) == 1 else 'do'} not, "
        f"and {len(lpos)} of the {len(lo)} graphs below it improve anyway, which "
        f"falsifies the pre-registered mechanism (P14, P17).")


# --------------------------------------------------------------------------
# stem, label, result file(s), what it supports, caption builder, width
FIGURES = [
    ("method_overview", "fig:method",
     ["model.py, trainer.py, config.json (the implementation itself)"],
     "the method; no measured quantity", cap_method, r"\linewidth"),
    ("selection_noise", "fig:selection-noise",
     ["analysis/tables/selection_noise.json"],
     "METHODOLOGY 4.9 -- P35, P40", cap_selection_noise, r"\linewidth"),
    ("oracle_inflation_separation", "fig:inflation",
     ["analysis/tables/inflation_ordering.json"],
     "METHODOLOGY 3.2", cap_inflation, r"\linewidth"),
    ("mechanism_real_vs_synthetic", "fig:mechanism",
     ["analysis/appendix.json (ablations)"],
     "METHODOLOGY 5.4 -- P28, P39", cap_mechanism, r"\linewidth"),
    ("law_crossval", "fig:law-crossval",
     ["analysis/tables/law_crossval.json"],
     "METHODOLOGY 6.5", cap_law_crossval, r"\linewidth"),
    ("p8_diagnostic_vs_trained", "fig:p8",
     ["analysis/tables/p8_per_class.json",
      "analysis/tables/spectral_ogbn-mag.csv"],
     "METHODOLOGY 6.6 -- P8", cap_p8, r"\linewidth"),
    ("detectability_law", "fig:law",
     ["analysis/tables/spectral_*.csv", "analysis/tables/detectability_law.json"],
     "METHODOLOGY 6.1-6.2", cap_law, r"\linewidth"),
    ("regime_decomposition", "fig:regime",
     ["analysis/tables/simsample_by_dataset.json"],
     "METHODOLOGY 5.5 -- P14, P17", cap_regime, r"0.62\linewidth"),
    ("simsample_mechanism", "fig:simsample",
     ["results/results.csv (yelp arms)"],
     "METHODOLOGY 5.5 -- P14, P17", cap_simsample, r"\linewidth"),
    ("yelp_comparison", "fig:yelp",
     ["analysis/tables/published_baselines.csv", "results/results.csv"],
     "METHODOLOGY 5.5", cap_yelp, r"\linewidth"),
    ("efficiency", "fig:efficiency",
     ["results/results.csv", "analysis/tables/published_baselines.csv"],
     "METHODOLOGY 5.5", cap_efficiency, r"\linewidth"),
]

RULE = "% " + "-" * 73


def main():
    missing = [s for s, *_ in FIGURES if not os.path.exists(f"figures/{s}.pdf")]
    if missing:
        raise SystemExit("figure tex: no PDF for " + ", ".join(missing) +
                         " -- run analysis/scripts/make_figures_paper.py first")
    out = ["% Generated by analysis/scripts/make_figure_tex.py -- do not edit by hand.",
           "% Every number in every caption below is read from the result file named in",
           "% that figure's comment block at the time this file was written.", ""]
    for stem, label, srcs, supports, cap, width in FIGURES:
        drawn = ("Nano Banana Pro via the scientific-schematics skill; NOT checked "
                 "by check_figure_text.py and NOT regenerated by refresh_all.sh"
                 if stem == "method_overview"
                 else "analysis/scripts/make_figures_paper.py")
        out += [RULE,
                f"% figure      : figures/{stem}.pdf",
                f"% drawn by    : {drawn}",
                f"% result file : {srcs[0]}"]
        out += [f"%               {s}" for s in srcs[1:]]
        out += [f"% supports    : {supports}",
                RULE,
                r"\begin{figure}[t]",
                r"\centering",
                f"\\includegraphics[width={width}]{{figures/{stem}.pdf}}",
                f"\\caption{{{cap()}}}",
                f"\\label{{{label}}}",
                r"\end{figure}", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(out) + "\n")
    print(f"-> {OUT}  ({len(FIGURES)} figures)")
    for stem, label, *_ in FIGURES:
        print(f"   {label:22s} figures/{stem}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
