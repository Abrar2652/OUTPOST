"""The submission-audit page, regenerated from the run record and the generated
artifacts. Was built inline on 2026-09-06; now a script so it can be re-run
after every campaign.

    python analysis/scripts/build_audit_page.py [out.html]
"""
import json, os, re, html, subprocess, time, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = sys.argv[1] if len(sys.argv) > 1 else "analysis/submission_audit.html"
A = json.load(open("analysis/appendix.json"))
e = html.escape
def mt(p): return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.stat(p).st_mtime)) if os.path.exists(p) else "—"
rc = os.stat("results/results.csv").st_mtime
sh = lambda pat: len([f for f in os.listdir("results/rotations") if re.search(pat, f)])

# --- experiment programme, with live counts ---------------------------------
def n_of(pat, total): n = sh(pat); return n, total, ("complete" if n >= total else ("RUNNING" if n > 0 else "queued"))
prog = [
 ("Main protocol, uniform 400 epochs, n=10 (photo/computers/cs/yelp/amazon), n=5 (arxiv, mag)", ("complete", ""), "main table, paired tests, oracle-vs-valsel"),
 ("DEMO full method, n=10 on the five paired datasets", ("complete", ""), "paired OUTPOST−DEMO; reproduction gap by budget"),
 ("NSReg (released code) @400 n=10 + 201-epoch arm n=5", ("complete", ""), "three-method ranking; NSReg reproduction"),
 ("Amazon-Fraud: both SimSample arms + DEMO + C_nopl", ("complete", ""), "inflation at n=2 real; P23–P25"),
 ("Mechanism decomposition (gate/PL/conformal/gate+PL) on Computers, CS, Photo, Yelp", ("complete", ""), "§5.2; P20–P22"),
 ("ogbn-mag, 5 seeds, full budget", ("complete", ""), "P5 split; regime result"),
 ("Epoch budget 200 vs 400", ("complete", ""), "epoch_budget; protocol decision"),
 ("Gate-drop: main protocol with the gate removed", ("complete", ""), "gate null on 5 datasets → gate-off tables (*_gateoff.tex)"),
]
live = [
 ("Decomposition fill: Photo conformal; Amazon gate/conformal/gate+PL/placebo", n_of(r"(photo_outpost_s\d+_C_noconformal_|amazon_outpost_s\d+_(C_nogate|C_noconformal|L_lean|C_simplacebo)_)", 50), "completes the decomposition on all five datasets"),
 ("ogbn-arxiv → n=10", n_of(r"ogbn-arxiv_outpost_s[4-8]_A_main_", 20), "thin vs-published tests"),
 ("Hyperparameters at 400 epochs (Photo, Computers) + Amazon's first sweep", n_of(r"(E400_T_|amazon_outpost_s\d+_T_)", 243), "selection re-done at the reported budget (§10.1-1)"),
 ("NSReg tuning grid, validation-only", n_of(r"_NT_lr", 81), "NSReg not tied to one released config (§10.1-5)"),
 ("T-Finance: eighth graph, third real one — OUTPOST both arms, DEMO, NSReg, C_nopl", n_of(r"^tfinance_", 60), "P26–P28; law consistency; PL on a third real graph"),
 ("ogbn-mag seeds 4–8 with per-node scores", n_of(r"ogbn-mag_outpost_s[4-8]_A_main_", 75), "mag n=10; P8 scorable"),
 ("Selection re-runs: Amazon at its validation-selected config (T_hidden16) and NSReg at its own, to n=10",
  # only the seeds this campaign runs: T_hidden16 already has 0/1/42 from the sweep,
  # and counting those as campaign progress would report work that has not happened
  n_of(r"(amazon_outpost_s[2-8]_T_hidden16_|amazon_outpost_s\d+_C_nopl_hidden16_|(photo|amazon)_nsreg_s[2-8]_NT_lr0\.003_wd0\.0_)", 38),
  "P32–P35; Amazon's main-table row moves after P31's retention clause was falsified"),
 ("DEMO tuning grid, matched to NSReg's, validation-only",
  n_of(r"_demo_s\d+_DT_lr", 81), "P36/P37 (blind); closes the last fairness asymmetry"),
 ("T-Finance decomposition: gate and conformal removed",
  n_of(r"tfinance_outpost_s\d+_(C_nogate|C_noconformal)_", 20), "P38/P39; gate on a seventh graph"),
]
rows_prog = "".join(f"<tr><td>{e(a)}</td><td><span class='tag ok'>complete</span></td><td>{e(c)}</td></tr>" for a, _, c in prog)
for a, (n, tot, st), c in live:
    cls = {"complete": "ok", "RUNNING": "warn", "queued": "warn"}[st]
    rows_prog += f"<tr><td>{e(a)}</td><td><span class='tag {cls}'>{st}</span> <span class='sub'>{n}/{tot}</span></td><td>{e(c)}</td></tr>"

# --- artifacts ----------------------------------------------------------------
# The flip count is the paper's headline; deriving it from the generated ranking
# note is the only way it cannot go stale. It moved from 3-of-5 to 2-of-6 on
# 2026-09-08 when NSReg's Amazon row went to its (surviving) selected arm.
def _rank_flip():
    try:
        t = open("analysis/tables/three_method_ranking.md", encoding="utf-8").read()
        m = re.search(r"Ranking changes on \*\*(\d+) of (\d+)\*\* datasets", t)
        if m:
            return f"Rankings flip on {m.group(1)}/{m.group(2)} datasets", f"{m.group(1)}-of-{m.group(2)}"
    except Exception:
        pass
    return "Rankings flip on some datasets (see the note)", "flip"


RANK_FLIP, RANK_FLIP_SHORT = _rank_flip()

art = [
 ("analysis/tables/table1_small.tex", "Table 1 (small graphs, oracle)", "make_paper_tables.py", "Main comparison incl. DEMO and NSReg re-runs", "§Results Table 1"),
 ("analysis/tables/table1_small_gateoff.tex", "Table 1, OUTPOST without the gate", "make_paper_tables.py --gate-off", "Same model with the atlas gate removed; the gate is non-parametric so the size is identical (0.157× DEMO on Photo either way) and only accuracy differs", "§Results Table 1 (if the paper describes the gate-off variant)"),
 ("analysis/tables/table1_small_valsel.tex", "Table 1 (val-selected)", "make_paper_tables.py --metric valsel", "Deployable selection", "beside Table 1 / appendix"),
 ("analysis/tables/table2_large.tex", "Table 2 (Yelp/Amazon/T-Finance/arxiv/mag)", "make_paper_tables.py", "Real + OGB graphs; mag regime result", "§Results Table 2"),
 ("analysis/tables/table2_large_gateoff.tex", "Table 2 without the gate", "make_paper_tables.py --gate-off", "as above", "§Results Table 2 (variant)"),
 ("analysis/tables/table2_large_valsel.tex", "Table 2 (val-selected)", "make_paper_tables.py --metric valsel", "Deployable counterpart", "appendix"),
 ("analysis/tables/tex/three_method_ranking.tex", "Three methods × two rules", "make_appendix_tex.py", RANK_FLIP, "§Results — the thesis table"),
 ("analysis/tables/tex/paired_vs_demo.tex", "Paired OUTPOST−DEMO", "make_appendix_tex.py", "head-to-head, both rules", "§Results / appendix"),
 ("analysis/tables/tex/paired_vs_nsreg.tex", "Paired OUTPOST−NSReg", "make_appendix_tex.py", "head-to-head, both rules", "§Results / appendix"),
 ("analysis/tables/tex/oracle_inflation.tex", "Oracle inflation, all + unseen", "make_appendix_tex.py", "semi-synthetic ≫ OGB ≫ real; unseen ~2×", "§Protocol"),
 ("analysis/tables/tex/decomposition.tex", "Component decomposition", "make_appendix_tex.py", "PL the only load-bearing component", "§5"),
 ("analysis/tables/tex/gate_off.tex", "Gate removed at the protocol", "make_appendix_tex.py", "null on every dataset", "§5 / efficiency"),
 ("analysis/tables/tex/epoch_budget.tex", "200 vs 400 epochs", "make_appendix_tex.py", "Photo tie→loss; budget decides verdicts", "§Protocol"),
 ("analysis/tables/tex/demo_gap.tex", "DEMO gap by budget", "make_appendix_tex.py", "77% of Photo gap is budget", "§Protocol"),
 ("analysis/tables/tex/nsreg_reproduction.tex", "NSReg vs its published numbers", "make_appendix_tex.py", "+0.04–0.10 above its paper", "§Protocol"),
 ("analysis/tables/tex/unseen_paired.tex", "Unseen-class paired", "make_appendix_tex.py", "the open-set metric proper", "§Results / appendix"),
 ("analysis/tables/three_method_ranking.md", "Ranking note (markdown)", "render_ranking_note.py", "same as the tex", "—"),
 ("analysis/tables/prereg_ledger.md", "Pre-registration ledger P1–P28", "prereg_ledger.py", "every prediction with a verdict", "Appendix"),
 ("analysis/tables/detectability_law.json", "Law, 7(→8) graphs, two-regime", "detectability_law.py", "ρ pooled vs propagation-regime", "§6"),
 ("analysis/appendix.html", "Every table behind the claims", "build_appendix.py / render_appendix.py", "single source", "Appendix (published)"),
 ("figures/detectability_law.png", "Law figure", "make_figures.py", "§6 figure", "§6"),
 ("figures/yelp_comparison.png", "Yelp field + re-runs", "make_figures.py", "real-fraud comparison", "§Results"),
 ("figures/simsample_mechanism.png", "Dose-response + placebo", "make_figures.py", "SimSample on Yelp", "§5"),
 ("figures/efficiency.png", "AUC-PR vs parameters", "make_figures.py", "0.34× DEMO above field (Yelp)", "§Efficiency"),
 ("figures/regime_decomposition.png", "SimSample vs degree, 6(→7) graphs", "make_figures.py", "P14/P17 falsification", "§5.1"),
]
art += [
 ("analysis/tables/table2_large_selarm.tex", "Table 2 with OUTPOST at the validation-selected arm",
  "make_paper_tables.py --selected-arm",
  "Only Amazon differs (T_hidden16, 0.0040 lower); generated because the pre-registered rule selected it, reported beside the default-arm table because P40 shows that selection reverses at ten seeds",
  "\u00a7Results Table 2 (framing decision: this or the default-arm table)"),
 ("analysis/tables/table2_large_valsel_selarm.tex", "Same, deployable selection rule",
  "make_paper_tables.py --metric valsel --selected-arm",
  "Deployable counterpart of the above",
  "appendix"),

 ("analysis/tables/tex/hparam_selection.tex", "Validation-only selection vs the best arm on test",
  "make_appendix_tex.py",
  "Validation saturates on the semi-synthetic graphs; on Amazon it discriminates and picks an arm 0.0071 worse on test",
  "\u00a7Protocol"),
 ("analysis/tables/tex/baseline_selection.tex", "The same selection rule applied to NSReg",
  "make_appendix_tex.py",
  "The baseline swept under our own rule gets better on Photo and Amazon; answers the 'you only tuned your own method' objection",
  "\u00a7Protocol or appendix"),
 ("analysis/tables/TABLE_STORIES.md", "What each table licenses, and what it does not",
  "table_stories.py",
  "Drafting aid: one sentence per table, plus the sentence a reader will wrongly write from it",
  "not published; for the authors"),
]

# Anything in analysis/tables/tex/ that the curated list above does not mention is
# a table the paper could compile and this audit would not have shown. Enumerate
# the directory so a new table can never be silently absent - the failure this
# page exists to catch. Same for the generated markdown companions.
_listed = {a[0] for a in art}
for _extra in sorted(glob.glob("analysis/tables/tex/*.tex")
                     + ["analysis/tables/TABLE_STORIES.md"]):
    if _extra in _listed or not os.path.exists(_extra):
        continue
    _stem = os.path.basename(_extra).rsplit(".", 1)[0]
    art.append((_extra, _stem.replace("_", " "),
                "table_stories.py" if _extra.endswith(".md") else "make_appendix_tex.py",
                "NOT YET DESCRIBED in build_audit_page.py - add a row for it",
                "appendix unless stated otherwise"))

rows_art = ""
for path, name, gen, claim, place in art:
    stale = os.path.exists(path) and os.stat(path).st_mtime < rc
    st = '<span class="tag bad">older than results.csv</span>' if stale else ('<span class="tag ok">current</span>' if os.path.exists(path) else '<span class="tag warn">not yet</span>')
    rows_art += f"<tr><td><code>{e(path)}</code><br><span class='sub'>{e(name)}</span></td><td>{e(gen)}</td><td>{mt(path)}<br>{st}</td><td>{e(claim)}</td><td>{e(place)}</td></tr>"
legacy = sorted(os.listdir("analysis/tables/legacy")) if os.path.isdir("analysis/tables/legacy") else []
led = open("analysis/tables/prereg_ledger.md", encoding="utf-8").read()
ledrows = ""
for line in led.splitlines():
    if line.startswith("| P"):
        c = [x.strip() for x in line.strip("|").split("|")]
        v = c[3]; cls = "ok" if v.upper().startswith("CONFIRMED") else ("bad" if ("FALSIFIED" in v.upper() or "CORRECTED" in v.upper()) else "warn")
        ledrows += f"<tr><td class='n'>{e(c[0])}</td><td>{e(c[2])}</td><td><span class='tag {cls}'>{e(v[:70])}</span></td></tr>"
audit = subprocess.run(["python3", "analysis/scripts/audit_claims.py"], capture_output=True, text=True).stdout
m1 = re.search(r"(\d+)/(\d+) numeric claims", audit); stale_n = audit.count("*** STALE"); nf = len(re.findall(r"^\s+L\d+\s", audit, re.M))
concerns = [
 ("Hyperparameters selected at 200 epochs, reused at 400", "hparam400: Photo 13/13 and Computers 10/10 arms inside the tie band at 400, default retained (P29)", "answered"),
 ("NSReg ran one released configuration", "nsreg_tune: same validation-only rule as OUTPOST; changes the winner on Photo and Amazon and the baseline gets better (nsreg_selection.md)", "answered"),
 ("You swept your own method but not the baselines", "NSReg swept; DEMO's matched grid registered blind (P36/P37) and running", "in progress"),
 ("Amazon's configuration was inherited from Yelp, never selected", "Amazon's first sweep ran; validation picked a NON-default arm that is 0.0071 worse on test, so the main-table row moves to it (P31 falsified, P32–P33 registered)", "in progress"),
 ("arxiv/mag at n=5; NSReg infeasible on OGB", "arxiv → n=10 and mag → n=10 queued; NSReg-on-OGB stays a stated limitation", "in progress / limitation"),
 ("DEMO's Computers reproduction gap unexplained", "report both rows; cannot be closed by us", "—"),
 ("CS was excluded from both baseline tuning grids", "cost: 77 min per rotation x 8 rotations x 4 configurations for NSReg, similar for DEMO. Stated wherever a tuned number appears; CS is also the dataset where OUTPOST's oracle and deployable rankings agree, so it is not the one a tuning artefact would rescue", "stated limitation"),
 ("OUTPOST has no hyperparameter sweep on CS at 400 epochs or on T-Finance", "P29 showed validation is saturated at 400 on both graphs that were swept (13/13 and 10/10 in band), and CS's 200-epoch sweep retained the default. On T-Finance all three methods run untuned, so the comparison there is symmetric", "stated limitation"),
 ("NSReg and DEMO are absent on ogbn-arxiv and ogbn-mag", "CORRECTED 2026-09-14: the stated reason was wrong. NSReg runs on arxiv in 2.17 GB peak (its CS memory problem came from 6,805-dim features; OGB has 128). DEMO runs on arxiv against a 114.7 GB PPR built in 8 minutes. Only DEMO-on-mag is a real wall (2.17 TB). The cells are empty for COST: DEMO-arxiv complete; NSReg-arxiv ~21 GPU-h (an earlier ~424 estimate extrapolated from a startup-dominated smoke test).", "in progress / cost-bound"),
 ("Two-regime law is post-hoc", "T-Finance registered (P26–P28); it is a consistency test, not a discriminating one", "in progress"),
 ("P8 unscorable (no per-class scores)", "mag seeds 4–8 run with --save-scores", "in progress"),
 ("Real fraud is not uniformly scattered: T-Finance same-class fraction 0.629 (§6.1 wording)", "METHODOLOGY \u00a76.1 restated 2026-09-07: the two real graphs the field reports are scattered; real fraud graphs are not uniformly so", "answered"),
 ("Pseudo-labelling might just fail on scattered graphs, not on real ones", "T-Finance is homophilous (0.629) and PL is still significantly harmful there (+0.0097, 9/10, p=0.0039), so the divider is label provenance, not neighbourhood (P28)", "answered"),
 ("Bold row marks 'ours' regardless of winning", "make_paper_tables.py now bolds the best value in each column among the three rows run under one protocol (DEMO, NSReg, OUTPOST); OUTPOST is unbolded where a baseline wins, e.g. T-Finance and Photo ROC. Caption states which rows are comparable.", "answered"),
]
rows_con = "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td>{e(c)}</td></tr>" for a, b, c in concerns)
draft = [
 "Every number traces to analysis/appendix.json or stats.json; run audit_claims.py after the final edit and read the STALE list, not the match count",
 "Both selection rules for every head-to-head; Photo and Amazon described as rule-dependent, not wins",
 "Tables 1/2 carry DEMO (re-run) AND NSReg (re-run) rows beside transcribed ones, caption saying which are comparable",
 "Decide which model the paper describes — gate-on (table1_small.tex) or gate-off (table1_small_gateoff.tex) — and use one consistently",
 f"Three-method ranking table in the main text; its {RANK_FLIP_SHORT} flip is the stated thesis",
 "NSReg above its published numbers and DEMO below both stated, with the budget analysis",
 "Law in two-regime form; pooled ρ and propagation-regime ρ both cited; two-regime called post-hoc; T-Finance as consistency test",
 "PL the only load-bearing component; essential on semi-synthetic, null-to-harmful on real; gate never described as contributing",
 "Amazon (and T-Finance when landed) in Table 2 and the inflation ordering; unseen-class inflation figures shown",
 "§6.1 restated: real fraud is not uniformly scattered (T-Finance 0.629)",
 "Every cited prediction carries its ledger verdict, falsifications included",
 "Nothing cited from analysis/tables/legacy/ or the old snapshots",
]
dl = "".join(f"<li>{e(x)}</li>" for x in draft)
stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
CSS = open("analysis/appendix.html").read().split("<style>")[1].split("</style>")[0]
extra = """
.tag{font-size:10px;text-transform:uppercase;letter-spacing:.09em;font-weight:600;padding:3px 7px;border-radius:2px;white-space:nowrap}
.tag.ok{color:var(--win);background:color-mix(in srgb,var(--win) 13%,transparent)}
.tag.bad{color:var(--loss);background:color-mix(in srgb,var(--loss) 13%,transparent)}
.tag.warn{color:var(--floor);background:color-mix(in srgb,var(--floor) 15%,transparent)}
td{white-space:normal;text-align:left;vertical-align:top;font-size:12.5px} th{text-align:left}
.sub{color:var(--ink3);font-size:11.5px}
.verd{border-left:3px solid var(--accent);padding:10px 14px;background:var(--card);margin:0 0 14px;max-width:80ch} .verd b{color:var(--ink)}
ol li{margin:0 0 6px;max-width:80ch}
"""
running = [a for a, (n, t, st), c in live if st != "complete"]
H = f"""<title>OUTPOST Submission Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@400;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}{extra}</style>
<div class="wrap"><header class="top"><h1>OUTPOST Submission Audit</h1>
<p class="sub-title">Are the experiments complete, are the tables and figures the ones the paper should compile from, and what can a reviewer still raise. Regenerated {e(stamp)}.</p>
<div class="meta"><span class="chip">run record <b>{time.strftime('%Y-%m-%d %H:%M', time.localtime(rc))}</b></span>
<span class="chip">audit <b>{m1.group(1) if m1 else '?'}/{m1.group(2) if m1 else '?'}</b> match, <b>{stale_n}</b> stale, <b>{nf}</b> not found</span>
<span class="chip{' live' if running else ''}">{len(running)} campaign(s) still to land</span></div></header>
<nav class="nav"><a href="#verdict">Verdict</a><a href="#exp">Experiments</a><a href="#art">Artifacts</a><a href="#legacy">Quarantined</a><a href="#prereg">Pre-registration</a><a href="#concerns">Reviewer concerns</a><a href="#draft">Draft checklist</a></nav>
<section id="verdict"><p class="eyebrow">verdict</p><h2>Three answers</h2>
<div class="verd"><b>Are all experiments complete?</b> The eight planned campaigns are complete, including gate-drop (the gate is null on every dataset at the paper's protocol). The user's directive of 2026-09-06 — run everything possible — added six more, listed below with live counts; they run unattended in value order, each followed by a refresh of every table.</div>
<div class="verd"><b>Are the tables and figures in a submittable state?</b> Every LaTeX table (four main, two gate-off variants, ten appendix), every figure and every generated markdown is regenerated from <code>results.csv</code> or the shards by <code>refresh_all.sh</code>, and each is listed with its generator, regeneration time and the claim it carries. Seven runlog-era tables that looked current were quarantined to <code>analysis/tables/legacy/</code> on 2026-09-06.</div>
<div class="verd"><b>Are they discussed at the correct places in the draft?</b> Not auditable from this repository: no OUTPOST manuscript exists here. The checklist at the bottom is what to hold the draft against; <code>audit_claims.py</code> can be pointed at it once it is in the repository.</div></section>
<section id="exp"><p class="eyebrow">experiments</p><h2>Programme status</h2><div class="tw"><table><thead><tr><th>Campaign</th><th>Status</th><th>Feeds</th></tr></thead><tbody>{rows_prog}</tbody></table></div></section>
<section id="art"><p class="eyebrow">artifacts</p><h2>What the paper compiles from</h2><div class="tw"><table><thead><tr><th>Artifact</th><th>Generator</th><th>Regenerated</th><th>Claim it carries</th><th>Belongs in</th></tr></thead><tbody>{rows_art}</tbody></table></div></section>
<section id="legacy"><p class="eyebrow">quarantined</p><h2>Do not cite: <code>analysis/tables/legacy/</code></h2><p class="sub">{e(', '.join(legacy))}</p></section>
<section id="prereg"><p class="eyebrow">pre-registration</p><h2>Ledger</h2><div class="tw"><table><thead><tr><th>P</th><th>Prediction</th><th>Verdict</th></tr></thead><tbody>{ledrows}</tbody></table></div></section>
<section id="concerns"><p class="eyebrow">residual</p><h2>What a reviewer can still raise</h2><div class="tw"><table><thead><tr><th>Concern</th><th>Addressed by</th><th>State</th></tr></thead><tbody>{rows_con}</tbody></table></div></section>
<section id="draft"><p class="eyebrow">draft</p><h2>Checklist for the manuscript</h2><ol>{dl}</ol></section>
<footer>Regenerate with <code>python analysis/scripts/build_audit_page.py</code> after <code>refresh_all.sh</code>.</footer></div>
"""
open(OUT, "w", encoding="utf-8").write(H); print("wrote", OUT, len(H), "bytes")
