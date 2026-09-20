"""Structural + numeric check of every generated .tex table.

    python analysis/scripts/check_tex_consistency.py        # exit 1 on any problem

Structure (no LaTeX compiler needed): every data row must fill exactly the
tabular's column count (\\multicolumn{k} counts k), braces and $ must balance,
captions that state a dataset count must match the header.
Numbers: the paper tables (from results.csv via make_paper_tables.py) must
agree cell-for-cell with analysis/appendix.json (from the per-rotation shards).
"""
import glob, os, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
A = json.load(open("analysis/appendix.json"))
NUMWORD = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
problems, cells = [], 0


def rows_of(tex):
    """(ncols from the alignment spec, list of (lineno, cell-list)) for the first tabular."""
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return None, []
    spec = re.sub(r"@\{[^}]*\}|\|", "", m.group(1))
    ncols = sum(int(k) if k else 1 for k in re.findall(r"\*\{(\d+)\}\{[^}]*\}", spec)) or len(re.sub(r"p\{[^}]*\}", "l", spec).replace(" ", ""))
    out = []
    for i, line in enumerate(m.group(2).splitlines()):
        t = line.strip()
        if not t or t.startswith(("\\toprule", "\\midrule", "\\bottomrule", "%")):
            continue
        t = re.sub(r"\\\\\s*$", "", t)
        out.append((i, [c.strip() for c in t.split("&")]))
    return ncols, out


def width(cell):
    m = re.match(r"\\multicolumn\{(\d+)\}", cell)
    return int(m.group(1)) if m else 1


def structural(path):
    tex = open(path, encoding="utf-8").read()
    if tex.count("{") != tex.count("}"):
        problems.append(f"{path}: unbalanced braces ({tex.count('{')} vs {tex.count('}')})")
    if tex.count("$") % 2:
        problems.append(f"{path}: odd number of $")
    ncols, rows = rows_of(tex)
    if ncols is None:
        # a figure file legitimately has no tabular; it is checked for escapes,
        # braces and math spans like any other, just not for columns
        if "\\begin{figure}" in tex and "\\begin{table}" not in tex:
            return
        problems.append(f"{path}: no tabular found"); return
    for i, cells_ in rows:
        w = sum(width(c) for c in cells_)
        if w != ncols:
            problems.append(f"{path}: row {i} has {w} columns, tabular declares {ncols}: {' & '.join(cells_)[:90]}")
    cap = re.search(r"\\caption\{(.*?)\}\s*\\label", tex, re.S)
    if cap:
        m = re.search(r"\b(two|three|four|five|six|seven|eight)\b\s+(?:small|large)-scale datasets", cap.group(1))
        if m and rows:
            hdr = rows[0][1]
            n_ds = len({re.sub(r"\s+(ROC|PR)$", "", h) for h in hdr[1:]})
            if NUMWORD[m.group(1)] != n_ds:
                problems.append(f"{path}: caption says {m.group(1)} datasets, header has {n_ds}")


def num(cell):
    """mean, sd (or None) from '\\textbf{0.8703 $\\pm$ 0.0243}' / '0.8703' / '--'."""
    c = re.sub(r"\\textbf\{|\}", "", cell)
    m = re.match(r"\s*([-+]?\d*\.\d+)\s*(?:\$\\pm\$\s*([-+]?\d*\.\d+))?\s*$", c)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else None


def check_main(path, rule, skip_datasets=()):
    """table{1,2}_*.tex rows 'OUTPOST' / 'DEMO (re-run, ours)' / 'NSReg (re-run, ours)' vs appendix main_table.

    `skip_datasets` exists for the *_selarm variants, whose OUTPOST row is at a
    different arm on exactly those datasets and is checked separately.
    """
    global cells
    ncols, rows = rows_of(open(path, encoding="utf-8").read())
    if not rows:
        return
    hdr = rows[0][1]
    meth = {"\\textbf{OUTPOST}": "OUTPOST", "OUTPOST": "OUTPOST", "DEMO (re-run, ours)": "DEMO", "NSReg (re-run, ours)": "NSREG"}
    app = {(r["dataset"].lower(), r["method"].upper()): r for r in A["main_table"]}
    alias = {"ogbn-arxiv": "ogbn-arxiv", "ogbn-mag": "ogbn-mag", "t-finance": "t-finance"}
    for _, r in rows[1:]:
        if r[0] not in meth:
            continue
        for h, cell in zip(hdr[1:], r[1:]):
            ds, met = re.match(r"(.+?)\s+(ROC|PR)$", h).groups()
            ds = alias.get(ds.lower(), ds.lower())
            if ds in skip_datasets and r[0] in ("OUTPOST", "\\textbf{OUTPOST}"):
                continue          # checked against the selected arm instead
            v = num(cell)
            ref = app.get((ds, meth[r[0]]))
            key = f"{rule} {met}"
            if ref is None or ref.get(key) is None:
                if v is not None:
                    problems.append(f"{path}: {r[0]} {h} = {cell} but appendix has no value")
                continue
            if v is None:
                problems.append(f"{path}: {r[0]} {h} is '{cell}' but appendix has {ref[key]}"); continue
            cells += 1
            if abs(v[0] - ref[key]["mean"]) > 5e-5 or (v[1] is not None and abs(v[1] - ref[key]["sd"]) > 5e-5):
                problems.append(f"{path}: {r[0]} {h} tex {cell} vs appendix {ref[key]['mean']:.4f}±{ref[key]['sd']:.4f}")


for f in sorted(glob.glob("analysis/tables/*.tex") + glob.glob("analysis/tables/tex/*.tex")):
    structural(f)
def check_gateoff(path, rule):
    """*_gateoff.tex OUTPOST row = the gate-off arm; reference is gate_off[ds][metric].mean_a (arm a = off)."""
    global cells
    ncols, rows = rows_of(open(path, encoding="utf-8").read())
    if not rows:
        return
    hdr = rows[0][1]
    ref = {r["dataset"].lower(): r for r in A.get("gate_off", [])}
    for _, r in rows[1:]:
        if r[0] not in ("\\textbf{OUTPOST}", "OUTPOST"):
            continue
        for h, cell in zip(hdr[1:], r[1:]):
            ds, met = re.match(r"(.+?)\s+(ROC|PR)$", h).groups()
            v = num(cell); g = ref.get(ds.lower(), {}).get(f"{rule} {met}")
            if g is None or "mean_a" not in g:
                if v is not None and ds.lower() in ref:
                    problems.append(f"{path}: OUTPOST {h} = {cell} but appendix gate_off has no {rule} {met} means")
                continue
            if v is None:
                problems.append(f"{path}: OUTPOST {h} is '{cell}' but appendix gate_off has {g['mean_a']}"); continue
            cells += 1
            if abs(v[0] - g["mean_a"]) > 5e-5 or (v[1] is not None and abs(v[1] - g["sd_a"]) > 5e-5):
                problems.append(f"{path}: OUTPOST {h} tex {cell} vs gate-off arm {g['mean_a']:.4f}±{g['sd_a']:.4f} (n={g['n']})")


def check_selarm(path, rule):
    """*_selarm.tex reports OUTPOST at each dataset's validation-SELECTED arm.

    Only Amazon differs from the default-arm table, so every other dataset is
    checked against main_table exactly as usual and Amazon is checked against the
    selected arm's own cells. Without this the checker flagged the one difference
    the table exists to show.
    """
    sf = A.get("selection_final") or {}
    row = sf.get("main_row")
    if not row:
        return
    ncols, rows = rows_of(open(path, encoding="utf-8").read())
    hdr = rows[0][1]
    for _, r in rows[1:]:
        if r[0].replace("\\textbf{", "").rstrip("}") != "OUTPOST":
            continue
        for i, h in enumerate(hdr):
            if not h.lower().startswith("amazon"):
                continue
            met = f"{rule} {'PR' if 'PR' in h else 'ROC'}"
            ref = row.get(met)
            if not ref:
                continue
            global cells
            cells += 1
            cell = r[i].replace("\\textbf{", "").rstrip("}")
            want = f"{ref['mean']:.4f} $\\pm$ {ref['sd']:.4f}"
            if cell != want:
                problems.append(f"{path}: OUTPOST {h} tex {cell} vs selected arm {want}")


for f in sorted(glob.glob("analysis/tables/table[12]_*.tex")):
    rule = "valsel" if "valsel" in f else "oracle"
    if "gateoff" in f:
        check_gateoff(f, rule)
    elif "selarm" in f:
        check_main(f, rule, skip_datasets={"amazon"})
        check_selarm(f, rule)
    else:
        check_main(f, rule)

# appendix tex tables carry the same numbers as appendix.json by construction; spot-check the paired tables
def d_of(r): return f"{r['delta']:+.4f} ({r['W']}/{r['T']}/{r['L']})"
for key, who in (("paired_vs_demo", "demo"), ("paired_vs_nsreg", "nsreg")):
    ncols, rows = rows_of(open(f"analysis/tables/tex/paired_vs_{who}.tex", encoding="utf-8").read())
    by = {r["dataset"]: r for r in A[key]}
    for _, r in rows[1:]:
        ref = by.get(r[0].replace("\\_", "_"))
        if ref is None or "status" in ref:
            continue
        for col, k in ((2, "oracle ROC"), (4, "oracle PR"), (6, "valsel ROC")):
            cells += 1
            if r[col] != d_of(ref[k]):
                problems.append(f"paired_vs_{who}.tex: {r[0]} {k}: tex '{r[col]}' vs appendix '{d_of(ref[k])}'")

# baseline_selection.tex and hparam_selection.tex are the newest tables and the
# ones most likely to drift, because both are built from a selection JSON rather
# than from results.csv directly. Check their numeric columns against the
# appendix sections they are rendered from.
def _num(x):
    try:
        return float(str(x).replace("\\textbf{", "").rstrip("}").replace("+", ""))
    except ValueError:
        return None

if os.path.exists("analysis/tables/tex/baseline_selection.tex") and A.get("baseline_selection"):
    _, rows = rows_of(open("analysis/tables/tex/baseline_selection.tex", encoding="utf-8").read())
    by = {r["dataset"]: r for r in A["baseline_selection"]}
    for _, r in rows[1:]:
        ref = by.get(r[0].replace("\\_", "_"))
        if ref is None:
            continue
        for col, k in ((4, "val_spread"), (5, "released_oracle"), (6, "selected_oracle"),
                       (7, "delta_oracle"), (8, "delta_valsel")):
            if col >= len(r) or ref.get(k) is None:
                continue
            cells += 1
            if _num(r[col]) is None or abs(_num(r[col]) - ref[k]) > 5e-5:
                problems.append(f"baseline_selection.tex: {r[0]} {k}: tex '{r[col]}' vs appendix {ref[k]}")

if os.path.exists("analysis/tables/tex/hparam_selection.tex") and A.get("hyperparameter_selection"):
    _, rows = rows_of(open("analysis/tables/tex/hparam_selection.tex", encoding="utf-8").read())
    seen = {}
    for r in A["hyperparameter_selection"]:
        seen[(r["dataset"], str(r["epochs"]))] = r
    for _, r in rows[1:]:
        ref = seen.get((r[0].replace("\\_", "_"), r[1]))
        if ref is None:
            continue
        for col, k in ((4, "val_winner_test_roc"), (6, "best_test_roc"), (7, "selection_gap")):
            if col >= len(r) or ref.get(k) is None:
                continue
            cells += 1
            if _num(r[col]) is None or abs(_num(r[col]) - ref[k]) > 5e-5:
                problems.append(f"hparam_selection.tex: {r[0]}@{r[1]} {k}: tex '{r[col]}' vs appendix {ref[k]}")

# Characters LaTeX will refuse. There is no pdflatex on this machine, so the
# tables cannot be compile-tested; this catches the failure mode that actually
# occurs when a tag like T_hidden16 reaches a caption without escaping. Labels
# are exempt (never typeset), as are math spans and \texttt.
for _f in sorted(glob.glob("analysis/tables/*.tex") + glob.glob("analysis/tables/tex/*.tex")):
    _s = open(_f, encoding="utf-8").read()
    _s = re.sub(r"\$[^$]*\$", "", _s)
    _s = re.sub(r"\\(label|ref|cite)\{[^}]*\}", "", _s)
    _s = re.sub(r"\\texttt\{[^}]*\}", "", _s)
    # A filename argument is read with the underscore made harmless, so
    # \includegraphics{figures/law_crossval.pdf} is valid and must not be flagged.
    _s = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", _s)
    for _ln, _line in enumerate(_s.split("\n"), 1):
        # a comment line is not typeset; flagging its '%' reported 73 false problems
        # on figures.tex, which is exactly how a real one would have gone unnoticed
        if _line.lstrip().startswith("%"):
            continue
        for _ch in "_%#":
            for _m in re.finditer(re.escape(_ch), _line):
                if _m.start() > 0 and _line[_m.start() - 1] == "\\":
                    continue
                problems.append(f"{_f}:{_ln} unescaped '{_ch}' would break LaTeX: "
                                f"{_line[max(0, _m.start()-40):_m.start()+25].strip()}")

print(f"tex consistency: {cells} numeric cells checked against appendix.json, {len(problems)} problem(s)")
for p in problems:
    print("  PROBLEM:", p)
sys.exit(1 if problems else 0)
