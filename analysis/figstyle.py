"""Figure style for the OUTPOST paper: one place, so every panel matches.

Conventions, fixed deliberately:
  * no in-figure titles or subtitles - the LaTeX caption carries the message, and
    a title duplicated above a caption wastes the most valuable strip of the page
  * black edgecolors on every marker, bar and line, so elements read bold at the
    single-column width a reviewer actually sees them at
  * serif type, to sit with the body text rather than against it
  * a warm high-contrast palette that survives greyscale printing and the common
    colour-vision deficiencies
  * legends outside the data, never floating over it
  * PNG for drafts and PDF (vector) for submission, from one call
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# warm, high contrast, distinguishable in greyscale by lightness order
CRIMSON = "#A5243D"
AMBER = "#D98324"
PLUM = "#5B2A54"
TEAL = "#2A6F6B"
SAND = "#C8B18B"
INK = "#141413"
GREY = "#8A8580"

# stable semantic roles, so a colour means one thing across the whole paper
METHOD = {"OUTPOST": CRIMSON, "DEMO": GREY, "NSREG": PLUM, "NSReg": PLUM}
KIND = {"semi-synthetic": AMBER, "OGB": TEAL, "real": PLUM}
RULE = {"oracle": AMBER, "val-selected": PLUM}

EDGE = dict(edgecolor=INK, linewidth=0.9)      # scatter/bar: patch kwargs
# Line2D (ax.plot, errorbar) spells the same thing differently and raises on
# `edgecolor`; keep both so a caller never has to remember which artist it has.
MEDGE = dict(markeredgecolor=INK, markeredgewidth=0.9)


def use():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 9,
        "axes.titlesize": 9,          # titles are not used; size set for safety
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D8D4CF",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.edgecolor": INK,
        "legend.framealpha": 1.0,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,            # embed real fonts, not type-3 outlines
        "ps.fonttype": 42,
    })


def save(fig, stem, outdir="figures"):
    """PNG for reading, PDF for submitting. No title is ever added here."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  figures/{stem}.pdf + .png")


# Result files spell datasets three ways: run-directory form (tfinance), appendix form
# (T-Finance) and OGB's own (ogbn-mag). Figures and captions must agree, so every
# dataset name that reaches a reader goes through here.
NICE = {"tfinance": "T-Finance", "t-finance": "T-Finance", "yelp": "Yelp",
        "amazon": "Amazon", "computers": "Computers", "photo": "Photo", "cs": "CS",
        "ogbn-arxiv": "ogbn-arxiv", "ogbn-mag": "ogbn-mag"}


def nice(name):
    """Paper spelling for a dataset, unchanged if it is not one we know."""
    return NICE.get(str(name).strip().lower(), name)


def textbox(t, renderer):
    """The box of the TEXT alone.

    Annotation.get_window_extent() returns the union of the text and its leader
    arrow, so a label with a leader measures far larger than it looks -- which
    made every placement and overlap test wrong in the same direction.
    """
    from matplotlib.text import Text as _T
    return _T.get_window_extent(t, renderer)


def _nudge(ann, dx, dy):
    x, y = ann.xyann                    # offset in points, for textcoords="offset points"
    ann.xyann = (x + dx, y + dy)


def declutter(fig, anns, pad=2.0, axis="y", iters=30, bounds=None):
    """Push offset-point annotations apart until their drawn boxes stop overlapping.

    Guessing a label's width from its character count is what let "Computers (5)"
    settle on top of "ogbn-arxiv (4)": the guess was narrower than the text. This
    measures the boxes matplotlib actually laid out and separates them by the real
    overlap, so the result does not depend on the font, the dpi or the string.

    axis="y" only ever moves labels vertically (a column of labels down one edge);
    axis="xy" separates along whichever direction needs the smaller move.

    `bounds` is a display-space bbox the labels must stay inside, normally the axes.
    Without it a crowded pair gets pushed clean off the plot and into the tick
    labels, which trades one overlap for a worse one.
    """
    anns = [a for a in anns if a is not None]
    if len(anns) < 2:
        return
    px2pt = 72.0 / fig.dpi
    for _ in range(iters):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bx = [textbox(a, r) for a in anns]
        clashed = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                bi, bj = bx[i], bx[j]
                ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0) + pad
                oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + pad
                if ox <= 0 or oy <= 0:
                    continue
                clashed = True
                if axis == "y" or oy <= ox:
                    s = oy / 2 * px2pt
                    hi, lo = ((i, j) if bi.y0 + bi.y1 >= bj.y0 + bj.y1 else (j, i))
                    _nudge(anns[hi], 0, s); _nudge(anns[lo], 0, -s)
                else:
                    s = ox / 2 * px2pt
                    rt, lf = ((i, j) if bi.x0 + bi.x1 >= bj.x0 + bj.x1 else (j, i))
                    _nudge(anns[rt], s, 0); _nudge(anns[lf], -s, 0)
        if bounds is not None:
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            for a in anns:
                b = textbox(a, r)
                dx = max(bounds.x0 - b.x0, 0) + min(bounds.x1 - b.x1, 0)
                dy = max(bounds.y0 - b.y0, 0) + min(bounds.y1 - b.y1, 0)
                if dx or dy:
                    _nudge(a, dx * px2pt, dy * px2pt)
                    clashed = True
        if not clashed:
            return


def drawn_points(ax, step=3.0):
    """Every plotted line and marker of `ax`, as display-space points.

    Lines are densified so a long segment is not missed between its two vertices.
    Used both to place labels clear of the data and to check that they are.
    """
    import numpy as _np
    out = []
    for ln in ax.lines:
        v = ln.get_path().vertices
        if len(v) == 0:
            continue
        d = ln.get_transform().transform(v)
        if len(d) < 2:
            out.append(d); continue
        dense = []
        for i in range(len(d) - 1):
            n = max(2, int(_np.hypot(*(d[i + 1] - d[i])) / step))
            dense.append(d[i] + (d[i + 1] - d[i]) * _np.linspace(0, 1, n)[:, None])
        out.append(_np.vstack(dense))
    for co in ax.collections:
        try:
            o = co.get_offsets()
            if len(o):
                out.append(co.get_offset_transform().transform(_np.asarray(o)))
        except Exception:
            pass
    return _np.vstack(out) if out else _np.empty((0, 2))


def place_labels(fig, ax, anns, avoid=None, keepout=(), radii=(13, 20, 30, 42, 56, 72),
                 ndirs=16):
    """Position annotations by measuring their text, never by guessing its width.

    Each label tries `ndirs` directions at each radius and keeps the first candidate
    that covers no plotted point, no already-placed label and nothing outside the
    axes; failing that, the least bad. Sizing a label by its character count is what
    put one dataset's name across another's: the guess was narrower than the text, so
    a position that scored clear was not.

    Overlap is weighted far above leaving the axes, because the page is saved with a
    tight bounding box: a label just outside is still drawn, a label on top of other
    text is unreadable.
    """
    import numpy as _np
    from matplotlib.transforms import Bbox as _Bbox
    if not anns:
        return []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    if avoid is None:
        avoid = drawn_points(ax)
    ab = ax.bbox
    dirs = [(_np.cos(t), _np.sin(t))
            for t in _np.linspace(0, 2 * _np.pi, ndirs, endpoint=False)]
    occupied = list(keepout)

    def cost(bx):
        off = ((max(ab.x0 - bx.x0, 0) + max(bx.x1 - ab.x1, 0)) * bx.height +
               (max(ab.y0 - bx.y0, 0) + max(bx.y1 - ab.y1, 0)) * bx.width)
        over = sum(max(0, min(bx.x1, k.x1) - max(bx.x0, k.x0)) *
                   max(0, min(bx.y1, k.y1) - max(bx.y0, k.y0)) for k in occupied)
        hit = 0.0
        if len(avoid):
            m = ((avoid[:, 0] > bx.x0 - 3) & (avoid[:, 0] < bx.x1 + 3) &
                 (avoid[:, 1] > bx.y0 - 3) & (avoid[:, 1] < bx.y1 + 3))
            hit = float(m.any())
        return off * 0.5 + over * 5 + hit * 400

    boxes = []
    for a in anns:
        bb = textbox(a, r)
        w, h = bb.width, bb.height
        x0d, y0d = ax.transData.transform(a.xy)
        best = bestbox = bestv = None
        for rad in radii:
            for v in dirs:
                cx = x0d + v[0] * rad * fig.dpi / 72
                cy = y0d + v[1] * rad * fig.dpi / 72
                lx = cx - w / 2 if abs(v[0]) < 0.6 else (cx if v[0] > 0 else cx - w)
                cand = _Bbox.from_bounds(lx, cy - h / 2, w, h)
                c = cost(cand)
                if best is None or c < best:
                    best, bestbox, bestv = c, cand, (v[0] * rad, v[1] * rad)
                if c == 0:
                    break
            if best == 0:
                break
        a.set_ha("center" if abs(bestv[0]) < 0.6 * radii[0]
                 else ("left" if bestv[0] > 0 else "right"))
        a.xyann = bestv
        occupied.append(bestbox)
        boxes.append(bestbox)
    return boxes
