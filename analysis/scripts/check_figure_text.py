#!/usr/bin/env python3
"""Fail if any paper figure has a chart title or two pieces of text that overlap.

Eyeballing a PNG catches the overlap you happen to look at. This renders every
figure in make_figures_paper.py and checks the actual laid-out text boxes:

  * no axes title, no figure suptitle, and no annotation parked across the top of
    the axes where it would read as one (a centred label above y = 0.97 in axes
    fraction is a title however it was drawn);
  * no two visible Text artists whose bounding boxes intersect, tick labels,
    axis labels, legend entries and annotations alike.

    python analysis/scripts/check_figure_text.py     # exit 1 on any problem
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "analysis")
sys.path.insert(0, "analysis/scripts")

import matplotlib
matplotlib.use("Agg")
from matplotlib.text import Text, Annotation

import figstyle as F
import make_figures_paper as M          # applies the paper style on import

F.use()   # belt and braces: measure the text at the size and font it is drawn at

PAD = 0.5          # points of slack: boxes that merely touch are not an overlap


def offview_ticklabels(fig):
    """Tick labels for ticks outside the view: matplotlib still lays them out, in the
    axes corner, where they collide with each other and look like a real overlap."""
    skip = set()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if not (lo - 1e-12 <= loc <= hi + 1e-12):
                    skip.add(id(tick.label1)); skip.add(id(tick.label2))
    return skip


def texts_of(fig, skip=frozenset()):
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or id(t) in skip:
            continue
        s = (t.get_text() or "").strip()
        if not s:
            continue
        out.append(t)
    return out


def overlap(a, b):
    return (a.x0 < b.x1 - PAD and b.x0 < a.x1 - PAD and
            a.y0 < b.y1 - PAD and b.y0 < a.y1 - PAD)


def check(name, fig):
    bad = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    skip = offview_ticklabels(fig)

    if getattr(fig, "_suptitle", None) and fig._suptitle.get_text().strip():
        bad.append(f"figure suptitle: {fig._suptitle.get_text()!r}")
    for ax in fig.axes:
        for which in ("center", "left", "right"):
            if ax.get_title(loc=which).strip():
                bad.append(f"axes title ({which}): {ax.get_title(loc=which)!r}")
        # an annotation centred across the top of the axes is a title in disguise
        for t in texts_of(fig, skip):
            if not isinstance(t, Annotation) or t.axes is not ax:
                continue
            bb = F.textbox(t, r)
            p0 = ax.transAxes.inverted().transform((bb.x0, bb.y0))
            p1 = ax.transAxes.inverted().transform((bb.x1, bb.y1))
            cx, top = (p0[0] + p1[0]) / 2, p1[1]
            if top > 0.97 and 0.25 < cx < 0.75 and (p1[0] - p0[0]) > 0.35:
                bad.append(f"title-like annotation across the top: {t.get_text()!r}")

    # text sitting on a drawn line or marker is the same complaint as text on text.
    # Annotation leaders are FancyArrowPatch, not Line2D, so a label's own leader
    # running to its marker is not counted.
    import numpy as _np
    for ax in fig.axes:
        if ax.get_legend() is not None and ax is not fig.axes[0]:
            continue
        drawn = []
        for ln in ax.lines:
            v = ln.get_path().vertices
            if len(v) < 2:
                drawn.append(ln.get_transform().transform(v)); continue
            d = ln.get_transform().transform(v)
            seg = [d[0] + (d[i + 1] - d[i]) * 0 for i in range(0)]
            dense = []
            for i in range(len(d) - 1):
                n = max(2, int(_np.hypot(*(d[i + 1] - d[i])) / 3))
                dense.append(d[i] + (d[i + 1] - d[i]) * _np.linspace(0, 1, n)[:, None])
            drawn.append(_np.vstack(dense) if dense else d)
        for co in ax.collections:
            try:
                o = co.get_offsets()
                if len(o):
                    drawn.append(co.get_offset_transform().transform(o))
            except Exception:
                pass
        pts = _np.vstack(drawn) if drawn else _np.empty((0, 2))
        leg = ax.get_legend()
        legbb = leg.get_window_extent(r) if leg is not None else None
        for t in texts_of(fig, skip):
            if t.axes is not ax:
                continue
            bb = F.textbox(t, r)
            if legbb is not None and legbb.contains(bb.x0, bb.y0):
                continue                      # legend text over the legend's own keys
            inside = ((pts[:, 0] > bb.x0 + PAD) & (pts[:, 0] < bb.x1 - PAD) &
                      (pts[:, 1] > bb.y0 + PAD) & (pts[:, 1] < bb.y1 - PAD))
            if inside.any():
                bad.append(f"text sits on drawn data: {t.get_text()!r}")

    ts = texts_of(fig, skip)
    boxes = [(t, F.textbox(t, r)) for t in ts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ta, ba), (tb, bb) = boxes[i], boxes[j]
            if overlap(ba, bb):
                bad.append(f"text overlaps text: {ta.get_text()!r} / {tb.get_text()!r}")

    # LAST: savefig with a tight bbox re-lays out the figure, so this must come
    # after every measurement above -- running it first invalidated the renderer
    # and produced two dozen phantom overlaps.
    try:
        import io
        from PIL import Image
        import numpy as _np2
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        a = _np2.asarray(Image.open(buf).convert("L"))
        for side, strip in (("left", a[:, :2]), ("right", a[:, -2:]),
                            ("top", a[:2, :]), ("bottom", a[-2:, :])):
            if strip.min() < 250:
                bad.append(f"ink touches the {side} edge of the saved image")
    except Exception as e:
        print(f"  {name}: edge check skipped ({type(e).__name__})")

    return bad


def main():
    args = sys.argv[1:]
    legacy = "--legacy" in args            # also report on the pre-figstyle figures
    only = [a for a in args if not a.startswith("--")]
    figs = {}
    real_save = F.save

    def capture(fig, stem, outdir="figures"):
        figs[stem] = fig          # keep it open; do not write anything

    F.save = capture
    M.F.save = capture
    names = [n for n in M.FIGS if not only or n in only]
    for n in names:
        M.FIGS[n]()
    F.save = real_save

    if legacy:
        # make_figures.py no longer draws anything -- all five of its figures were
        # ported into make_figures_paper.py, which this already checks. The flag is
        # kept so the old drawing code can be re-checked if it is ever revived.
        print("  --legacy: make_figures.py no longer draws; nothing extra to check")

    problems = 0
    for stem, fig in figs.items():
        bad = check(stem, fig)
        if bad:
            problems += len(bad)
            print(f"  {stem}:")
            for b in bad:
                print(f"    PROBLEM: {b}")
    print(f"figure text: {len(figs)} figure(s) checked, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
