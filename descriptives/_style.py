"""Shared figure style, carried from the PCS construction note's figures."""

import os
import sys
from pathlib import Path

import matplotlib as mpl

BLUE, ORANGE, GREEN, VERM, PURPLE = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7B52AB")
CLUSTER = ["#08306b", "#3182bd", "#6baed6"]   # sequential, worst -> best health
INK, INK2, INK3 = "#1a1a1a", "#4a4a4a", "#7a7a7a"
SURFACE = "#fcfcfb"
NOTES_DIR = "notes"   # each saved figure's footer goes to <figure dir>/notes/<stem>.txt
_REPO = Path(__file__).resolve().parents[1]


def apply_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "text.color": INK, "axes.labelcolor": INK2,
    })
    _install_footer_guard()


def fit_footers(fig, margin: float = 0.985) -> None:
    """Re-wrap any footer wider than the figure so it never sticks out past the panels.

    A footer is a figure-level text anchored at its top near the bottom edge (va="top",
    y <= 0.05). Its widest line is measured against the figure width; when it overflows,
    the text is re-flowed to as many characters per line as fit. Explicit newlines in the
    original are dropped in that case, so authors need not size footers by hand."""
    import textwrap
    if not fig.texts:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = fig.bbox.width * margin
    for t in list(fig.texts):
        if t.get_va() != "top" or t.get_position()[1] > 0.05 or not t.get_text().strip():
            continue
        ext = t.get_window_extent(renderer=renderer)
        if ext.width <= width:
            continue
        text = t.get_text()
        longest = max(len(line) for line in text.split("\n"))
        chars = max(40, int(longest * width / ext.width) - 1)
        t.set_text("\n".join(textwrap.wrap(text.replace("\n", " "), chars)))


def find_footers(fig) -> list:
    """The figure's footers (its notes): figure-level texts hung from their top (va="top")
    near the bottom edge (y <= 0.05) or below the lowest panel. The suptitle is va="top" too
    but sits above the panels, so it is never picked up."""
    axes = [a for a in fig.axes if a.get_visible()]
    bottom = min((a.get_position().y0 for a in axes), default=0.0)
    out = []
    for t in fig.texts:
        if t.get_va() != "top" or not t.get_text().strip():
            continue
        y = t.get_position()[1]
        if y <= 0.05 or y < bottom:
            out.append(t)
    return out


def note_path(fname):
    """Where a saved figure's note goes: ``notes/<stem>.txt`` beside the figure, e.g.
    paper/figures/notes/fig_returns_bayes.txt for paper/figures/fig_returns_bayes.png.
    None for a file object."""
    if not isinstance(fname, (str, os.PathLike)):
        return None
    p = Path(fname)
    return p.parent / NOTES_DIR / f"{p.stem}.txt"


def write_note(fname, footers) -> None:
    """Write the figure's note: the figure name, the command that made it, then the footer
    text unwrapped (one paragraph per footer). Written on every save, so it is as new as
    the figure; a figure without a footer gets "(no note)"."""
    path = note_path(fname)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    try:
        script = script.relative_to(_REPO).as_posix()
    except (AttributeError, ValueError):
        script = script.name if script else "?"
    cmd = " ".join([script] + sys.argv[1:])
    body = "\n\n".join(" ".join(t.get_text().split()) for t in footers) or "(no note)"
    path.write_text(f"{Path(fname).stem}\nmade by: {cmd}\n\n{body}\n")


def _install_footer_guard() -> None:
    """Make every Figure.savefig write the figure's footer to its notes file and leave it
    off the image (idempotent). Set PHC_FOOTERS_ON_IMAGE=1 to draw the footers on the
    image as before (re-flowed to fit by fit_footers); the note is written either way."""
    from matplotlib.figure import Figure
    if getattr(Figure, "_phc_footer_guard", False):
        return
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        try:
            self.canvas.draw()   # settle the panel positions first (layout engines)
        except Exception:
            pass
        footers = find_footers(self)
        write_note(fname, footers)
        if os.environ.get("PHC_FOOTERS_ON_IMAGE"):
            fit_footers(self)
            return original(self, fname, *args, **kwargs)
        shown = [t.get_visible() for t in footers]
        for t in footers:
            t.set_visible(False)   # hidden artists are left out of the tight bounding box too
        try:
            return original(self, fname, *args, **kwargs)
        finally:
            for t, v in zip(footers, shown):
                t.set_visible(v)

    Figure.savefig = savefig
    Figure._phc_footer_guard = True


def share_ylim(axes, pad: float = 0.0, label_first_only: bool = False) -> None:
    """Give every axis in ``axes`` the same vertical limits, the union of their own.

    For panels that show the same quantity (the same measure across class counts,
    specifications or windows), so that heights can be compared across panels.
    ``label_first_only`` hides the tick labels on all but the first axis.
    """
    axes = [a for a in axes if a is not None]
    if not axes:
        return
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    span = hi - lo
    for i, a in enumerate(axes):
        a.set_ylim(lo - pad * span, hi + pad * span)
        if label_first_only and i > 0:
            a.tick_params(labelleft=False)
