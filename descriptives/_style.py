"""Shared figure style, carried from the PCS construction note's figures."""

import matplotlib as mpl

BLUE, ORANGE, GREEN, VERM, PURPLE = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7B52AB")
CLUSTER = ["#08306b", "#3182bd", "#6baed6"]   # sequential, worst -> best health
INK, INK2, INK3 = "#1a1a1a", "#4a4a4a", "#7a7a7a"
SURFACE = "#fcfcfb"


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


def _install_footer_guard() -> None:
    """Make every Figure.savefig fit its footers first (idempotent)."""
    from matplotlib.figure import Figure
    if getattr(Figure, "_phc_footer_guard", False):
        return
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        fit_footers(self)
        return original(self, *args, **kwargs)

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
