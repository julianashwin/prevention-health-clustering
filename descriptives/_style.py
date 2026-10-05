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
