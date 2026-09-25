"""Exhibit 1: the health measure's moments over the life course, model-free.

The framework's Figure 3 on the project's measure: the mean path, the
cross-sectional variance profile, and the rows of the age-covariance matrix,
each row starting on the diagonal at a five-year base-age bin and running nine
years. Two versions, h and theta, one row of panels each, in the measure's own
units. Five-year centred rolling means; cells with n < 100 dropped. Variance
and covariance panels share a vertical scale within a variant, so the drop
from the profile to the row leaving it is the one-year spike.

A second figure (fig_exhibit1_balanced.png) redraws the covariance rows on
balanced samples, each row over the people observed at all ten of its ages,
against the pairwise rows above them, for the appendix.

Outputs: paper/figures/fig_exhibit1.png, fig_exhibit1_balanced.png
         artifacts/descriptives/paper_exhibit1_moments.csv (kind = profile / row / row_balanced)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import MAX_AGE, MIN_AGE, VARIANTS as _V, draw_exhibit1_row, profile_rows, profile_rows_balanced  # noqa: E402
from _style import INK2, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402

VARIANTS = [("h", "$h$ (expected score, 0-1)"), ("theta", r"$\theta$ (latent, pooled $N(0,1)$)")]
MIN_CELL, ROW_LEN = 100, 9


def main() -> int:
    apply_style()
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                        columns=["pidp", "age", "h", "theta"])
    d = d[d["age"].between(20, 90)].assign(age=lambda x: x["age"].astype(int))
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), gridspec_kw={"hspace": 0.33, "wspace": 0.22})
    out_rows = []
    for j, (v, label) in enumerate(VARIANTS):
        smooth, rows = profile_rows(d, v, row_len=ROW_LEN, min_cell=MIN_CELL)
        out_rows.append(smooth.reset_index().assign(variant=v, kind="profile"))
        out_rows.append(rows.assign(variant=v, kind="row"))
        letters = "ace" if j == 0 else "bdf"
        titles = [f"({letters[0]}) {label}: mean", f"({letters[1]}) variance",
                  f"({letters[2]}) rows of the covariance matrix"]
        draw_exhibit1_row(axes[:, j], smooth, rows, label, legend=(j == 0), titles=titles)
    for ax in axes[2]:
        ax.set_xlabel("age")
    fig.text(0.01, -0.01,
             f"{len(d):,} person-waves, {d['pidp'].nunique():,} people, ages 20-90. Profiles are five-year centred "
             "rolling means; cells with fewer than 100 person-waves are dropped. Each covariance row starts on the\n"
             "diagonal at its five-year base-age bin (dotted: the variance profile) and runs nine years. Rows two and "
             "three share a vertical scale within each column.",
             fontsize=7.4, color=INK2, va="top")
    out = ROOT_DIR / "paper" / "figures" / "fig_exhibit1.png"
    fig.savefig(out)
    plt.close(fig)

    # ---- appendix: the covariance rows, pairwise against balanced ------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), gridspec_kw={"hspace": 0.35, "wspace": 0.22})
    cmap = plt.get_cmap("viridis")
    npeople = {}
    for c, (v, label) in enumerate(VARIANTS):
        smooth, pair = profile_rows(d, v, row_len=ROW_LEN, min_cell=MIN_CELL)
        bal = profile_rows_balanced(d, v, row_len=ROW_LEN, min_cell=MIN_CELL)
        out_rows.append(bal.assign(variant=v, kind="row_balanced"))
        npeople[v] = bal.groupby("start")["n"].first()
        for r, (name, rr) in enumerate([("pairwise, as in Exhibit 1", pair), ("balanced: observed at all ten ages of the row", bal)]):
            ax = axes[r, c]
            starts = sorted(rr["start"].unique())
            for k, s0 in enumerate(starts):
                seg = rr[rr["start"] == s0].sort_values("lag")
                ax.plot(seg["later_age"], seg["cov"], color=cmap(0.9 * k / max(len(starts) - 1, 1)), lw=1.0, marker="o", ms=2)
            ax.plot(smooth.index, smooth["var"], color=VERM, lw=0.8, ls=":", alpha=0.7)
            ax.set_title(f"({'abcd'[2 * r + c]}) {label.split(' (')[0]}: {name}", loc="left", fontsize=9.5)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if r == 1:
                ax.set_xlabel("age")
        lo = min(pair["cov"].min(), bal["cov"].min(), 0)
        hi = max(pair["cov"].max(), bal["cov"].max(), smooth["var"].max()) * 1.05
        for r in range(2):
            axes[r, c].set_ylim(lo, hi)
    n = npeople["h"]
    fig.text(0.01, -0.01,
             "Rows of the age-covariance matrix leaving the diagonal at five-year base-age bins and running nine years; dotted red is "
             "the smoothed variance profile. Top: each cell over the people\nobserved at both of its ages, as in Exhibit 1. Bottom: each "
             "row over the people observed at every one of its ten ages, which leaves "
             f"{int(n.min()):,} to {int(n.max()):,} people per row and nothing from 80.\nCells with fewer than 100 people dropped; cells pooled "
             "into bins by count. Panels share a vertical scale within each column.",
             fontsize=7.4, color=INK2, va="top")
    out2 = ROOT_DIR / "paper" / "figures" / "fig_exhibit1_balanced.png"
    fig.savefig(out2, bbox_inches="tight")
    pd.concat(out_rows).to_csv(ARTIFACTS_DIR / "descriptives" / "paper_exhibit1_moments.csv", index=False)
    print(f"wrote {out} and {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
