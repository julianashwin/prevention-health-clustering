"""Exhibit 1: the health measure's moments over the life course, model-free.

The framework's Figure 3 on the project's measure: the mean path, the
cross-sectional variance profile, and the rows of the age-covariance matrix,
each row starting on the diagonal at a five-year base-age bin and running nine
years. Two versions, h and theta, one row of panels each, in the measure's own
units. Five-year centred rolling means; cells with n < 100 dropped. Variance
and covariance panels share a vertical scale within a variant, so the drop
from the profile to the row leaving it is the one-year spike.

Outputs: paper/figures/fig_exhibit1.png
         artifacts/descriptives/paper_exhibit1_moments.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import draw_exhibit1_row, profile_rows  # noqa: E402
from _style import INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402

VARIANTS = [("h", "$h$ (expected score, 0-1)"), ("theta", r"$\theta$ (latent, pooled N(0,1))")]
MIN_CELL, ROW_LEN = 100, 9


def main() -> int:
    apply_style()
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                        columns=["pidp", "age", "h", "theta"])
    d = d[d["age"].between(20, 90)].assign(age=lambda x: x["age"].astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    out_rows = []
    for i, (v, label) in enumerate(VARIANTS):
        smooth, rows = profile_rows(d, v, row_len=ROW_LEN, min_cell=MIN_CELL)
        out_rows.append(smooth.reset_index().assign(variant=v, kind="profile"))
        out_rows.append(rows.assign(variant=v, kind="row"))
        draw_exhibit1_row(axes[i], smooth, rows, label, "abc" if i == 0 else "def", legend=(i == 0))
    for ax in axes[1]:
        ax.set_xlabel("age")
    fig.text(0.01, -0.01,
             f"{len(d):,} person-waves, {d['pidp'].nunique():,} people, ages 20-90. Profiles are five-year centred "
             "rolling means; cells with fewer than 100 person-waves are dropped. Each covariance row starts on the\n"
             "diagonal at its five-year base-age bin (dotted: the variance profile) and runs nine years. The variance "
             "and covariance panels share a vertical scale within each variant.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "paper" / "figures" / "fig_exhibit1.png"
    fig.savefig(out)
    pd.concat(out_rows).to_csv(ARTIFACTS_DIR / "descriptives" / "paper_exhibit1_moments.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
