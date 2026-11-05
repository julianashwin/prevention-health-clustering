"""Figure 2 (Exhibit 1) on PSID: theta's mean path, variance profile and covariance rows, model-free.

The main mode of descriptives/23_paper_exhibit1.py with the same helpers (_paper_common.profile_rows,
draw_exhibit1_row): ten-year centred rolling means, cells with fewer than 100 person-waves dropped,
covariance rows leaving the diagonal at five-year base-age bins. PSID waves are biennial, so most
within-person lags are even; odd lags (interview dates move within the year) rest on about a tenth
of the pairs, make the rows saw-toothed and are left out of the figure (kept in the csv). Rows run ten years here (lags 0-10) so the
last even lag is kept; the UKHLS rows run nine.

Outputs: descriptives/psid/figures/main/fig_exhibit1.png, artifacts/descriptives/psid/exhibit1_moments.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, FIG, scores  # noqa: E402
from _paper_common import PROFILE_WINDOW, draw_exhibit1_row, profile_rows  # noqa: E402
from _style import INK2, apply_style  # noqa: E402

MIN_CELL, ROW_LEN = 100, 10


def main() -> int:
    apply_style()
    d = scores()[["pidp", "age", "theta"]].dropna()
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 9), gridspec_kw={"hspace": 0.33})
    smooth, rows = profile_rows(d, "theta", row_len=ROW_LEN, min_cell=MIN_CELL)
    rows_all = rows
    rows = rows[rows["lag"] % 2 == 0]          # odd lags rest on a tenth of the pairs and make the rows saw-toothed
    label = r"$\theta$ (latent, pooled $N(0,1)$)"
    draw_exhibit1_row(axes, smooth, rows, label, legend=True,
                      titles=[f"(a) {label}: mean", "(b) variance", "(c) rows of the covariance matrix"])
    axes[2].set_xlabel("age")
    fig.text(0.01, -0.01,
             f"PSID, {len(d):,} person-waves, {d['pidp'].nunique():,} people, ages 20-90.\nProfiles are {PROFILE_WINDOW}-year centred rolling means;\n"
             "cells with fewer than 100 person-waves are dropped.\nEach covariance row starts on the diagonal at its\n"
             "five-year base-age bin (dotted: the variance profile)\nand runs ten years, at even lags only: waves are two years\n"
             "apart, and the odd lags (interview dates moving within\nthe year) rest on about a tenth of the pairs.\n"
             "Panels (b) and (c) share a vertical scale.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_exhibit1.png", bbox_inches="tight")
    pd.concat([smooth.reset_index().assign(kind="profile"), rows_all.assign(kind="row")]).to_csv(DESC / "exhibit1_moments.csv", index=False)
    lag = rows.groupby("lag")["n"].sum()
    print("person-pairs by lag:", lag.astype(int).to_dict())
    print(smooth.iloc[::10].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
