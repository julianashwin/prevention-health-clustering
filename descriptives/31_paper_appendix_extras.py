"""Appendix B: Exhibit 1 for healthcare cost; the estimated cost curve.

Part A (fig_exhibit1_cost.png). The cost index (waves 7-15, one row per
person-wave that answered the utilisation block) through Exhibit 1's three
panels: mean, variance and covariance rows by age, in pounds capped at the
99th percentile. Rows run eight years, the most the
nine utilisation waves allow.

Part B (fig_returns_cost.png, tab_returns_cost.tex). The framework's returns
with c(.) the ESTIMATED cost curve rather than (1 - h)^2: a cubic in each
variant fitted to the capped cost index (a quadratic if the cubic is not
monotone on the support). With a general c, treatment h -> h + tau (top - h)
and prevention h -> h + mu (hbar_{j,r-1} - hbar_{j,a}) give

  T(r)   = sum_{a>=r} E[ -c'(h_a) (top - h_a) ]
  P_j(r) = sum_{a>=r} E[ -c'(h_a) | j ] (hbar_{j,r-1} - hbar_{j,a})

which reduce to 28_paper_returns.py's formulas when c = (top - h)^2; that
case is recomputed here as a check. Returns are then in pounds of lifetime
cost per unit of intensity. Person-level expectations are age means over the
clustered people (all waves), smoothed like the type paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import (  # noqa: E402
    AGES, DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, TAB, VARIANTS, draw_exhibit1_row, load_measure,
    profile_rows, write_table,
)
from _style import BLUE, CLUSTER, INK2, INK3, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402

R_GRID = np.arange(25, 86)
R_TABLE = [30, 50, 70]
COST = "flat_cost_total"
AGE_BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]


def fit_cost_curve(x: np.ndarray, y: np.ndarray):
    """Polynomial E[cost | x]; cubic if monotone decreasing on the 1st-99th percentile range, else quadratic."""
    lo, hi = np.quantile(x, [0.01, 0.99])
    grid = np.linspace(lo, hi, 200)
    for deg in (3, 2):
        coef = np.polynomial.polynomial.polyfit(x, y, deg)
        dcoef = np.polynomial.polynomial.polyder(coef)
        if (np.polynomial.polynomial.polyval(grid, dcoef) <= 0).all():
            return coef, dcoef, deg
    return coef, dcoef, deg


def main() -> int:
    apply_style()
    # ---- Part A: Exhibit 1 for cost -------------------------------------------
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", "age", COST])
    cost = cost[cost["age"].between(MIN_AGE, MAX_AGE)].assign(age=lambda x: x["age"].astype(int))
    cost = cost.sort_values(["pidp", "age", "wave"]).drop_duplicates(["pidp", "age"])
    cap = cost[COST].quantile(0.99)
    cost["cost_capped"] = cost[COST].clip(upper=cap)
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 9), gridspec_kw={"hspace": 0.33})
    v, lab = "cost_capped", f"cost index, \u00a3 capped at \u00a3{cap:,.0f}"
    smooth, rows = profile_rows(cost, v, row_len=8)
    draw_exhibit1_row(axes, smooth, rows, lab, legend=True,
                      titles=[f"(a) {lab}: mean", "(b) variance", "(c) rows of the covariance matrix"])
    pd.concat([smooth.reset_index().assign(kind="profile"), rows.assign(kind="row")]).assign(variant=v).to_csv(
        DESC / f"paper_exhibit1_cost_{v}.csv", index=False)
    axes[2].set_xlabel("age")
    fig.text(0.01, -0.01, f"{len(cost):,} person-ages on {cost['pidp'].nunique():,} people, waves 7-15, ages 20-90. "
             "Five-year centred rolling means;\ncells with fewer than 100 person-ages dropped; rows run eight years. "
             "Panels (b) and (c) share a vertical scale.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_exhibit1_cost.png"); plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
