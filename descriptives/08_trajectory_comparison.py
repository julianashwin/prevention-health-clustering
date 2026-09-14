"""Fitted class trajectories and shares across the eleven univariate fits.

Each latent class implies a mean trajectory alpha_k + beta_k a + gamma_k a^2
in the channel's standardised units, with a = (age - 55)/10. Plotting all
eleven fits on the same standardised axis makes the specifications directly
comparable: the classes' levels, slopes and curvature, and the share each
class carries.

Two panels per row of the figure: the trajectories (line width proportional
to the class share) and a stacked share bar. Channels are standardised, so
one vertical axis serves all of them; the raw-scale conversion for each
channel is printed.

Outputs: docs/measurement/figures/fig_trajectory_comparison.png,
         artifacts/descriptives/trajectory_comparison.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, ROOT_DIR

AGES = np.arange(20, 91)
A = (AGES - 55) / 10.0

# (tag, label, artifacts subdirectory)
FITS = [
    ("pcs-ar1", "UKHLS PCS · AR(1)", "overnight"),
    ("pcs-ar1-ho", "UKHLS PCS · AR(1) + holdout", "overnight"),
    ("physgrm-base", "GRM physical (P-FULL) · baseline", "overnight"),
    ("physgrm-ar1", "GRM physical (P-FULL) · AR(1)", "overnight"),
    ("physgrm-ar1-ho", "GRM physical (P-FULL) · AR(1) + holdout", "overnight"),
    ("combgrm-base", "GRM combined (17 items) · baseline", "overnight"),
    ("combgrm-ar1", "GRM combined (17 items) · AR(1)", "overnight"),
    ("combgrm-ar1-ho", "GRM combined (17 items) · AR(1) + holdout", "overnight"),
    ("grm-ssm", "GRM original · AR(1)+ME", "ssm"),
    ("physfunc-ssm", "GRM phys (P-FUNC) · AR(1)+ME", "ssm"),
    ("physfull-ssm", "GRM phys (P-FULL) · AR(1)+ME", "ssm"),
]


def main() -> int:
    apply_style()
    rows, curves = [], {}
    for tag, label, subdir in FITS:
        r = json.loads((ARTIFACTS_DIR / subdir / tag /
                        "run_summary.json").read_text())
        p = r["params"]
        theta = np.array([p[f"theta[{k}]"]["mean"] for k in (1, 2, 3)])
        coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)]
                         for k in (1, 2, 3)])
        curves[tag] = (theta, np.column_stack(
            [coef[k, 0] + coef[k, 1] * A + coef[k, 2] * A**2 for k in range(3)]))
        for k in range(3):
            rows.append({
                "fit": tag, "label": label, "class": k + 1,
                "share": theta[k], "alpha": coef[k, 0], "beta": coef[k, 1],
                "gamma": coef[k, 2],
                "level_age30": curves[tag][1][AGES == 30, k][0],
                "level_age50": curves[tag][1][AGES == 50, k][0],
                "level_age80": curves[tag][1][AGES == 80, k][0],
                "drop_30_80": (curves[tag][1][AGES == 80, k][0]
                               - curves[tag][1][AGES == 30, k][0]),
                "rho": p.get(f"rho[{k+1}]", {}).get("mean", np.nan),
                "sigma": p["sigma[1,1]"]["mean"],
                "sigma_meas": p.get("sigma_meas[1]", {}).get("mean", np.nan),
            })
    tab = pd.DataFrame(rows)
    out_dir = ARTIFACTS_DIR / "descriptives"
    tab.to_csv(out_dir / "trajectory_comparison.csv", index=False)

    # How much more steeply does the worst class decline than the best?
    # (ratio of 30->80 drops) -- the trajectory-level counterpart of fanning.
    spread = []
    for tag, label, _ in FITS:
        d = tab[tab["fit"] == tag].set_index("class")
        spread.append({"fit": tag, "label": label,
                       "share_worst": d.loc[1, "share"],
                       "level_gap_age30": d.loc[3, "level_age30"] - d.loc[1, "level_age30"],
                       "level_gap_age80": d.loc[3, "level_age80"] - d.loc[1, "level_age80"],
                       "drop_worst": d.loc[1, "drop_30_80"],
                       "drop_best": d.loc[3, "drop_30_80"],
                       "drop_ratio": d.loc[1, "drop_30_80"] / d.loc[3, "drop_30_80"]})
    sp = pd.DataFrame(spread)
    sp.to_csv(out_dir / "trajectory_spread.csv", index=False)
    print(sp.round(3).to_string(index=False))
    print()

    comp = pd.read_csv(ARTIFACTS_DIR / "descriptives"
                       / "class_composition_by_age.csv")
    n_row = (len(FITS) + 3) // 4
    fig = plt.figure(figsize=(13.4, 3.8 * n_row))
    gs = fig.add_gridspec(2 * n_row, 4,
                          height_ratios=[3.2, 0.7] * n_row,
                          hspace=0.42, wspace=0.22)
    for i, (tag, label, _) in enumerate(FITS):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[2 * r, c])
        axc = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        theta, mat = curves[tag]
        for k in range(3):
            ax.plot(AGES, mat[:, k], color=CLUSTER[k],
                    lw=0.8 + 4.0 * theta[k],
                    label=f"class {k+1}: {theta[k]:.0%}")
        ax.axhline(0, color="#cccccc", lw=0.7, ls=":")
        ax.set_title(label, fontsize=8.0)
        ax.legend(fontsize=6.4, loc="lower left")
        ax.grid(True, axis="y")
        ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel("standardised channel units", fontsize=8.5)
        g = comp[comp["fit"] == tag].sort_values("age")
        axc.stackplot(g["age"], *[g[f"class{k+1}"] for k in range(3)],
                      colors=CLUSTER, alpha=0.9)
        axc.set_ylim(0, 1); axc.set_yticks([]); axc.set_xlim(20, 90)
        axc.spines["left"].set_visible(False)
        if r == n_row - 1 or i + 4 >= len(FITS):
            axc.set_xlabel("age")
    fig.suptitle("Fitted class trajectories and shares across the eleven "
                 "univariate fits", fontweight="bold", y=1.0)
    fig.text(0.01, -0.006,
             "Line width is proportional to the class share. All channels are standardised (mean 0, sd 1 over\n"
             "the fitted sample), so levels are comparable across panels; class 1 is worst health by the\n"
             "anchor convention. The strip beneath each panel is the model's class composition of the\n"
             "person-waves OBSERVED at each age: theta is a lifetime constant, but who is in the sample\n"
             "is not. ME marks the state-space specification of the last three panels: an AR(1) latent health\n"
             "state carrying the persistence, with an independent one-period measurement error on top.",
             fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "measurement" / "figures" / "fig_trajectory_comparison.png"
    fig.savefig(out)
    print(tab.round(3).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
