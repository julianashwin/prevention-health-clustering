"""Mean, variance and covariance-matrix rows by age (empirics-note Figure 3).

One column per measure, three stacked panels on a common age axis: the mean
profile, the cross-sectional variance profile (both five-year centred rolling
means), and the rows of the age-covariance matrix — each line starts on the
diagonal at a five-year base-age bin and runs nine years, cells kept at
n >= 100. Variance and covariance panels share a vertical scale per measure.
The chronic count cannot fall by construction, so its covariance row carries
no persistence information; it is included for the mean and variance reads.
GHQ and the chronic count are negated (higher = better) so every panel reads
the same way. Twelve measures, two banks of six.

Outputs: docs/measurement/figures/fig_moments_grid.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import BLUE, INK2, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR, ROOT_DIR

MEASURES = {
    "sf12pcs_dv": "UKHLS PCS",
    "sf12mcs_dv": "UKHLS MCS",
    "PCS_phys_only": "physical-only composite",
    "PCS_uk_promax": "UK promax PCS",
    "sf6d_utility": "SF-6D utility",
    "ghq_flipped": "GHQ-12 (flipped)",
    "grm_theta": "GRM original",
    "theta_phys_func": "GRM-2 phys (FUNC)",
    "theta_phys_full": "GRM-2 phys (FULL)",
    "theta_ment": "GRM-2 mental",
    "theta_combined": "GRM-2 combined",
    "chronic_flipped": "chronic conditions (negated)",
}
PER_ROW = 6
MIN_CELL = 100


def main() -> int:
    apply_style()
    cols = ["pidp", "age", "ghq_likert", "n_chronic"] + [
        m for m in MEASURES if m not in ("ghq_flipped", "chronic_flipped")]
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=list(dict.fromkeys(cols)))
    # Both are negated so that on EVERY panel higher means better health,
    # which is what makes the twelve columns readable side by side.
    panel["ghq_flipped"] = -panel["ghq_likert"]
    panel["chronic_flipped"] = -panel["n_chronic"]
    panel = panel[panel["age"].notna()]
    panel["age"] = panel["age"].astype(int)
    panel = panel[panel["age"].between(20, 90)]

    n_banks = int(np.ceil(len(MEASURES) / PER_ROW))
    fig, axes = plt.subplots(3 * n_banks, PER_ROW,
                             figsize=(2.45 * PER_ROW, 6.9 * n_banks))
    for idx, (m, title) in enumerate(MEASURES.items()):
        bank, j = divmod(idx, PER_ROW)
        row0 = 3 * bank
        d = panel[["pidp", "age", m]].dropna().rename(columns={m: "v"})
        prof = d.groupby("age")["v"].agg(["mean", "var", "count"])
        prof = prof[prof["count"] >= MIN_CELL]
        smooth = prof[["mean", "var"]].rolling(5, center=True).mean().dropna()

        d84 = d[d["age"] <= 84]
        pairs = d84.merge(d, on="pidp", suffixes=("", "_t"))
        pairs = pairs[(pairs["age_t"] >= pairs["age"])
                      & (pairs["age_t"] - pairs["age"] <= 9)]
        cell = pairs.groupby(["age", "age_t"]).apply(
            lambda g: pd.Series({"cov": np.cov(g["v"], g["v_t"], ddof=1)[0, 1],
                                 "n": len(g)}), include_groups=False).reset_index()
        cell = cell[cell["n"] >= MIN_CELL]
        cell["start"] = 5 * (cell["age"] // 5)
        cell["lag"] = cell["age_t"] - cell["age"]
        rows = (cell.groupby(["start", "lag"])
                .apply(lambda g: pd.Series(
                    {"cov": np.average(g["cov"], weights=g["n"])}),
                    include_groups=False)
                .reset_index())
        rows["later_age"] = rows["start"] + rows["lag"]

        ylim = (min(0, rows["cov"].min(), smooth["var"].min()),
                max(rows["cov"].max(), smooth["var"].max()) * 1.05)
        axes[row0, j].plot(smooth.index, smooth["mean"], color=BLUE, lw=1.6)
        axes[row0, j].set_title(title, fontsize=9)
        axes[row0 + 1, j].plot(smooth.index, smooth["var"], color=VERM, lw=1.6)
        axes[row0 + 1, j].set_ylim(*ylim)
        cmap = plt.get_cmap("viridis")
        starts = sorted(rows["start"].unique())
        for k, st in enumerate(starts):
            seg = rows[rows["start"] == st]
            axes[row0 + 2, j].plot(seg["later_age"], seg["cov"],
                                   color=cmap(0.9 * k / max(len(starts) - 1, 1)),
                                   lw=0.9, marker="o", ms=1.5)
        axes[row0 + 2, j].set_ylim(*ylim)
        if bank == n_banks - 1:
            axes[row0 + 2, j].set_xlabel("age")
        for r, lab in ((0, "mean"), (1, "variance"), (2, "covariance")):
            if j == 0:
                axes[row0 + r, 0].set_ylabel(lab)
            axes[row0 + r, j].grid(True, axis="y")
            axes[row0 + r, j].tick_params(labelsize=7)
    fig.suptitle("Mean, variance and the rows of the covariance matrix",
                 fontweight="bold", y=1.0)
    fig.text(0.01, -0.004,
             "Profiles are five-year centred rolling means; cells with n < 100 are dropped. Each covariance line\n"
             "starts on the diagonal at its five-year bin and runs nine years. Rows two and three share a vertical\n"
             "scale within each measure. The chronic count cannot fall by construction (so its negated series\n"
             "cannot rise), and its bottom row therefore carries no information about persistence.",
             fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "measurement" / "figures" / "fig_moments_grid.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
