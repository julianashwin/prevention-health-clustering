"""Figure B.1 on PSID: the measure, h against theta and where each item speaks.

The paper's version of descriptives/21_health_measure.py's scale_panels (copied; that script reads
UKHLS microdata), on the PSID five-testlet GRM (data_cleaning/02_build_psid_health.py):
  (a) the expected-score curve mapping theta to h, over the pooled distribution of theta
  (b) each testlet's discrimination against its category thresholds
(psid_health_measure_items.csv: GH, RP, LIM_FL, LIM_SELF, COND).

Output: descriptives/psid/figures/main/fig_health_scale.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import FIG, PSID  # noqa: E402
from _style import BLUE, INK, INK2, INK3, apply_style  # noqa: E402


def main() -> int:
    apply_style()
    scores = pd.read_parquet(PSID / "measures" / "psid_health_measure.parquet", columns=["theta", "h"]).dropna()
    items = pd.read_csv(PSID / "measures" / "psid_health_measure_items.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax = axs[0]
    o = scores.sort_values("theta")
    ax.plot(o["theta"], o["h"], color=INK, lw=1.8)
    ax.set_xlim(-3.2, 2.2); ax.set_xlabel(r"$\theta$"); ax.set_ylabel("$h$")
    ax.set_title(r"(a) $h$ is the expected score at $\theta$", loc="left")
    tw = ax.twinx()
    tw.hist(scores["theta"], bins=80, color=BLUE, alpha=0.18, density=True)
    tw.set_yticks([]); tw.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax = axs[1]
    for _, r in items.iterrows():
        b = r[[c for c in items.columns if c.startswith("b")]].dropna().to_numpy(float)
        ax.plot(b, np.full(len(b), r["a"]), marker="|", ms=9, lw=0.8, color=INK3)
        ax.annotate(r["item"], (b.max(), r["a"]), textcoords="offset points", xytext=(6, -2.5), ha="left", fontsize=7.6, color=INK2)
    ax.set_xlim(-5.1, 3.4); ax.set_xlabel("threshold location"); ax.set_ylabel("discrimination $a$")
    ax.set_title("(b) where each item speaks", loc="left"); ax.grid(True, axis="y")
    fig.text(0.01, -0.03,
             f"PSID, {len(scores):,} person-waves, 2003-2023. Left: the expected-score curve that maps theta to h, with the pooled distribution of theta behind it. "
             "Right: each testlet's fitted\ndiscrimination against its category thresholds (five testlets asked in every wave).", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig_health_scale.png")
    print(items[["item", "categories", "a"]].round(2).to_string(index=False))
    print("theta share at max:", round(float((scores["theta"] >= scores["theta"].max() - 1e-9).mean()), 3), "h quantiles", scores["h"].quantile([.1, .5, .9]).round(3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
