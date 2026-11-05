"""Figure 7 on PSID, adapted: type paths and shares by number of types, partial K-means.

The paper's Figure 7 (descriptives/36_paper_trajectories.py) draws Bayesian growth mixtures at
K = 3, 4 and 5. Without Bayesian fits, this draws the partial K-means types on PSID theta at
K = 3, 4 and 5 (22_psid_kmeans.py: theta, theta_k4, theta_k5) in one row, laid out as the paper's
panels: type means where at least 25 members are observed, line width proportional to the type
share, the share in the legend, and the type composition of the person-waves observed at each age
as a strip beneath; one vertical scale for all three.

Outputs: descriptives/psid/figures/main/fig_class_trajectories_by_K.png,
artifacts/descriptives/psid/trajectories_by_K.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, FIG, MAX_AGE, MIN_AGE, load_composition, load_labels, load_trajectories  # noqa: E402
from _style import INK2, apply_style, share_ylim  # noqa: E402

PALETTE = importlib.import_module("36_paper_trajectories").PALETTE
KS = {3: "theta", 4: "theta_k4", 5: "theta_k5"}
MIN_SUPPORT = 25


def main() -> int:
    apply_style()
    fig = plt.figure(figsize=(13.5, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 0.55], hspace=0.12, wspace=0.16, top=0.92, bottom=0.12)
    axes, rows = [], []
    for c, (Kc, var) in enumerate(KS.items()):
        ax = fig.add_subplot(gs[0, c]); axc = fig.add_subplot(gs[1, c])
        L = load_labels(var); tr = load_trajectories(var); share = L.value_counts(normalize=True).sort_index()
        cols = PALETTE[Kc]
        for k in range(Kc):
            t = tr[(tr["cluster"] == k) & (tr["count"] >= MIN_SUPPORT)]
            ax.plot(t["age"], t["mean"], color=cols[k], lw=1.0 + 3.5 * share[k], label=f"{k + 1}: {share[k]:.0%}")
            for a in (30, 55, 80):
                q = tr[(tr["cluster"] == k) & (tr["age"] == a)]
                rows.append({"K": Kc, "type": k + 1, "share": share[k], "age": a, "mean": float(q["mean"].iloc[0]) if len(q) else np.nan})
        ax.set_title(f"$K = {Kc}$", fontsize=11, loc="center")
        ax.legend(fontsize=6.6, loc="lower left", ncols=1 if Kc == 3 else 2, handlelength=1.6, columnspacing=0.8)
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(r"$\theta$, partial K-means", fontsize=9)
        comp = load_composition(var).pivot(index="age", columns="cluster", values="share")
        axc.stackplot(comp.index, *[comp[k] for k in range(Kc)], colors=cols, alpha=0.9)
        axc.set_ylim(0, 1); axc.set_yticks([]); axc.set_xlim(MIN_AGE, MAX_AGE); axc.spines["left"].set_visible(False); axc.set_xlabel("age")
        axes.append(ax)
    share_ylim(axes)
    n = len(load_labels("theta"))
    fig.text(0.01, 0.0,
             f"Partial K-means on PSID theta (the PSID contract: {n:,} people with at least three observed ages 20-89), 50 seeded starts, at K = 3, 4 and 5; type 1 is worst health. "
             "Lines are the type means where at least 25 members\nare observed, width proportional to the type share (in the legend). "
             "The strip beneath each panel is the type composition of the person-waves observed at each age. The three panels share one vertical scale. "
             "The paper's figure draws\nBayesian growth mixtures (independent residuals and AR(1) plus \"spike\") instead; no Bayesian model is fitted on PSID.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_class_trajectories_by_K.png", bbox_inches="tight")
    t = pd.DataFrame(rows); t.to_csv(DESC / "trajectories_by_K.csv", index=False)
    print(t.pivot_table(index=["K", "type", "share"], columns="age", values="mean").round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
