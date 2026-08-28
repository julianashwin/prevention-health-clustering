"""Class trajectories and shares for the multidimensional fits.

The univariate fits each carry one channel and are drawn in
08_trajectory_comparison.py. Each multidimensional fit instead carries three,
so one class structure implies three trajectories: latent physical health,
latent mental health, and the expected chronic-condition count. All three are
drawn here, one row per variant, so the shares (identical across a row, since
there is one class structure per fit) can be read against what each class
does on every channel.

The Gaussian channels are returned to their original GRM theta scale using
the standardising moments recorded by the fit; the count channel is plotted
as the implied mean number of conditions.

Outputs: docs/figures/fig_multidim_trajectories.png,
         artifacts/descriptives/multidim_trajectories.csv
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
VARIANTS = [("baseline", "baseline"), ("holdout", "baseline + holdout"),
            ("ar1", "AR(1)"), ("ar1-holdout", "AR(1) + holdout")]
CHANNELS = [
    ("coef[1,{k},{p}]", "theta_phys_func", "latent physical health",
     r"GRM $\theta$, physical"),
    ("coef[2,{k},{p}]", "theta_ment_nodepr", "latent mental health",
     r"GRM $\theta$, mental"),
    ("coef_chronic[{k},{p}]", None, "chronic conditions",
     "expected count"),
]


def main() -> int:
    apply_style()
    rows = []
    comp = pd.read_csv(ARTIFACTS_DIR / "descriptives"
                       / "class_composition_by_age.csv")
    fig, axes = plt.subplots(len(VARIANTS), len(CHANNELS) + 1,
                             figsize=(14.6, 2.9 * len(VARIANTS)), sharex=True)
    for r, (tag, vlabel) in enumerate(VARIANTS):
        summ = json.loads((ARTIFACTS_DIR / "multidim" / tag
                           / "run_summary.json").read_text())
        p, mom = summ["params"], summ["channel_moments"]
        theta = np.array([p[f"theta[{k}]"]["mean"] for k in (1, 2, 3)])
        for c, (pat, chan, title, ylab) in enumerate(CHANNELS):
            ax = axes[r, c]
            for k in (1, 2, 3):
                b = [p[pat.format(k=k, p=j)]["mean"] for j in (1, 2, 3)]
                lin = b[0] + b[1] * A + b[2] * A**2
                if chan is not None:      # back to the original theta scale
                    y = lin * mom[chan]["sd"] + mom[chan]["mean"]
                else:                      # count channel: log link
                    y = np.exp(lin)
                ax.plot(AGES, y, color=CLUSTER[k - 1],
                        lw=0.8 + 4.0 * theta[k - 1],
                        label=f"class {k}: {theta[k-1]:.0%}")
                rows.append({"variant": tag, "channel": title, "class": k,
                             "share": theta[k - 1],
                             "at_30": y[AGES == 30][0], "at_55": y[AGES == 55][0],
                             "at_80": y[AGES == 80][0]})
            if chan is not None:
                ax.axhline(0, color="#cccccc", lw=0.7, ls=":")
            ax.grid(True, axis="y")
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{vlabel}\n{ylab}", fontsize=8.5)
            else:
                ax.set_ylabel(ylab, fontsize=8)
            if r == len(VARIANTS) - 1:
                ax.set_xlabel("age")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower left")
        # fourth column: the observed class composition, a row-level property
        axc = axes[r, len(CHANNELS)]
        g = comp[comp["fit"] == f"multidim-{tag}"].sort_values("age")
        axc.stackplot(g["age"], *[g[f"class{k+1}"] for k in (0, 1, 2)],
                      colors=CLUSTER, alpha=0.9)
        axc.set_ylim(0, 1); axc.set_xlim(20, 90)
        axc.set_ylabel("share observed", fontsize=8)
        if r == 0:
            axc.set_title("class composition by age", fontsize=10)
        if r == len(VARIANTS) - 1:
            axc.set_xlabel("age")
    fig.suptitle("Multidimensional fits: one class structure, three channels",
                 fontweight="bold", y=1.0)
    fig.text(0.01, -0.004,
             "One row per variant, one column per channel. Line width is proportional to the class share,\n"
             "which is a property of the fit and so identical across each row. The Gaussian channels are\n"
             "shown on the original GRM theta scale; the count channel as the implied mean number of\n"
             "conditions. Class 1 is worst physical health by the anchor convention. The fourth column\n"
             "is the class composition of the person-waves observed at each age: the shares theta are\n"
             "lifetime constants, but who is in the sample at each age is not.",
             fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "figures" / "fig_multidim_trajectories.png"
    fig.savefig(out)
    tab = pd.DataFrame(rows)
    tab.to_csv(ARTIFACTS_DIR / "descriptives" / "multidim_trajectories.csv",
               index=False)
    print(tab.round(3).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
