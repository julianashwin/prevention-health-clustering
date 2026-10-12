"""The multidimensional K = 5 fit chain by chain.

The four chains of theta-ssm-mort-k5 (AR(1) + "spike" + Gompertz-Makeham mortality, K = 5)
each settled at a different mode. This draws each chain as its own row of the
multidimensional figure (physical and mental class paths, class hazards, composition by
age), with the chain's mean log posterior in the row label, so the modes can be compared.

Output: paper/figures/fig_multidim_k5_chains.png,
        artifacts/descriptives/paper_multidim_k5_chains.csv.

    PYTHONPATH=src .venv/bin/python descriptives/49_paper_multidim_k5_chains.py
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, DESC, FIG, MAX_AGE, MIN_AGE  # noqa: E402
from _style import INK2, apply_style  # noqa: E402

M39 = importlib.import_module("39_paper_multidim")
TAG, KC = "theta-ssm-mort-k5", 5
CHAINS = [1, 2, 3, 4]


def main() -> int:
    apply_style()
    fits = [M39.load_fit(TAG, "theta", [c], kc=KC) for c in CHAINS]
    order = np.argsort([-f["lps"][0] for f in fits])            # best log posterior first
    CL = M39.PALETTE[KC]
    fig, axes = plt.subplots(len(CHAINS), 4, figsize=(15, 3.3 * len(CHAINS)), squeeze=False,
                             gridspec_kw={"wspace": 0.28, "hspace": 0.35, "width_ratios": [1, 1, 1, 0.8]})
    rows = []
    for r, i in enumerate(order):
        f = fits[i]; chain = CHAINS[i]
        for c in range(2):
            ax = axes[r, c]
            for k in range(KC):
                ax.plot(AGES, f["paths"][c][k], color=CL[k], lw=1.0 + 3.5 * f["share"][k], label=f"{k + 1}: {f['share'][k]:.0%}, $\\rho$ {f['rho'][k, c]:.2f}")
                obs = f["observed"][c][f"class{k + 1}"]
                ax.plot(obs.index, obs.to_numpy(), color=CL[k], lw=1.0, ls=":")
            ax.legend(fontsize=6.5, loc="lower left" if c == 0 else "lower right", handlelength=1.6)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if r == 0:
                ax.set_title(["physical channel", "mental channel"][c], fontsize=10, loc="left")
            ax.set_ylabel(f"$\\theta$, chain {chain} (lp {f['lps'][0]:,.0f})" if c == 0 else r"mental $\theta$", fontsize=9)
        ax = axes[r, 2]
        for k in range(KC):
            ax.plot(AGES, f["hazard"][k], color=CL[k], lw=1.0 + 3.5 * f["share"][k], label=f"{k + 1}: slope {f['slope'][k]:.3f}")
            xs, ys = zip(*f["crude"][f"class{k + 1}"])
            ax.plot(xs, ys, color=CL[k], lw=0, marker="o", ms=3.5, mfc="none")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 0.3); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        ax.legend(fontsize=6.5, loc="upper left", handlelength=1.6); ax.set_ylabel("yearly hazard", fontsize=9)
        if r == 0:
            ax.set_title("mortality: Gompertz-Makeham hazard", fontsize=10, loc="left")
        ax = axes[r, 3]
        comp = f["comp"]
        ax.stackplot(comp.index, *[comp[c] for c in comp.columns], colors=CL, alpha=0.9)
        ax.set_ylim(0, 1); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylabel("share observed", fontsize=9)
        if r == 0:
            ax.set_title("class composition by age", fontsize=10, loc="left")
        if r == len(CHAINS) - 1:
            for ax in axes[r]:
                ax.set_xlabel("age")
        for k in range(KC):
            rows.append({"chain": chain, "lp": f["lps"][0], "rhat_within": f["rhat"], "class": k + 1, "share": f["share"][k],
                         "rho_phys": f["rho"][k, 0], "rho_ment": f["rho"][k, 1], "phys_20": f["paths"][0][k][0], "phys_55": f["paths"][0][k][35],
                         "phys_90": f["paths"][0][k][-1], "ment_20": f["paths"][1][k][0], "ment_90": f["paths"][1][k][-1],
                         "hazard_55": f["hazard"][k][35], "gomp_slope": f["slope"][k]})
    share_cols = [[axes[r, c] for r in range(len(CHAINS))] for c in range(2)]
    for col in share_cols:
        lo = min(ax.get_ylim()[0] for ax in col); hi = max(ax.get_ylim()[1] for ax in col)
        for ax in col:
            ax.set_ylim(lo, hi)
    pd.DataFrame(rows).to_csv(DESC / "paper_multidim_k5_chains.csv", index=False)
    footer = (f"The multidimensional K = 5 AR(1) + \"spike\" fit with Gompertz-Makeham mortality ({fits[0]['n_person']:,} people), one row per chain, best mean log posterior first: "
              "the four chains settled at four different modes, so the rows are four different five-class solutions of the same likelihood. Columns as in the multidimensional figure: class paths at each "
              "chain's posterior mean (dotted: the observed class mean at each age, posterior-weighted), the class hazards (circles: crude death rates by five-year band), the class composition by age. "
              "Classes ordered worst to best on the physical channel within each chain; one vertical scale per channel across rows.")
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(footer, 230)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_multidim_k5_chains.png", bbox_inches="tight")
    for r, i in enumerate(order):
        f = fits[i]
        print(f"chain {CHAINS[i]}: lp {f['lps'][0]:,.1f}, within-chain rhat {f['rhat']:.3f}, shares {np.round(f['share'], 3)}, "
              f"phys at 55 {np.round(f['paths'][0][:, 35], 2)}, hazard ratio 1/5 at 55 {f['hazard'][0][35] / f['hazard'][-1][35]:.1f}x")
    print(f"wrote fig_multidim_k5_chains.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
