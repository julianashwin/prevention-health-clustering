"""The worst class on the age windows, with credible bands: K = 3 and K = 4.

A slimmed companion to the windows figure (50_paper_windows_bayes.py, its panels (a) and (b)):
only each window's worst class, its quadratic path in theta drawn at the posterior median with
a 95% credible band from the draws, against the lifecycle 20-90 fit's worst class in black.
Rows K = 3 and K = 4; columns the sliding twenty-year windows and the windows expanding
backwards from 90. The worst class is the one with the lowest mean path over the window's own
ages at the posterior mean; the draws are read over the same chains as 50 (WINDOW_CHAINS).

Output: paper/figures/fig_windows_worst_ci.png and artifacts/descriptives/paper_windows_worst_ci.csv
(median, band and the dotted observed mean by window, K and age). Dotted: the worst class's
posterior-weighted observed mean of theta by age, smoothed with a centred three-year rolling mean.

    PYTHONPATH=src .venv/bin/python descriptives/55_paper_windows_worst_ci.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import CONTRACT, DESC, FIG, MAX_AGE, MIN_AGE, bayes_fit, observed_class_means  # noqa: E402
from _style import INK, INK2, apply_style, share_ylim  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(70, 90), (60, 90), (50, 90), (40, 90), (30, 90)]
WIN = ARTIFACTS_DIR / "health-windows"
V = "theta"
WINDOW_CHAINS = {"theta-ssm-k4-60-90": [1, 2, 3], "theta-ssm-k4-70-90": [1, 4]}   # as 50: the chains that share the better mode


def fit_dir(k: int, lo: int, hi: int) -> Path:
    if (lo, hi) == (MIN_AGE, MAX_AGE):
        return ARTIFACTS_DIR / "health-ssm" / f"health-{V}-ssm" if k == 3 else ARTIFACTS_DIR / "health-k45" / f"health-{V}-ssm-k{k}"
    return WIN / (f"{V}-ssm-{lo}-{hi}" if k == 3 else f"{V}-ssm-k{k}-{lo}-{hi}")


def contract_of(lo: int, hi: int) -> Path:
    return CONTRACT if (lo, hi) == (MIN_AGE, MAX_AGE) else PROCESSED_DATA_DIR / "contracts" / f"health_window_{lo}_{hi}_v1"


def worst_paths(k: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws (n_draws x ages) of the worst class's path in theta's units over the window, and the
    worst class's posterior-weighted observed mean by age (three-year centred rolling mean)."""
    d = fit_dir(k, lo, hi)
    contract = contract_of(lo, hi)
    mom = json.load(open(contract / "manifest.json"))["metric_moments"][V]
    chains = WINDOW_CHAINS.get(d.name)
    files = [f for f in sorted((d / "chains").glob("*_[0-9].csv")) if chains is None or int(f.stem.rsplit("_", 1)[1]) in chains]
    cols = [f"coef.1.{c}.{j}" for c in range(1, k + 1) for j in (1, 2, 3)]
    draws = pd.concat([pd.read_csv(f, comment="#", usecols=cols)[cols] for f in files]).to_numpy().reshape(-1, k, 3)
    ages = np.arange(lo, hi + 1); A = (ages - 55) / 10
    paths = mom["mean"] + mom["sd"] * (draws[:, :, :1] + draws[:, :, 1:2] * A + draws[:, :, 2:3] * A ** 2)   # draws x K x ages
    worst = int(np.argmin(paths.mean(axis=0).mean(axis=1)))
    r = bayes_fit(d, V, k, chains=chains, contract=None if (lo, hi) == (MIN_AGE, MAX_AGE) else contract)
    pm = (r["coef"][:, :1] + r["coef"][:, 1:2] * A + r["coef"][:, 2:3] * A ** 2).mean(axis=1)
    assert int(np.argmin(pm)) == worst, f"{d.name}: worst class {int(np.argmin(pm)) + 1} at the posterior mean, {worst + 1} from the draws"
    obs = observed_class_means(r["post"], V, k, rows=r["obs_rows"])[f"class{worst + 1}"].reindex(ages)
    obs = obs.rolling(3, center=True, min_periods=1).mean().to_numpy()
    return ages, paths[:, worst, :], obs


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, gridspec_kw={"wspace": 0.12, "hspace": 0.28})
    cmap = plt.get_cmap("viridis")
    rows = []
    for i, k in enumerate((3, 4)):
        la, lp, lo_obs = worst_paths(k, MIN_AGE, MAX_AGE)
        for j, (wins, name) in enumerate([(SLIDING, "sliding twenty-year windows"), (BACKWARD, "windows expanding backwards from 90")]):
            ax = axes[i, j]
            lq = np.quantile(lp, [0.025, 0.5, 0.975], axis=0)
            ax.fill_between(la, lq[0], lq[2], color=INK, alpha=0.12, lw=0, zorder=1)
            ax.plot(la, lq[1], color=INK, lw=2.0, zorder=4, label="lifecycle 20-90")
            ax.plot(la, lo_obs, color=INK, lw=1.3, ls=":", zorder=5)
            for n, (lo, hi) in enumerate(wins):
                ages, p, obs = worst_paths(k, lo, hi)
                q = np.quantile(p, [0.025, 0.5, 0.975], axis=0)
                c = cmap(0.9 * n / max(len(wins) - 1, 1))
                ax.fill_between(ages, q[0], q[2], color=c, alpha=0.25, lw=0, zorder=2)
                ax.plot(ages, q[1], color=c, lw=1.6, zorder=3, label=f"{lo}-{hi}")
                ax.plot(ages, obs, color=c, lw=1.3, ls=":", zorder=5)
                rows += [{"K": k, "kind": "sliding" if j == 0 else "backward", "lo": lo, "hi": hi, "age": int(a), "median": m, "lo95": a0, "hi95": a1, "observed": o}
                         for a, m, a0, a1, o in zip(ages, q[1], q[0], q[2], obs)]
            ax.set_title(f"({'abcd'[2 * i + j]}) K = {k}, {name}", loc="left", fontsize=9.5)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if i == 1:
                ax.set_xlabel("age")
            if j == 0:
                ax.set_ylabel("$\\theta$, worst class")
            ax.legend(fontsize=7, ncols=2, loc="upper right")
        rows += [{"K": k, "kind": "lifecycle", "lo": MIN_AGE, "hi": MAX_AGE, "age": int(a), "median": m, "lo95": a0, "hi95": a1, "observed": o}
                 for a, m, a0, a1, o in zip(la, *np.quantile(lp, [0.5, 0.025, 0.975], axis=0), lo_obs)]
    share_ylim(axes.ravel())
    footer = ("The worst class of the AR(1) + \"spike\" mixture on theta refitted on each age window (colours) and of the lifecycle 20-90 fit (black), "
              "at K = 3 (top) and K = 4 (bottom): the class path at the posterior median with a 95% credible band from the posterior draws of its quadratic coefficients. "
              "The worst class is the one with the lowest mean path over the window's own ages. "
              "Dotted: the worst class's posterior-weighted observed mean by age (three-year rolling mean; ages with under 25 people of summed weight left out). The K = 4 70-90 and 60-90 fits are read over the chains that share the better mode (1, 4 and 1-3).")
    fig.text(0.01, 0.0, "\n".join(textwrap.wrap(footer, 170)), fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_windows_worst_ci.png", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(DESC / "paper_windows_worst_ci.csv", index=False)
    print(f"wrote fig_windows_worst_ci.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
