"""The frailty index and log frailty under AR(1) + "spike" by number of classes.

Rows are K = 3, 4 and 5 (a row is drawn when at least one of the two measures has the fit);
columns are the class paths and the posterior class composition by age for the 31-deficit
frailty index and for log(frailty + 1/31). K = 3 comes from artifacts/health-frailty/,
K = 4 and 5 from artifacts/health-k45/ (clustering/runs/health_k45_queue.py --frailty),
read with _paper_common.bayes_fit over the chains that share the better mode
(_paper_common.MODE_CHAINS). Class 1 is the frailest in every panel; line width is
proportional to the class share and the legend gives each class's persistence. A fit that
has not finished leaves its panels blank and says so.

Outputs: paper/figures/fig_frailty_by_K.png, artifacts/descriptives/paper_frailty_by_K.csv.

    PYTHONPATH=src .venv/bin/python descriptives/51_paper_frailty_by_K.py
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
from _paper_common import AGES, DESC, FIG, MAX_AGE, MIN_AGE, MODE_CHAINS, bayes_fit  # noqa: E402
from _style import INK2, INK3, apply_style, share_ylim  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

PALETTE = importlib.import_module("39_paper_multidim").PALETTE
MEASURES = [("frailty", "frailty index"), ("logfrailty", "log(frailty + 1/31)")]
KS = (3, 4, 5)
A = (AGES - 55) / 10


def fit_dir(v: str, k: int) -> Path:
    return ARTIFACTS_DIR / "health-frailty" / f"health-{v}-ssm" if k == 3 else ARTIFACTS_DIR / "health-k45" / f"health-{v}-ssm-k{k}"


def main() -> int:
    apply_style()
    fits = {}
    for v, _ in MEASURES:
        for k in KS:
            d = fit_dir(v, k)
            if (d / "run_summary.json").exists():
                f = bayes_fit(d, v, k, chains=MODE_CHAINS.get(d.name))
                f["paths"] = np.vstack([f["mom"]["mean"] + f["mom"]["sd"] * (c[0] + c[1] * A + c[2] * A ** 2) for c in f["coef"]])
                fits[(v, k)] = f
    ks = [k for k in KS if any((v, k) in fits for v, _ in MEASURES)]
    fig, axes = plt.subplots(len(ks), 4, figsize=(15, 3.4 * len(ks)), squeeze=False,
                             gridspec_kw={"wspace": 0.28, "hspace": 0.38, "width_ratios": [1.3, 0.75, 1.3, 0.75]})
    rows = []
    for r, k in enumerate(ks):
        for j, (v, lab) in enumerate(MEASURES):
            ax, axc = axes[r, 2 * j], axes[r, 2 * j + 1]
            f = fits.get((v, k))
            if f is None:
                for a in (ax, axc):
                    a.text(0.5, 0.5, f"K = {k} on {lab}:\nnot yet fitted", ha="center", va="center", fontsize=9, color=INK3, transform=a.transAxes)
                    a.set_xticks([]); a.set_yticks([])
                continue
            CL = PALETTE[k]
            for c in range(k):
                ax.plot(AGES, f["paths"][c], color=CL[c], lw=1.0 + 3.5 * f["theta"][c], label=f"{c + 1}: {f['theta'][c]:.0%}, $\\rho$ {f['rho'][c]:.2f}")
                rows.append({"measure": v, "K": k, "class": c + 1, "share": f["theta"][c], "rho": f["rho"][c], "level_20": f["paths"][c][0],
                             "level_55": f["paths"][c][35], "level_90": f["paths"][c][-1], "rhat": f["rhat"], "chains": "all" if f["chains"] is None else ",".join(map(str, f["chains"]))})
            ax.legend(fontsize=6.8, loc="upper left", ncols=1 if k == 3 else 2, handlelength=1.6, columnspacing=0.8)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylabel(f"{lab}, K = {k}", fontsize=9)
            comp = f["comp"]
            axc.stackplot(comp.index, *[comp[c] for c in comp.columns], colors=CL, alpha=0.9)
            axc.set_ylim(0, 1); axc.set_xlim(MIN_AGE, MAX_AGE); axc.set_ylabel("share observed", fontsize=8)
            if r == 0:
                ax.set_title(f"{lab}: class paths (class 1 frailest)", loc="left", fontsize=10)
                axc.set_title("composition by age", loc="left", fontsize=10)
            if r == len(ks) - 1:
                ax.set_xlabel("age"); axc.set_xlabel("age")
    for j in range(2):
        share_ylim([axes[r, 2 * j] for r in range(len(ks)) if axes[r, 2 * j].lines])
    pd.DataFrame(rows).to_csv(DESC / "paper_frailty_by_K.csv", index=False)
    chains_note = "; ".join(f"{fit_dir(v, k).name}: chains {','.join(map(str, f['chains']))}" for (v, k), f in sorted(fits.items()) if f["chains"] is not None)
    footer = ("AR(1) + \"spike\" mixtures on the health contract's frailty columns, K = 3 (artifacts/health-frailty) and K = 4, 5 (artifacts/health-k45), class paths at the posterior mean in the measure's own units, "
              "line width proportional to the class share, the legend giving each class's persistence; the strip is the posterior class composition of the person-waves observed at each age. Classes ordered frailest first. "
              "Each measure's path panels share one vertical scale across rows."
              + (f" Fits read over the chains that share the better mode: {chains_note}." if chains_note else ""))
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(footer, 230)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_frailty_by_K.png", bbox_inches="tight")
    for (v, k), f in sorted(fits.items()):
        print(f"{v} K = {k}: shares {np.round(f['theta'], 3)}, rho {np.round(f['rho'], 2)}, rhat {f['rhat']:.3f}, chains {f['chains'] or 'all'}")
    print(f"wrote fig_frailty_by_K.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
