"""The worst type by method and by number of classes, on theta.

Partial K-means at K = 3, 4, 5 (22_paper_kmeans.py; the K = 4 and 5 runs are the variants
theta_k4, theta_k5 of 22 --k 4,5) against the univariate AR(1) + "spike" mixture at K = 3, 4, 5
(artifacts/health-ssm, health-k45). For each method and K: the worst type's share (and for the
mixture its probability mass), its path at 20, 50 and 80, its slopes, the gap to the next type,
the within-type spread at 50 (K-means: the sd of theta among its members at 48-52; mixture:
the stationary within-class sd), the mixture's persistence and the share of its modal members
with a posterior above 0.9, the share of the K = 3 worst type still worst at this K, and the
share of this method's worst type that is worst under the other method at the same K.

Outputs: paper/figures/fig_worst_type_by_K.png, paper/tables/tab_worst_type_by_K.tex,
         artifacts/descriptives/paper_worst_type_by_K.csv.

    PYTHONPATH=src .venv/bin/python descriptives/53_paper_worst_type.py
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
from _paper_common import AGES, DESC, FIG, MAX_AGE, MIN_AGE, MODE_CHAINS, TAB, bayes_fit, load_labels, load_measure, smooth_path  # noqa: E402
from _style import INK2, INK3, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

PALETTE = importlib.import_module("39_paper_multidim").PALETTE
KS = (3, 4, 5)
A = (AGES - 55) / 10


def km_fit(k: int, d: pd.DataFrame) -> dict:
    L = load_labels("theta" if k == 3 else f"theta_k{k}")
    s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    cell = s.groupby(["cluster", "age"])["theta"].agg(["mean", "std", "count"]).reset_index()
    paths = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(), min_count=25, window=10).to_numpy()
                       for c in range(k) for t in [cell[cell["cluster"] == c]]])
    return {"paths": paths, "share": L.value_counts(normalize=True).sort_index().reindex(range(k)).to_numpy(), "modal": L,
            "sd50": float(cell[(cell["cluster"] == 0) & cell["age"].between(48, 52)]["std"].mean()), "rho": None, "certain": np.nan, "pi": np.nan}


def ssm_fit_k(k: int) -> dict:
    fd = ARTIFACTS_DIR / ("health-ssm" if k == 3 else "health-k45") / ("health-theta-ssm" if k == 3 else f"health-theta-ssm-k{k}")
    r = bayes_fit(fd, "theta", k, chains=MODE_CHAINS.get(fd.name))
    m, sd = r["mom"]["mean"], r["mom"]["sd"]
    P = r["post"][[f"class{j + 1}" for j in range(k)]].to_numpy()
    modal = pd.Series(P.argmax(1), index=r["post"]["pidp"])
    return {"paths": np.vstack([m + sd * (c[0] + c[1] * A + c[2] * A ** 2) for c in r["coef"]]), "share": r["theta"], "modal": modal, "pi": float(r["theta"][0]),
            "rho": r["rho"], "certain": float((P[modal.to_numpy() == 0, 0] > 0.9).mean()),
            "sd50": float(sd * np.sqrt(r["sigma"] ** 2 / (1 - r["rho"][0] ** 2) + r["sigma_meas"] ** 2))}


def main() -> int:
    apply_style()
    d = load_measure(["theta"])
    fits = {("K-means", k): km_fit(k, d) for k in KS}
    fits.update({('AR(1) + "spike"', k): ssm_fit_k(k) for k in KS})
    rows = []
    for (meth, k), f in fits.items():
        p = f["paths"]; worst = f["modal"] == 0
        base = fits[(meth, 3)]["modal"]; both = worst.index.intersection(base.index)
        retained = float((worst.loc[both] & (base.loc[both] == 0)).sum() / (base.loc[both] == 0).sum())
        other = fits[('AR(1) + "spike"' if meth == "K-means" else "K-means", k)]["modal"]; b2 = worst.index.intersection(other.index)
        overlap = float((worst.loc[b2] & (other.loc[b2] == 0)).sum() / worst.loc[b2].sum())
        rows.append({"method": meth, "K": k, "share_modal": float(worst.mean()), "share_mass": f["pi"], "theta_20": p[0][0], "theta_50": p[0][30], "theta_80": p[0][60],
                     "slope_20_50": (p[0][30] - p[0][0]) / 30, "slope_50_80": (p[0][60] - p[0][30]) / 30, "gap_next_50": p[1][30] - p[0][30], "gap_next_80": p[1][60] - p[0][60],
                     "within_sd_50": f["sd50"], "rho": f["rho"][0] if f["rho"] is not None else np.nan, "certain_members": f["certain"],
                     "k3_worst_retained": retained, "worst_under_other_method": overlap})
    T = pd.DataFrame(rows); T.to_csv(DESC / "paper_worst_type_by_K.csv", index=False)
    print(T.round(3).to_string(index=False))
    with open(TAB / "tab_worst_type_by_K.tex", "w") as fh:
        fh.write("\\begin{tabular}{llrrrrrrrrr}\n\\toprule\n")
        fh.write("method & $K$ & share & $\\theta$ at 20 & at 50 & at 80 & slope to 50 & within sd at 50 & $\\rho$ & $K=3$ worst retained & worst under the other method \\\\\n\\midrule\n")
        for meth in ("K-means", 'AR(1) + "spike"'):
            for k in KS:
                q = T[(T["method"] == meth) & (T["K"] == k)].iloc[0]
                share = f"{q['share_modal']:.0%}" + (f" ({q['share_mass']:.0%})" if pd.notna(q["share_mass"]) else "")
                fh.write(f"{meth if k == 3 else ''} & {k} & {share} & {q['theta_20']:.2f} & {q['theta_50']:.2f} & {q['theta_80']:.2f} & {q['slope_20_50']:.3f} & {q['within_sd_50']:.2f} & "
                         f"{'' if pd.isna(q['rho']) else f'{q["rho"]:.2f}'} & {q['k3_worst_retained']:.0%} & {q['worst_under_other_method']:.0%} \\\\\n")
            if meth == "K-means":
                fh.write("\\addlinespace\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), gridspec_kw={"hspace": 0.4, "wspace": 0.22}, sharey=True)
    for i, meth in enumerate(("K-means", 'AR(1) + "spike"')):
        for j, k in enumerate(KS):
            f = fits[(meth, k)]; ax = axes[i, j]
            for c in range(k):
                ax.plot(AGES, f["paths"][c], color=PALETTE[k][0] if c == 0 else INK3, lw=2.6 if c == 0 else 1.2, alpha=1 if c == 0 else 0.6,
                        label=(f"worst type ({f['share'][0]:.0%}" + (f", $\\rho$ {f['rho'][0]:.2f})" if f["rho"] is not None else ")")) if c == 0 else None)
            if k > 3:
                ax.plot(AGES, fits[(meth, 3)]["paths"][0], color="#d62728", lw=1.2, ls=":", label="$K=3$ worst type")
            ax.set_title(f"{meth}, $K = {k}$", loc="left", fontsize=10); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.legend(fontsize=7.5, loc="lower left")
            if i == 1:
                ax.set_xlabel("age")
            if j == 0:
                ax.set_ylabel(r"$\theta$")
    footer = ("The worst type by method and K on theta: partial K-means (top; type 1 the lowest person mean; K = 4 and 5 from 22 --k) and the univariate AR(1) + \"spike\" mixture (bottom; class 1 the lowest "
              "anchor intercept; K = 4 and 5 from artifacts/health-k45, chains sharing the better mode). The worst type is drawn in colour, the others in grey; the red dotted line repeats the K = 3 worst type. "
              "K-means paths are ten-year rolling means of the type means; mixture paths the quadratic class paths at the posterior mean. For the mixture the legend's share is the probability mass; the table gives the modal share too.")
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(footer, 200)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_worst_type_by_K.png", bbox_inches="tight")
    print(f"wrote fig_worst_type_by_K.png and tab_worst_type_by_K.tex to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
