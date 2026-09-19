"""A first pass at the returns to prevention and treatment from the types.

Types come from partial K-means (default) or, with `ssm` or `base` on the
command line, from the Bayesian mixtures on the health contract
(32_paper_bayes_fits.py): the class shares are the fitted mixing weights and
the type paths the fitted quadratics in each variant's units, on the contract
people. Outputs then carry the suffix: fig_returns_ssm.png, tab_returns_ssm.tex.

The framework's formulas applied to the non-parametric types, for each of
the three variants of the measure mapped to [0, 1] (h and the deficit index
as they are; theta rescaled affinely so the sample minimum is 0 and maximum
1). With c(h) = (1 - h)^2 and a horizon at age 90, for a start age r:

  T(r)   = 2 sum_{a>=r} [ (1 - hbar_a)^2 + Var(h_a) ]                 treatment
  P_j(r) = 2 sum_{a>=r} (1 - hbar_{j,a}) (hbar_{j,r-1} - hbar_{j,a})  prevention, type j
  P(r)   = sum_j pi_j P_j(r)                                          uniform prevention
         = mean channel + between-type covariance channel
  type targeting: max_j P_j(r) - P(r), and the spread sd_j(P_j(r))

Population moments are on the clustered people; type paths are the smoothed
cluster means (five-year rolling means, ends extended linearly). No survival
weighting and no discounting: the sum runs to 90 for everyone.

Outputs: paper/figures/fig_returns.png, paper/tables/tab_returns.tex,
         artifacts/descriptives/paper_returns.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import (  # noqa: E402
    AGES, DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, TAB, VARIANTS, load_labels, load_measure, load_trajectories,
    smooth_path, write_table,
)
from _style import BLUE, CLUSTER, INK2, ORANGE, VERM, apply_style  # noqa: E402

R_GRID = np.arange(25, 86)
R_TABLE = [30, 50, 70]
TYPES = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("ssm", "base") else "kmeans"
SUFFIX = "" if TYPES == "kmeans" else f"_{TYPES}"
TYPE_LAB = {"kmeans": "K-means types", "ssm": "AR(1) plus measurement error classes", "base": "baseline mixture classes"}[TYPES]


def bayes_paths(v: str, A: np.ndarray):
    """Class shares and paths (variant units, on every age) from the fitted mixture."""
    import json
    from prevention_health_clustering.config import PROCESSED_DATA_DIR
    cl = pd.read_csv(DESC / f"paper_bayes_{TYPES}_classes.csv")
    cl = cl[cl["variant"] == v].sort_values("class")
    mom = json.load(open(PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1" / "manifest.json"))["metric_moments"][v]
    paths = np.vstack([mom["mean"] + mom["sd"] * (r["alpha"] + r["beta"] * A + r["gamma"] * A ** 2) for _, r in cl.iterrows()])
    pids = pd.read_parquet(DESC / f"paper_bayes_{TYPES}_posteriors.parquet", columns=["pidp", "variant"])
    return cl["share_bayes"].to_numpy(), paths, pids.loc[pids["variant"] == v, "pidp"].to_numpy()


def main() -> int:
    apply_style()
    d = load_measure()
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9), gridspec_kw={"hspace": 0.4, "wspace": 0.28})
    recs, table = [], []
    for j, (v, lab) in enumerate(VARIANTS):
        A = (AGES - 55) / 10
        if TYPES == "kmeans":
            L = load_labels(v)
            s = d[d["pidp"].isin(L.index)].copy()
            traj = load_trajectories(v)
            pi = L.value_counts(normalize=True).sort_index().reindex(range(K)).to_numpy()
        else:
            pi, bpaths, pids = bayes_paths(v, A)
            s = d[d["pidp"].isin(pids)].copy()
        if v == "theta":                       # affine map to [0, 1] on the clustered person-waves
            lo, hi = s[v].min(), s[v].max()
            s[v] = (s[v] - lo) / (hi - lo)
            if TYPES == "kmeans":
                traj["mean"] = (traj["mean"] - lo) / (hi - lo)
            else:
                bpaths = (bpaths - lo) / (hi - lo)
        prof = s.groupby("age")[v].agg(["mean", "var", "count"])
        hbar = smooth_path(prof.index.to_numpy(), prof["mean"].to_numpy(), prof["count"].to_numpy(), min_count=100)
        var = smooth_path(prof.index.to_numpy(), prof["var"].to_numpy(), prof["count"].to_numpy(), min_count=100)
        if TYPES == "kmeans":
            paths = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(), min_count=25).to_numpy()
                               for c in range(K) for t in [traj[traj["cluster"] == c]]])
        else:
            paths = bpaths
        hb, va = hbar.to_numpy(), var.to_numpy()
        for r in R_GRID:
            idx = slice(r - MIN_AGE, MAX_AGE - MIN_AGE + 1)
            T = 2 * np.sum((1 - hb[idx]) ** 2 + va[idx])
            gap = 1 - paths[:, idx]                                   # K x ages
            prevd = paths[:, r - 1 - MIN_AGE][:, None] - paths[:, idx]  # preventable decline
            Pj = 2 * np.sum(gap * prevd, axis=1)
            P = float(pi @ Pj)
            mean_ch = 2 * np.sum((1 - hb[idx]) * (hb[r - 1 - MIN_AGE] - hb[idx]))
            recs.append({"variant": v, "r": r, "T": T, "P": P, "P_over_T": P / T, "mean_channel": mean_ch,
                         "cov_channel": P - mean_ch, "P_max_type": Pj.max(), "sd_Pj": float(np.sqrt(pi @ (Pj - P) ** 2)),
                         "type_gain": Pj.max() - P, **{f"P_{c + 1}": Pj[c] for c in range(K)},
                         **{f"pi_{c + 1}": pi[c] for c in range(K)}})
        R = pd.DataFrame([x for x in recs if x["variant"] == v]).set_index("r")
        ax = axes[0, j]
        for c in range(K):
            ax.plot(R.index, R[f"P_{c + 1}"], color=CLUSTER[c], lw=1.8, label=f"type {c + 1} ({pi[c]:.0%})")
        ax.plot(R.index, R["P"], color=INK2, lw=1.4, ls="--", label="uniform $P(r)$")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{lab}: return to prevention by type, $P_j(r)$" + ("" if TYPES == "kmeans" else f"\n{TYPE_LAB}"), loc="left", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.5); ax.grid(True, axis="y")
        ax = axes[1, j]
        ax.plot(R.index, R["T"], color=VERM, lw=1.8, label="treatment $T(r)$")
        ax.plot(R.index, R["P"], color=BLUE, lw=1.8, label="uniform prevention $P(r)$")
        ax.plot(R.index, R["mean_channel"], color=BLUE, lw=1.2, ls=":", label="  of which mean channel")
        ax.set_yscale("log")
        ax.set_title("returns by start age (log scale)", loc="left", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=7.5); ax.grid(True, axis="y")
        ax = axes[2, j]
        ax.plot(R.index, R["P_over_T"], color=ORANGE, lw=1.8, label="$P(r)/T(r)$")
        ax.plot(R.index, R["type_gain"] / R["P"], color=CLUSTER[0], lw=1.8,
                label=r"$(\max_j P_j - P)/P$: type targeting")
        ax.set_title("ratios", loc="left", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.5); ax.grid(True, axis="y"); ax.set_xlabel("start age $r$")
        for r in R_TABLE:
            x = R.loc[r]
            table.append([lab if r == R_TABLE[0] else "", str(r), f"{x['T']:.2f}", f"{x['P']:.3f}",
                          f"{x['P_over_T']:.3f}", f"{x['cov_channel'] / x['P']:.2f}",
                          f"{x['P_1']:.3f}", f"{x['P_2']:.3f}", f"{x['P_3']:.3f}", f"{x['type_gain'] / x['P']:.2f}"])
    fig.text(0.01, -0.01,
             f"K = 3 {TYPE_LAB}" + (" from partial K-means on the clustered people" if TYPES == "kmeans" else
                                   " on the health contract, shares and quadratic paths at the posterior mean")
             + ". c(h) = (1 - h)^2, horizon age 90, no survival "
             "weighting. theta is mapped affinely to [0, 1] by its sample range.\n"
             + ("Type paths are five-year rolling means of the cluster means, ends extended linearly; " if TYPES == "kmeans"
                else "Type paths are the fitted quadratics in the variant's units; ")
             + "the population mean and variance are five-year rolling means. P_j(r) uses the type's own path from r - 1.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / f"fig_returns{SUFFIX}.png")
    write_table(TAB / f"tab_returns{SUFFIX}.tex",
                ["variant", "$r$", "$T(r)$", "$P(r)$", "$P/T$", "cov.\\ channel share", "$P_1$", "$P_2$", "$P_3$",
                 "type gain $/P$"], table, colspec="llrrrrrrrr")
    out = pd.DataFrame(recs)
    out.to_csv(DESC / f"paper_returns{SUFFIX}.csv", index=False)
    print(out[out["r"].isin(R_TABLE)].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
