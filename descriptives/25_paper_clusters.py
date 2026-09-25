"""Figures 3 and 4: the non-parametric types and their between/within anatomy.

Figure 3 (fig_clusters.png, h and theta; the deficit index and, beside it, the same
three panels for partial K-means on predicted cost go to fig_clusters_fi10_cost.png
for the appendix), one column per variant of the measure:
  top     cluster mean paths, K = 3, with the within-cluster +/- 1 sd band; the
          class shares in the title
  middle  autocorrelation of within-cluster deviations by lag in years, per
          cluster: the persistence of what the types do not explain
  bottom  cluster composition of the person-waves observed at each age

Figure 4 (fig_cluster_decomposition.png, h and theta; the deficit index goes to
fig_cluster_decomposition_fi10.png for the appendix), the clustering version of Exhibit 1:
  (a) the cross-sectional variance at each age split into between-cluster
      (Var of the cluster means, age-specific shares) and within
  (b) Var_j(d_j,a): the between-type variance of the one-year change of the
      smoothed cluster paths, person-level shares
  (c) Cov_j(hbar_j,a, d_j,a): the level-change covariance across types, whose
      sign says whether types are diverging or converging at that age

Outputs also artifacts/descriptives/paper_cluster_decomposition.csv and
paper_cluster_agreement.csv (ARI between the three variants' typologies).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import (  # noqa: E402
    AGES, DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, VARIANTS, load_labels, load_measure, load_predicted_cost_rows,
    load_trajectories, smooth_path,
)
from _style import BLUE, CLUSTER, GREEN, INK2, VERM, apply_style  # noqa: E402

MIN_SUPPORT, MAX_LAG = 25, 9


def main() -> int:
    apply_style()
    d = load_measure()
    labels = {v: load_labels(v) for v, _ in VARIANTS}
    g3 = {"height_ratios": [3.2, 1.9, 0.7], "hspace": 0.35}
    fig3, ax3m = plt.subplots(3, 2, figsize=(9.0, 8.2), gridspec_kw={**g3, "wspace": 0.22})
    fig3b, ax3b = plt.subplots(3, 2, figsize=(9.0, 8.2), gridspec_kw={**g3, "wspace": 0.22})
    ax3 = np.column_stack([ax3m, ax3b])            # columns 2-3: the deficit index and the cost index, on their own figure
    fig4, ax4m = plt.subplots(3, 2, figsize=(9.0, 8.4), gridspec_kw={"hspace": 0.38, "wspace": 0.25})
    fig4b, ax4b = plt.subplots(3, 1, figsize=(4.6, 8.4), gridspec_kw={"hspace": 0.38})
    ax4 = np.column_stack([ax4m, ax4b])            # column 2 is the deficit index, on its own figure
    dec_rows, note = [], []
    cost_rows = load_predicted_cost_rows("h")
    for j, (v, lab) in enumerate(VARIANTS + [("predcost", "predicted cost from $h$, \u00a3")]):
        L = labels[v] if v != "predcost" else load_labels("predcost")
        base = cost_rows if v == "predcost" else d
        s = base[base["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
        traj = load_trajectories(v)
        pi = L.value_counts(normalize=True).sort_index()
        # ---- figure 3, top: paths and bands
        ax = ax3[0, j]
        for c in range(K):
            t = traj[(traj["cluster"] == c) & (traj["count"] >= MIN_SUPPORT)]
            ax.plot(t["age"], t["mean"], color=CLUSTER[c], lw=1.9, label=f"type {c + 1} ({pi[c]:.0%})")
            ax.fill_between(t["age"], t["mean"] - t["sd"], t["mean"] + t["sd"], color=CLUSTER[c], alpha=0.12, lw=0)
        ax.set_title(f"{lab}: {len(L):,} people, K = 3", loc="left")
        ax.legend(loc="upper left" if v == "predcost" else "lower left")
        ax.grid(True, axis="y")
        ax.set_xlim(MIN_AGE, MAX_AGE)
        # ---- figure 3, middle: within-cluster residual autocorrelation
        m = traj.set_index(["cluster", "age"])["mean"]
        s = s.assign(resid=lambda x: x[v] - m.reindex(pd.MultiIndex.from_frame(x[["cluster", "age"]])).to_numpy())
        s = s[s["resid"].notna()]          # the types are fitted on the contract's ages (20-89); drop what it does not cover
        ax = ax3[1, j]
        for c in range(K):
            r = s[s["cluster"] == c][["pidp", "age", "resid"]]
            pairs = r.merge(r, on="pidp", suffixes=("", "_t"))
            pairs = pairs[(pairs["age_t"] > pairs["age"]) & (pairs["age_t"] - pairs["age"] <= MAX_LAG)]
            pairs["lag"] = pairs["age_t"] - pairs["age"]
            ac = pairs.groupby("lag").apply(lambda g: np.corrcoef(g["resid"], g["resid_t"])[0, 1], include_groups=False)
            ax.plot(ac.index, ac.to_numpy(), color=CLUSTER[c], marker="o", ms=3, lw=1.4)
            note.append({"variant": v, "cluster": c + 1, **{f"ac{k}": ac.get(k, np.nan) for k in range(1, MAX_LAG + 1)}})
        ax.set_ylim(0, 1)
        ax.set_xlabel("lag, years")
        ax.set_title("autocorrelation of within-type deviations", loc="left", fontsize=9)
        ax.grid(True, axis="y")
        # ---- figure 3, bottom: composition strip
        comp = pd.read_csv(DESC / "paper_kmeans_composition.csv")
        comp = comp[(comp["variant"] == v) & (comp["lo"] == MIN_AGE) & (comp["hi"] == MAX_AGE)].pivot(index="age", columns="cluster", values="share")
        ax = ax3[2, j]
        ax.stackplot(comp.index, *[comp[c] for c in range(K)], colors=CLUSTER, alpha=0.9)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
        ax.spines["left"].set_visible(False)

        if v == "predcost":
            continue                                   # the decomposition figure stays on the health measure
        # ---- figure 4
        prof = s.groupby("age")[v].agg(["var", "count"])
        cell = s.groupby(["age", "cluster"])[v].agg(["mean", "count"]).reset_index()
        cell["share"] = cell["count"] / cell.groupby("age")["count"].transform("sum")
        cell["gmean"] = cell.groupby("age").apply(lambda g: np.average(g["mean"], weights=g["share"]),
                                                  include_groups=False).reindex(cell["age"]).to_numpy()
        between = cell.assign(b=lambda x: x["share"] * (x["mean"] - x["gmean"]) ** 2).groupby("age")["b"].sum()
        tot = prof["var"].where(prof["count"] >= 100)
        sm = lambda x: x.rolling(5, center=True).mean()  # noqa: E731
        ax = ax4[0, j]
        ax.plot(tot.index, sm(tot), color=VERM, lw=1.8, label="total")
        ax.plot(between.index, sm(between), color=BLUE, lw=1.8, label="between types")
        ax.plot(tot.index, sm(tot - between), color=GREEN, lw=1.8, label="within types")
        ax.set_title(f"(a) variance of {lab} by age", loc="left")
        ax.legend(loc="upper left"); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
        # smoothed cluster paths on every age, then d and the covariance across types
        paths = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(),
                                       min_count=MIN_SUPPORT).to_numpy()
                           for c in range(K) for t in [traj[traj["cluster"] == c]]])
        w = pi.reindex(range(K)).to_numpy()
        dpath = np.diff(paths, axis=1)                       # change from a to a+1, ages 20..89
        lev = paths[:, :-1]
        mean_l, mean_d = w @ lev, w @ dpath
        var_l = w @ (lev - mean_l) ** 2
        var_d = w @ (dpath - mean_d) ** 2
        cov_ld = w @ ((lev - mean_l) * (dpath - mean_d))
        a = AGES[:-1]
        ax = ax4[1, j]
        roll = lambda x: pd.Series(x).rolling(5, center=True).mean().to_numpy()  # noqa: E731
        ax.plot(a, roll(var_d), color=BLUE, lw=1.8)
        ax.set_title(r"(b) $\mathrm{Var}_j(d_{j,a})$, one-year change", loc="left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
        ax = ax4[2, j]
        ax.plot(a, roll(cov_ld), color=BLUE, lw=1.8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(r"(c) $\mathrm{Cov}_j(\bar h_{j,a}, d_{j,a})$", loc="left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
        for age_, vt, vb, vd, cv in zip(a, tot.reindex(a), between.reindex(a), var_d, cov_ld):
            dec_rows.append({"variant": v, "age": age_, "var_total": vt, "var_between": vb,
                             "var_within": vt - vb if pd.notna(vt) else np.nan, "var_d": vd, "cov_level_d": cv,
                             "corr_level_d": cv / np.sqrt(var_l[age_ - MIN_AGE] * vd) if vd > 0 else np.nan})
    fig3.text(0.01, -0.01,
              "The health contract: 38,963 people with at least three observed ages 20-89, one row per person-age; partial K-means with 50 seeded starts, "
              "types ordered worst to best.\nTop: cluster means where at least 25 members are observed, shaded to "
              "+/- 1 within-cluster sd. Middle: correlation of a person's deviation from\ntheir type's mean at age a "
              "with their deviation k years later. Bottom: type composition of the person-waves observed at each age.",
              fontsize=7.4, color=INK2, va="top")
    fig3.savefig(FIG / "fig_clusters.png")
    fig3b.text(0.01, -0.01, "As the main-text types figure, for the ten-deficit index (left) and for partial K-means on predicted cost (right): the pooled cost curve on h\n"
               "(cubic in the standardised score on costs capped at p99, waves 7-15, made monotone) evaluated at every contract row, so a cost-anchored scale of the measure.", fontsize=7.4, color=INK2, va="top")
    fig3b.savefig(FIG / "fig_clusters_fi10_cost.png", bbox_inches="tight")
    fig4.text(0.01, -0.01,
              "Row (a): the cross-sectional variance of the measure among the clustered people, split by the age-"
              "specific type composition; five-year rolling means.\nRows (b) and (c): the smoothed type mean paths "
              "(five-year rolling means, ends extended linearly), d the one-year change, moments across types with "
              "person-level shares, then a further five-year rolling mean.",
              fontsize=7.4, color=INK2, va="top")
    fig4.savefig(FIG / "fig_cluster_decomposition.png")
    fig4b.text(0.01, -0.01, "As the main-text decomposition figure, for the ten-deficit index.", fontsize=7.4, color=INK2, va="top")
    fig4b.savefig(FIG / "fig_cluster_decomposition_fi10.png", bbox_inches="tight")
    pd.DataFrame(dec_rows).to_csv(DESC / "paper_cluster_decomposition.csv", index=False)
    pd.DataFrame(note).to_csv(DESC / "paper_cluster_autocorr.csv", index=False)
    rows = []
    for a_, _ in VARIANTS:
        for b_, _ in VARIANTS:
            both = labels[a_].index.intersection(labels[b_].index)
            rows.append({"a": a_, "b": b_, "n": len(both),
                         "ari": adjusted_rand_score(labels[a_].loc[both], labels[b_].loc[both]),
                         "same": float((labels[a_].loc[both] == labels[b_].loc[both]).mean())})
    agree = pd.DataFrame(rows)
    agree.to_csv(DESC / "paper_cluster_agreement.csv", index=False)
    print(agree.round(3).to_string(index=False))
    print(pd.DataFrame(note).round(2).to_string(index=False))
    print(f"wrote fig_clusters.png and fig_cluster_decomposition.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
