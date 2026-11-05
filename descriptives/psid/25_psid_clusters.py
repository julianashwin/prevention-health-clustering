"""Figures 4 and 5 on PSID: the K-means types and their between/within anatomy, theta, K = 3.

The theta column of descriptives/25_paper_clusters.py (logic copied; that script reads the UKHLS
contract), on the PSID types from 22_psid_kmeans.py:
  Figure 4 (fig_clusters.png): type means with the +/- 1 within-type sd band where at least 25
      members are observed; the autocorrelation of within-type deviations by lag; the type
      composition of the person-waves observed at each age. PSID lags are the even ones, 2-10
      years (waves are biennial; odd lags rest on few pairs and are dropped).
  Figure 5 (fig_cluster_decomposition.png): (a) the cross-sectional variance split into between
      and within types; (b) sd_j(d_j,a), the spread of the type slopes (one-year changes of the
      smoothed type paths); (c) Cov_j(hbar_j,a, d_j,a). Ten-year rolling means, age-specific shares.

Outputs: descriptives/psid/figures/main/fig_clusters.png, fig_cluster_decomposition.png;
artifacts/descriptives/psid/cluster_decomposition.csv, cluster_autocorr.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import AGES, DESC, FIG, K, MAX_AGE, MIN_AGE, contract, load_composition, load_labels, load_trajectories  # noqa: E402
from _paper_common import PROFILE_WINDOW, smooth_path  # noqa: E402
from _style import BLUE, CLUSTER, GREEN, INK2, VERM, apply_style  # noqa: E402

MIN_SUPPORT = 25
LAGS = [2, 4, 6, 8, 10]


def main() -> int:
    apply_style()
    v, lab = "theta", r"$\theta$"
    L = load_labels(v)
    d = contract()
    s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    traj = load_trajectories(v)
    pi = L.value_counts(normalize=True).sort_index()
    fig3, ax3 = plt.subplots(3, 1, figsize=(4.8, 8.2), gridspec_kw={"height_ratios": [3.2, 1.9, 0.7], "hspace": 0.35})
    ax = ax3[0]
    for c in range(K):
        t = traj[(traj["cluster"] == c) & (traj["count"] >= MIN_SUPPORT)]
        ax.plot(t["age"], t["mean"], color=CLUSTER[c], lw=1.9, label=f"type {c + 1} ({pi[c]:.0%})")
        ax.fill_between(t["age"], t["mean"] - t["sd"], t["mean"] + t["sd"], color=CLUSTER[c], alpha=0.12, lw=0)
    ax.set_title(f"{lab}: {len(L):,} people, K = 3", loc="left"); ax.legend(loc="lower left")
    ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    m = traj.set_index(["cluster", "age"])["mean"]
    s = s.assign(resid=lambda x: x[v] - m.reindex(pd.MultiIndex.from_frame(x[["cluster", "age"]])).to_numpy()).dropna(subset=["resid"])
    ax = ax3[1]; note = []
    for c in range(K):
        r = s[s["cluster"] == c][["pidp", "age", "resid"]]
        pairs = r.merge(r, on="pidp", suffixes=("", "_t"))
        pairs["lag"] = pairs["age_t"] - pairs["age"]
        pairs = pairs[pairs["lag"].isin(LAGS)]
        ac = pairs.groupby("lag").apply(lambda g: np.corrcoef(g["resid"], g["resid_t"])[0, 1], include_groups=False)
        ax.plot(ac.index, ac.to_numpy(), color=CLUSTER[c], marker="o", ms=3, lw=1.4)
        note.append({"cluster": c + 1, **{f"ac{k}": ac.get(k, np.nan) for k in LAGS}})
    ax.set_ylim(0, 1); ax.set_xlabel("lag, years"); ax.set_xticks(LAGS)
    ax.set_title("autocorrelation of within-type deviations", loc="left", fontsize=9); ax.grid(True, axis="y")
    comp = load_composition(v).pivot(index="age", columns="cluster", values="share")
    ax = ax3[2]
    ax.stackplot(comp.index, *[comp[c] for c in range(K)], colors=CLUSTER, alpha=0.9)
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age"); ax.spines["left"].set_visible(False)
    fig3.text(0.01, -0.01,
              f"The PSID contract: {len(L):,} people with at least three observed ages 20-89,\none row per person-age; partial K-means with 50 seeded starts,\n"
              "types ordered worst to best. Top: cluster means where at least 25 members\nare observed, shaded to +/- 1 within-cluster sd. Middle: correlation of a\n"
              "person's deviation from their type's mean at age a with their deviation\nk years later, at the even lags of the biennial waves. Bottom: type\n"
              "composition of the person-waves observed at each age.", fontsize=7.4, color=INK2, va="top")
    fig3.savefig(FIG / "fig_clusters.png", bbox_inches="tight")

    # ---- Figure 5: between versus within
    fig4, ax4 = plt.subplots(3, 1, figsize=(4.8, 8.4), gridspec_kw={"hspace": 0.38})
    prof = s.groupby("age")[v].agg(["var", "count"])
    cell = s.groupby(["age", "cluster"])[v].agg(["mean", "count"]).reset_index()
    cell["share"] = cell["count"] / cell.groupby("age")["count"].transform("sum")
    cell["gmean"] = cell.groupby("age").apply(lambda g: np.average(g["mean"], weights=g["share"]), include_groups=False).reindex(cell["age"]).to_numpy()
    between = cell.assign(b=lambda x: x["share"] * (x["mean"] - x["gmean"]) ** 2).groupby("age")["b"].sum()
    tot = prof["var"].where(prof["count"] >= 100)
    W_ = PROFILE_WINDOW
    sm = lambda x: x.rolling(W_, center=True).mean()  # noqa: E731
    paths = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(), min_count=MIN_SUPPORT, window=W_).to_numpy()
                       for c in range(K) for t in [traj[traj["cluster"] == c]]])
    comp_k = cell.pivot(index="age", columns="cluster", values="share").reindex(AGES).ffill().bfill().to_numpy()
    dpath = np.diff(paths, axis=1); lev = paths[:, :-1]; a = AGES[:-1]; w = comp_k[:-1]
    ml = (w * lev.T).sum(1); md = (w * dpath.T).sum(1)
    var_l = (w * (lev.T - ml[:, None]) ** 2).sum(1); var_d = (w * (dpath.T - md[:, None]) ** 2).sum(1)
    cov_ld = (w * (lev.T - ml[:, None]) * (dpath.T - md[:, None])).sum(1)
    roll = lambda x: pd.Series(x).rolling(W_, center=True).mean().to_numpy()  # noqa: E731
    ax = ax4[0]
    ax.plot(tot.index, sm(tot), color=VERM, lw=1.8, label="total")
    ax.plot(between.index, sm(between), color=BLUE, lw=1.8, label="between types")
    ax.plot(tot.index, sm(tot - between), color=GREEN, lw=1.8, label="within types")
    ax.set_title(f"(a) variance of {lab} by age", loc="left"); ax.legend(loc="upper left")
    ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
    ax = ax4[1]
    ax.plot(a, np.sqrt(roll(var_d)), color=BLUE, lw=1.8)
    ax.set_title(r"(b) $\mathrm{sd}_j(d_{j,a})$, spread of the type slopes, per year", loc="left")
    ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
    ax = ax4[2]
    ax.plot(a, roll(cov_ld), color=BLUE, lw=1.8); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(r"(c) $\mathrm{Cov}_j(\bar h_{j,a}, d_{j,a})$", loc="left")
    ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
    fig4.text(0.01, -0.01,
              f"PSID partial K-means types, K = 3. Row (a): the cross-sectional\nvariance of the measure among the clustered people, split\ninto between types (the variance of the type means under the\n"
              f"age-specific composition) and within; {PROFILE_WINDOW}-year rolling means.\nRows (b) and (c): the smoothed type paths ({PROFILE_WINDOW}-year rolling\nmeans, ends extended linearly), d the one-year change, "
              f"moments\nacross types with age-specific composition shares, then a\nfurther {PROFILE_WINDOW}-year rolling mean.", fontsize=7.4, color=INK2, va="top")
    fig4.savefig(FIG / "fig_cluster_decomposition.png", bbox_inches="tight")
    dec = pd.DataFrame({"age": a, "var_total": tot.reindex(a).to_numpy(), "var_between": between.reindex(a).to_numpy(),
                        "var_d": var_d, "cov_level_d": cov_ld})
    dec["between_share"] = dec["var_between"] / dec["var_total"]
    dec.to_csv(DESC / "cluster_decomposition.csv", index=False)
    pd.DataFrame(note).to_csv(DESC / "cluster_autocorr.csv", index=False)
    print(pd.DataFrame(note).round(2).to_string(index=False))
    print(dec.set_index("age").loc[[30, 50, 70, 85], ["var_total", "var_between", "between_share"]].round(3).to_string())
    print("type means at 30/50/70:", traj[traj["age"].isin([30, 50, 70])].pivot(index="age", columns="cluster", values="mean").round(2).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
