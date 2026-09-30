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

--frailty draws both figures on the 31-deficit frailty index and its shifted
log, log(frailty + 1/31) (higher = frailer, so type 1 is the top line) instead,
to fig_clusters_frailty.png and fig_cluster_decomposition_frailty.png, with the
csvs suffixed _frailty and the agreement table against the h and theta types.
Needs 22_paper_kmeans.py --frailty first.

The paper's figures (default run) are theta alone, one column each, with the predicted-cost
types drawn as their own appendix figure (fig_clusters_predcost.png); --measures draws both
figures on h, the frailty index and log frailty for the appendix (fig_clusters_measures.png,
fig_cluster_decomposition_measures.png). The agreement table covers every typology on disk.
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
    AGES, DESC, FIG, FRAILTY_SUFFIX, FRAILTY_VARIANTS, HEALTHIER_HIGH, K, LABEL, MAX_AGE, MENTAL_SUFFIX, MENTAL_VARIANTS, MIN_AGE, VARIANTS, load_labels,
    load_measure, load_predicted_cost_rows, load_trajectories, smooth_path,
)
from _style import BLUE, CLUSTER, GREEN, INK2, VERM, apply_style  # noqa: E402

MIN_SUPPORT, MAX_LAG = 25, 9
FRAIL = "--frailty" in sys.argv
MENTAL = "--mental" in sys.argv    # the mental GRM alone, K-means on the multidim contract rows
SUFFIX = MENTAL_SUFFIX if MENTAL else FRAILTY_SUFFIX if FRAIL else ""


def main() -> int:
    apply_style()
    # which figures: the paper's are theta alone, one column each, with the predicted-cost types as a
    # separate appendix figure; --measures the appendix versions on h and the two frailty indices;
    # --frailty the two frailty indices only
    RULERS = "--measures" in sys.argv
    if MENTAL:
        VLIST, tag = MENTAL_VARIANTS, SUFFIX
    elif FRAIL:
        VLIST, tag = FRAILTY_VARIANTS, SUFFIX
    elif RULERS:
        VLIST, tag = [("h", LABEL["h"])] + FRAILTY_VARIANTS, "_measures"
    else:
        VLIST, tag = [("theta", LABEL["theta"]), ("predcost", "predicted cost from $h$, £")], ""
    health = [(v, lab) for v, lab in VLIST if v != "predcost"]
    d = load_measure([v for v, _ in health])
    labels = {v: load_labels(v) for v, _ in VLIST}
    nh = len(health)
    g3 = {"height_ratios": [3.2, 1.9, 0.7], "hspace": 0.35}
    fig3, ax3m = plt.subplots(3, nh, figsize=(4.8 * nh, 8.2), gridspec_kw={**g3, "wspace": 0.22}, squeeze=False)
    cols3 = [ax3m[:, j] for j in range(nh)]
    if not (FRAIL or RULERS or MENTAL):
        fig3p, ax3p = plt.subplots(3, 1, figsize=(4.8, 8.2), gridspec_kw=g3, squeeze=False)
        cols3.append(ax3p[:, 0])
    fig4, ax4m = plt.subplots(3, nh, figsize=(4.8 * nh, 8.4), gridspec_kw={"hspace": 0.38, "wspace": 0.25}, squeeze=False)
    cols4 = [ax4m[:, j] for j in range(nh)]
    dec_rows, note = [], []
    cost_rows = load_predicted_cost_rows("h") if any(v == "predcost" for v, _ in VLIST) else None
    comp_all = pd.read_csv(DESC / "paper_kmeans_composition.csv")
    for j, (v, lab) in enumerate(VLIST):
        L = labels[v]
        base = cost_rows if v == "predcost" else d
        s = base[base["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
        traj = load_trajectories(v)
        pi = L.value_counts(normalize=True).sort_index()
        ax3 = cols3[j]
        # ---- types, top: paths and bands
        ax = ax3[0]
        for c in range(K):
            t = traj[(traj["cluster"] == c) & (traj["count"] >= MIN_SUPPORT)]
            ax.plot(t["age"], t["mean"], color=CLUSTER[c], lw=1.9, label=f"type {c + 1} ({pi[c]:.0%})")
            ax.fill_between(t["age"], t["mean"] - t["sd"], t["mean"] + t["sd"], color=CLUSTER[c], alpha=0.12, lw=0)
        ax.set_title(f"{lab}: {len(L):,} people, K = 3", loc="left")
        ax.legend(loc="upper left" if (v == "predcost" or not HEALTHIER_HIGH.get(v, True)) else "lower left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        # ---- types, middle: within-cluster residual autocorrelation
        m = traj.set_index(["cluster", "age"])["mean"]
        s = s.assign(resid=lambda x: x[v] - m.reindex(pd.MultiIndex.from_frame(x[["cluster", "age"]])).to_numpy())
        s = s[s["resid"].notna()]          # the types are fitted on the contract's ages (20-89); drop what it does not cover
        ax = ax3[1]
        for c in range(K):
            r = s[s["cluster"] == c][["pidp", "age", "resid"]]
            pairs = r.merge(r, on="pidp", suffixes=("", "_t"))
            pairs = pairs[(pairs["age_t"] > pairs["age"]) & (pairs["age_t"] - pairs["age"] <= MAX_LAG)]
            pairs["lag"] = pairs["age_t"] - pairs["age"]
            ac = pairs.groupby("lag").apply(lambda g: np.corrcoef(g["resid"], g["resid_t"])[0, 1], include_groups=False)
            ax.plot(ac.index, ac.to_numpy(), color=CLUSTER[c], marker="o", ms=3, lw=1.4)
            note.append({"variant": v, "cluster": c + 1, **{f"ac{k}": ac.get(k, np.nan) for k in range(1, MAX_LAG + 1)}})
        ax.set_ylim(0, 1); ax.set_xlabel("lag, years")
        ax.set_title("autocorrelation of within-type deviations", loc="left", fontsize=9)
        ax.grid(True, axis="y")
        # ---- types, bottom: composition strip
        comp = comp_all[(comp_all["variant"] == v) & (comp_all["lo"] == MIN_AGE) & (comp_all["hi"] == MAX_AGE)].pivot(index="age", columns="cluster", values="share")
        ax = ax3[2]
        ax.stackplot(comp.index, *[comp[c] for c in range(K)], colors=CLUSTER, alpha=0.9)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
        ax.spines["left"].set_visible(False)
        if v == "predcost":
            continue                                   # the decomposition stays on the health measures
        # ---- between versus within
        ax4 = cols4[j]
        prof = s.groupby("age")[v].agg(["var", "count"])
        cell = s.groupby(["age", "cluster"])[v].agg(["mean", "count"]).reset_index()
        cell["share"] = cell["count"] / cell.groupby("age")["count"].transform("sum")
        cell["gmean"] = cell.groupby("age").apply(lambda g: np.average(g["mean"], weights=g["share"]),
                                                  include_groups=False).reindex(cell["age"]).to_numpy()
        between = cell.assign(b=lambda x: x["share"] * (x["mean"] - x["gmean"]) ** 2).groupby("age")["b"].sum()
        tot = prof["var"].where(prof["count"] >= 100)
        sm = lambda x: x.rolling(5, center=True).mean()  # noqa: E731
        let = "abc" if nh == 1 else ["adg", "beh", "cfi"][j] if nh == 3 else ["ace", "bdf"][j]
        ax = ax4[0]
        ax.plot(tot.index, sm(tot), color=VERM, lw=1.8, label="total")
        ax.plot(between.index, sm(between), color=BLUE, lw=1.8, label="between types")
        ax.plot(tot.index, sm(tot - between), color=GREEN, lw=1.8, label="within types")
        ax.set_title(f"({let[0]}) variance of {lab} by age", loc="left")
        ax.legend(loc="upper left"); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
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
        roll = lambda x: pd.Series(x).rolling(5, center=True).mean().to_numpy()  # noqa: E731
        ax = ax4[1]
        ax.plot(a, roll(var_d), color=BLUE, lw=1.8)
        ax.set_title(rf"({let[1]}) $\mathrm{{Var}}_j(d_{{j,a}})$, one-year change", loc="left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
        ax = ax4[2]
        ax.plot(a, roll(cov_ld), color=BLUE, lw=1.8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(rf"({let[2]}) $\mathrm{{Cov}}_j(\bar h_{{j,a}}, d_{{j,a}})$", loc="left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
        for age_, vt, vb, vd, cv in zip(a, tot.reindex(a), between.reindex(a), var_d, cov_ld):
            dec_rows.append({"variant": v, "age": age_, "var_total": vt, "var_between": vb,
                             "var_within": vt - vb if pd.notna(vt) else np.nan, "var_d": vd, "cov_level_d": cv,
                             "corr_level_d": cv / np.sqrt(var_l[age_ - MIN_AGE] * vd) if vd > 0 else np.nan})
    measure = {"": "theta", "_measures": "h (left), the 31-deficit frailty index (middle) and log(frailty + 1/31) (right)",
             FRAILTY_SUFFIX: "the 31-deficit frailty index (left) and log(frailty + 1/31) (right)",
             MENTAL_SUFFIX: "the mental GRM"}[tag]
    narrow = nh == 1
    sample = ("The multidim health contract: 38,181 people with a mental score on every row,\nat least three observed ages 20-89, one row per person-age; the mental GRM\n(three SF-12 mental testlets and two GHQ-12 testlets, no depression diagnosis).\nPartial K-means with 50 seeded starts,\n"
              if MENTAL else
              "The health contract: 38,963 people with at least three observed ages 20-89,\none row per person-age; partial K-means with 50 seeded starts,\n")
    fig3.text(0.01, -0.01,
              (sample +
               "types ordered worst to best. Top: cluster means where at least 25 members\nare observed, shaded to +/- 1 within-cluster sd. Middle: correlation of a\n"
               "person's deviation from their type's mean at age a with their deviation\nk years later. Bottom: type composition of the person-waves observed at each age."
               if narrow else
               f"The health contract's people, on {measure}: partial K-means with 50 seeded starts, type 1 worst health (frailest on the frailty indices). "
               "Top: type means, +/- 1 within-type sd.\nMiddle: autocorrelation of within-type deviations. Bottom: type composition by age. "
               "Log frailty is log(frailty + 1/31), so person-ages with no deficit sit at the floor."),
              fontsize=7.4, color=INK2, va="top")
    fig3.savefig(FIG / f"fig_clusters{tag}.png", bbox_inches="tight")
    if not (FRAIL or RULERS or MENTAL):
        fig3p.text(0.01, -0.01, "Partial K-means on each person's predicted cost:\nthe pooled cost curve on h (cubic in the standardised\n"
                   "score on costs capped at p99, waves 7-15, made\nmonotone) evaluated at every contract row, a\ncost-anchored scale of the measure.",
                   fontsize=7.4, color=INK2, va="top")
        fig3p.savefig(FIG / "fig_clusters_predcost.png", bbox_inches="tight")
    fig4.text(0.01, -0.01,
              ("Row (a): the cross-sectional variance of the measure\namong the clustered people, split by the age-specific\ntype composition; "
               "five-year rolling means. Rows (b)\nand (c): the smoothed type mean paths (five-year\nrolling means, ends extended linearly), d the one-year\n"
               "change, moments across types with person-level\nshares, then a further five-year rolling mean."
               if narrow else f"As the main-text decomposition figure, on {measure}."),
              fontsize=7.4, color=INK2, va="top")
    fig4.savefig(FIG / f"fig_cluster_decomposition{tag}.png", bbox_inches="tight")
    pd.DataFrame(dec_rows).to_csv(DESC / f"paper_cluster_decomposition{tag}.csv", index=False)
    pd.DataFrame(note).to_csv(DESC / f"paper_cluster_autocorr{tag}.csv", index=False)
    # agreement between every pair of typologies on disk
    lab_all = pd.read_parquet(DESC / "paper_kmeans_labels.parquet")
    avail = [v for v in ("theta", "h", "frailty", "logfrailty", "fi10", "predcost", "mental") if ((lab_all["variant"] == v) & (lab_all["lo"] == MIN_AGE)).any()]
    LL = {v: load_labels(v) for v in avail}
    rows = []
    for a_ in avail:
        for b_ in avail:
            both = LL[a_].index.intersection(LL[b_].index)
            rows.append({"a": a_, "b": b_, "n": len(both), "ari": adjusted_rand_score(LL[a_].loc[both], LL[b_].loc[both]),
                         "same": float((LL[a_].loc[both] == LL[b_].loc[both]).mean())})
    agree = pd.DataFrame(rows)
    agree.to_csv(DESC / "paper_cluster_agreement.csv", index=False)
    print(agree[agree["a"] < agree["b"]].round(3).to_string(index=False))
    print(pd.DataFrame(note).round(2).to_string(index=False))
    print(f"wrote fig_clusters{tag}.png and fig_cluster_decomposition{tag}.png" + ("" if (FRAIL or RULERS) else " and fig_clusters_predcost.png") + f" to {FIG}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
