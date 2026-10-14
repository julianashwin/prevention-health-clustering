"""Figures 3 and 4: the non-parametric types and their between/within anatomy.

Figure 3 (fig_clusters.png, theta; the same three panels for partial K-means on
predicted cost go to fig_clusters_predcost.png
for the appendix), one column per variant of the measure:
  top     cluster mean paths, K = 3, with the within-cluster +/- 1 sd band; the
          class shares in the title
  middle  autocorrelation of within-cluster deviations by lag in years, per
          cluster: the persistence of what the types do not explain
  bottom  cluster composition of the person-waves observed at each age

Figure 4 (fig_cluster_decomposition.png, theta; h and the frailty indices with
--measures), the clustering version of Exhibit 1, for two typologies on each
measure, the AR(1) + "spike" classes (solid; _paper_common.ssm_fit) and the
partial K-means types (dashed):
  (a) the cross-sectional variance at each age split into between-type
      (Var of the type means under the age-specific composition) and within
  (b) sd_j(d_j,a): the spread of the type slopes across types (the square root of
      Var_j(d_j,a), in the measure's units per year), age-specific composition
      shares; K-means slopes are one-year changes of the smoothed type paths,
      mixture slopes the derivative of the quadratic class paths
  (c) Cov_j(hbar_j,a, d_j,a): the level-slope covariance across types, whose
      sign says whether types are diverging or converging at that age
Rolling windows are PROFILE_WINDOW years (ten) on the K-means paths and series.

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
    AGES, DESC, FIG, FRAILTY_SUFFIX, FRAILTY_VARIANTS, HEALTHIER_HIGH, K, LABEL, MAX_AGE, MENTAL_SUFFIX, MENTAL_VARIANTS, MIN_AGE, PROFILE_WINDOW, VARIANTS, load_labels,
    load_measure, load_predicted_cost_rows, load_trajectories, smooth_path, ssm_fit,
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
        W_ = PROFILE_WINDOW
        sm = lambda x: x.rolling(W_, center=True).mean()  # noqa: E731
        let = "abc" if nh == 1 else ["adg", "beh", "cfi"][j] if nh == 3 else ["ace", "bdf"][j]
        # ---- K-means: smoothed type paths, one-year changes, age-specific composition as the weights
        paths = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(),
                                       min_count=MIN_SUPPORT, window=W_).to_numpy()
                           for c in range(K) for t in [traj[traj["cluster"] == c]]])
        comp_k = cell.pivot(index="age", columns="cluster", values="share").reindex(AGES).ffill().bfill().to_numpy()
        dpath = np.diff(paths, axis=1)                       # change from a to a+1, ages 20..89
        lev = paths[:, :-1]
        a = AGES[:-1]
        def type_moments(lev, d, w):
            ml = (w * lev.T).sum(1); md = (w * d.T).sum(1)
            var_l = (w * (lev.T - ml[:, None]) ** 2).sum(1); var_d = (w * (d.T - md[:, None]) ** 2).sum(1)
            cov = (w * (lev.T - ml[:, None]) * (d.T - md[:, None])).sum(1)
            return var_l, var_d, cov
        var_l, var_d, cov_ld = type_moments(lev, dpath, comp_k[:-1])
        roll = lambda x: pd.Series(x).rolling(W_, center=True).mean().to_numpy()  # noqa: E731
        # ---- the AR(1) + "spike" classes on the same measure: quadratic paths, their derivative, the posterior composition
        f = ssm_fit(v)
        comp_b = f["comp"][[f"class{k + 1}" for k in range(K)]].reindex(AGES).ffill().bfill().to_numpy()
        gm_b = (comp_b * f["paths"].T).sum(1)
        between_b = pd.Series((comp_b * (f["paths"].T - gm_b[:, None]) ** 2).sum(1), index=AGES)
        var_l_b, var_d_b, cov_b = type_moments(f["paths"], f["slopes"], comp_b)
        ax = ax4[0]
        ax.plot(tot.index, sm(tot), color=VERM, lw=1.8, label="total")
        ax.plot(AGES, between_b.reindex(AGES), color=BLUE, lw=1.8, label="between types")
        ax.plot(tot.index, sm(tot) - between_b.reindex(tot.index), color=GREEN, lw=1.8, label="within types")
        ax.plot(between.index, sm(between), color=BLUE, lw=1.6, ls="--")
        ax.plot(tot.index, sm(tot - between), color=GREEN, lw=1.6, ls="--")
        ax.plot([], [], color=INK2, lw=1.6, label='AR(1) + "spike" (solid), K-means (dashed)')
        ax.set_title(f"({let[0]}) variance of {lab} by age", loc="left")
        ax.legend(loc="upper left", fontsize=7.5); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0)
        ax = ax4[1]
        ax.plot(AGES, np.sqrt(var_d_b), color=BLUE, lw=1.8, label='AR(1) + "spike" classes')
        ax.plot(a, np.sqrt(roll(var_d)), color=BLUE, lw=1.6, ls="--", label="K-means types")
        ax.set_title(rf"({let[1]}) $\mathrm{{sd}}_j(d_{{j,a}})$, spread of the type slopes, per year", loc="left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylim(bottom=0); ax.legend(fontsize=7.5, loc="upper left")
        ax = ax4[2]
        ax.plot(AGES, cov_b, color=BLUE, lw=1.8)
        ax.plot(a, roll(cov_ld), color=BLUE, lw=1.6, ls="--")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(rf"({let[2]}) $\mathrm{{Cov}}_j(\bar h_{{j,a}}, d_{{j,a}})$", loc="left")
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
        for age_, vt, vb, vd, vl, cv in zip(a, tot.reindex(a), between.reindex(a), var_d, var_l, cov_ld):
            dec_rows.append({"variant": v, "typology": "kmeans", "age": age_, "var_total": vt, "var_between": vb,
                             "var_within": vt - vb if pd.notna(vt) else np.nan, "var_d": vd, "cov_level_d": cv,
                             "corr_level_d": cv / np.sqrt(vl * vd) if vd > 0 else np.nan})
        for age_, vt, vb, vd, vl, cv in zip(AGES, tot.reindex(AGES), between_b, var_d_b, var_l_b, cov_b):
            dec_rows.append({"variant": v, "typology": "ssm", "age": age_, "var_total": vt, "var_between": vb,
                             "var_within": vt - vb if pd.notna(vt) else np.nan, "var_d": vd, "cov_level_d": cv,
                             "corr_level_d": cv / np.sqrt(vl * vd) if vd > 0 else np.nan})
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
              (f"Two typologies, K = 3: the AR(1) + \"spike\" classes (solid)\nand the partial K-means types (dashed). Row (a): the\ncross-sectional variance of the measure among the contract's\npeople, "
               f"split into between types (the variance of the type\nmeans under the age-specific composition) and within;\n{PROFILE_WINDOW}-year rolling means. Rows (b) and (c): moments of the\ntype slopes across types "
               f"with age-specific composition shares.\nMixture: quadratic class paths and their derivative. K-means:\nsmoothed type paths ({PROFILE_WINDOW}-year rolling means, ends extended\nlinearly), "
               f"one-year changes, then a further {PROFILE_WINDOW}-year rolling mean."
               if narrow else f"As the main-text decomposition figure, on {measure}: AR(1) + \"spike\" classes solid, K-means types dashed, {PROFILE_WINDOW}-year rolling means, age-specific composition shares."),
              fontsize=7.4, color=INK2, va="top")
    fig4.savefig(FIG / f"fig_cluster_decomposition{tag}.png", bbox_inches="tight")
    pd.DataFrame(dec_rows).to_csv(DESC / f"paper_cluster_decomposition{tag}.csv", index=False)
    pd.DataFrame(note).to_csv(DESC / f"paper_cluster_autocorr{tag}.csv", index=False)
    # agreement between every pair of typologies on disk
    lab_all = pd.read_parquet(DESC / "paper_kmeans_labels.parquet")
    avail = [v for v in ("theta", "h", "frailty", "logfrailty", "predcost", "mental") if ((lab_all["variant"] == v) & (lab_all["lo"] == MIN_AGE)).any()]
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
