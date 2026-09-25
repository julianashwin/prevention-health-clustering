"""Cohort adjustment within partial K-means: shift, then cluster (Appendix on K-means).

For each variant, on the health contract's people: regress the score on a
full set of single-year age dummies and birth-decade dummies (reference the
1950s), subtract the estimated decade shifts, and re-run partial K-means
(K = 3, 50 seeded starts) on the adjusted person-by-age profiles. With age
fully flexible the decade shifts are identified only from the overlap of
neighbouring decades at the ages they share, which is the honest version of
what the mixture's quadratic-plus-decade fit did. Compares the adjusted
typology with the unadjusted one (22_paper_kmeans.py) person by person, and
the decade shifts with the mixture's (paper_cohort_effects.csv).

Outputs: paper/figures/fig_kmeans_cohort.png
         artifacts/descriptives/paper_kmeans_cohort_shifts.csv, _labels.parquet, _summary.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, VARIANTS, load_labels, load_measure  # noqa: E402
from _style import BLUE, CLUSTER, INK2, INK3, ORANGE, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402
from prevention_health_clustering.trajectories.partial_kmeans import partial_kmeans  # noqa: E402

CONTRACT = PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1"
DECADES = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
REF = 1950
MIN_OBS = 3


def shifts(s: pd.DataFrame, v: str):
    """OLS of the score on age dummies and decade dummies; returns the decade shifts with person-clustered SEs."""
    ages = np.sort(s["age"].unique())
    A = (s["age"].to_numpy()[:, None] == ages[None, :]).astype(float)          # full age dummies, no intercept
    decs = [d for d in DECADES if d != REF]
    D = (s["decade"].to_numpy()[:, None] == np.array(decs)[None, :]).astype(float)
    X = np.column_stack([A, D]); y = s[v].to_numpy(float)
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ (X.T @ y); u = y - X @ b
    g = s["pidp"].to_numpy(); order = np.argsort(g, kind="stable")
    Xs, us, gs = X[order], u[order], g[order]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for a, z in zip(edges[:-1], edges[1:]):
        sc = Xs[a:z].T @ us[a:z]; meat += np.outer(sc, sc)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    k = len(ages)
    out = pd.DataFrame({"decade": decs, "shift": b[k:], "se": se[k:]})
    return pd.concat([out, pd.DataFrame({"decade": [REF], "shift": [0.0], "se": [0.0]})]).sort_values("decade").reset_index(drop=True)


def cluster(args):
    v, wide = args
    lab = pd.Series(partial_kmeans(wide, n_clusters=K), index=wide.index)
    order = wide.mean(axis=1).groupby(lab).mean().sort_values().index
    return v, lab.map({old: new for new, old in enumerate(order)}).astype(int)


def main() -> int:
    apply_style()
    d = load_measure()
    birthy = pd.read_csv(CONTRACT / "long.csv", usecols=["pidp", "birthy"]).groupby("pidp")["birthy"].first()
    d = d[d["pidp"].isin(birthy.index)].copy()
    d["decade"] = (d["pidp"].map(birthy) // 10 * 10).astype(int).clip(lower=1920)   # the two 1900s people join the 1920s
    mix = pd.read_csv(DESC / "paper_cohort_effects.csv")
    shift_rows, tasks, adj = [], [], {}
    for v, lab in VARIANTS:
        sh = shifts(d, v)
        sh["variant"] = v
        shift_rows.append(sh)
        d[f"{v}_adj"] = d[v] - d["decade"].map(sh.set_index("decade")["shift"])
        w = d.pivot_table(index="pidp", columns="age", values=f"{v}_adj", aggfunc="mean").reindex(columns=range(MIN_AGE, MAX_AGE + 1))
        w = w[w.notna().sum(axis=1) >= MIN_OBS]
        tasks.append((v, w))
        print(f"{v}: decade shifts (age dummies), units of {v}:\n" + "  ".join(f"{int(r.decade)}s {r.shift:+.3f} ({r.se:.3f})" for r in sh.itertuples()), flush=True)
    shifts_df = pd.concat(shift_rows)
    shifts_df.to_csv(DESC / "paper_kmeans_cohort_shifts.csv", index=False)
    with Pool(3) as pool:
        for v, lab in pool.map(cluster, tasks):
            adj[v] = lab
    pd.DataFrame(adj).to_parquet(DESC / "paper_kmeans_cohort_labels.parquet")

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    summ = []
    for j, (v, lab) in enumerate(VARIANTS):
        sh = shifts_df[shifts_df["variant"] == v].set_index("decade")
        mx = mix[mix["variant"] == v].set_index("decade")
        ax = axes[0, j]
        ax.errorbar(sh.index, sh["shift"], yerr=1.96 * sh["se"], fmt="o-", color=BLUE, lw=1.3, capsize=2, ms=4, label="age dummies + decade dummies (OLS)")
        mxd = mx.reindex(DECADES)
        ax.errorbar(mxd.index, mxd["effect_units"], yerr=[mxd["effect_units"] - mxd["lo_units"], mxd["hi_units"] - mxd["effect_units"]],
                    fmt="s--", color=ORANGE, lw=1.1, capsize=2, ms=4, label="mixture: quadratic age + decade")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_title(f"{lab}: decade shift relative to the 1950s", loc="left", fontsize=9.5)
        ax.grid(True, axis="y"); ax.set_xlabel("birth decade")
        if j == 0:
            ax.legend(fontsize=7.5, loc="lower left")
        L0 = load_labels(v); L1 = adj[v]
        both = L0.index.intersection(L1.index)
        ari = adjusted_rand_score(L0.loc[both], L1.loc[both]); same = float((L0.loc[both] == L1.loc[both]).mean())
        s = d[d["pidp"].isin(both)].copy()
        s["c0"] = s["pidp"].map(L0); s["c1"] = s["pidp"].map(L1)
        ax = axes[1, j]
        for c in range(K):
            t0 = s[s["c0"] == c].groupby("age")[v].agg(["mean", "count"]); t0 = t0[t0["count"] >= 25]
            t1 = s[s["c1"] == c].groupby("age")[f"{v}_adj"].agg(["mean", "count"]); t1 = t1[t1["count"] >= 25]
            ax.plot(t0.index, t0["mean"], color=CLUSTER[c], lw=1.2, ls="--")
            ax.plot(t1.index, t1["mean"], color=CLUSTER[c], lw=1.8,
                    label=f"type {c + 1}: {(L0.loc[both] == c).mean():.0%} unadjusted, {(L1.loc[both] == c).mean():.0%} adjusted")
        ax.plot([], [], color=INK3, lw=1.2, ls="--", label="dashed: unadjusted; solid: cohort-adjusted")
        ax.set_title(f"type means: ARI {ari:.2f}, same type {same:.0%}", loc="left", fontsize=9.5)
        ax.legend(fontsize=6.6, loc="lower left"); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        ax = axes[2, j]
        per = s.drop_duplicates("pidp").set_index("pidp")
        for name, col, ls, mk in (("unadjusted", "c0", "--", "o"), ("cohort-adjusted", "c1", "-", "s")):
            sb = per.groupby("decade")[col].apply(lambda x: (x == 0).mean()).reindex(DECADES)
            ax.plot(sb.index, sb, color=CLUSTER[0], lw=1.6, ls=ls, marker=mk, ms=3.5, mfc="white" if mk == "o" else None,
                    label=f"worst type, {name}")
            sb3 = per.groupby("decade")[col].apply(lambda x: (x == 2).mean()).reindex(DECADES)
            ax.plot(sb3.index, sb3, color=CLUSTER[2], lw=1.6, ls=ls, marker=mk, ms=3.5, mfc="white" if mk == "o" else None,
                    label=f"best type, {name}")
        ax.set_title("type shares by birth decade", loc="left", fontsize=9.5); ax.set_xlabel("birth decade"); ax.grid(True, axis="y")
        if j == 0:
            ax.legend(fontsize=7, loc="center left", handlelength=3.2)
        summ.append({"variant": v, "ari": ari, "same": same, **{f"share_adj_{c + 1}": float((L1.loc[both] == c).mean()) for c in range(K)},
                     **{f"share_unadj_{c + 1}": float((L0.loc[both] == c).mean()) for c in range(K)}})
    fig.text(0.01, -0.01, "Health contract people. Top: decade level shifts from OLS on single-year age dummies plus decade dummies (95% person-clustered "
             "intervals), against the mixture's quadratic-plus-decade estimates.\nMiddle: K-means type means, unadjusted scores (dashed) and scores net "
             "of the OLS shifts re-clustered (solid). Bottom: share of people in the worst and best type by birth decade under both.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_kmeans_cohort.png", bbox_inches="tight")
    sm = pd.DataFrame(summ); sm.to_csv(DESC / "paper_kmeans_cohort_summary.csv", index=False)
    print(sm.round(3).to_string(index=False)); print(f"wrote {FIG / 'fig_kmeans_cohort.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
