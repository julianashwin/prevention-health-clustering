"""Figure 9: predicting next wave's outcomes from what is known at t.

The variant (h by default; theta or fi10 on the command line) sets the
measure, the K-means types and the class posteriors used; outputs for the
other variants carry the variant as a suffix.

On the health contract's people (38,963), every pair of adjacent waves (t,
t+1) is one observation. Outcomes at t+1: the measure h; death before the
next wave (the cross-wave death record, at risk = not recorded dead by t);
any in-patient stay in the twelve months before the t+1 interview; the GP
visit band at t+1 (both waves 7-15 only). Predictors at t, each set carrying a
quadratic in age:
  age                 age only
  + observables       education (4 groups), income rank, sex
  + K-means type      the K-means types on h
  + baseline class    posterior class probabilities, baseline mixture on h
  + AR(1)+ME class    posterior class probabilities, AR(1) plus measurement error on h
  + h(t)              the measure at t
  + h(t), h(t-1)      and its lag, where observed (a smaller sample; kept separate)
  + AR(1)+ME class, h(t)
  + all               AR(1)+ME class, h(t), observables
People are split 70/30 on a fixed hash; models fitted on the 70, scored on
the 30. Binary outcomes report AUC, continuous ones R-squared, both out of
sample, and the gain over the age-only set.

Caveat carried from the earlier version: the class posteriors and K-means
labels come from fits on every wave, including t+1, so the class rows test
the class-to-outcome link out of sample but not the whole pipeline.

Outputs: paper/figures/fig_prediction.png, paper/tables/tab_prediction.tex,
         artifacts/descriptives/paper_prediction.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, TAB, load_labels, load_measure, write_table  # noqa: E402
from _style import BLUE, CLUSTER, GREEN, INK2, ORANGE, PURPLE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402

V = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("h", "theta", "fi10") else "h"
SUFFIX = "" if V == "h" else f"_{V}"
VL = {"h": "h", "theta": r"\theta", "fi10": r"\mathrm{fi}"}[V]
EDUC = ["GCSE", "A-level", "degree or higher"]         # other/none is the reference


def logistic_fit(X, y, iters=60, ridge=1e-4):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    apply_style()
    d = load_measure([V]).sort_values(["pidp", "wave"])
    L = load_labels(V)
    d = d[d["pidp"].isin(L.index)].copy()
    # next-wave outcomes
    nxt = d[["pidp", "wave", V]].rename(columns={"wave": "wave_next", V: "h_next"})
    nxt["wave"] = nxt["wave_next"] - 1
    d = d.merge(nxt[["pidp", "wave", "h_next"]], on=["pidp", "wave"], how="left")
    lag = d[["pidp", "wave", V]].rename(columns={V: "h_lag"}); lag["wave"] = lag["wave"] + 1
    d = d.merge(lag, on=["pidp", "wave"], how="left")
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", "hosp", "hl2gp"])
    cost["inpat"] = np.where(cost["hosp"] == 1, 1.0, np.where(cost["hosp"] == 2, 0.0, np.nan))
    cost["wave"] = cost["wave"] - 1                     # align to the predicting wave
    d = d.merge(cost[["pidp", "wave", "inpat", "hl2gp"]].rename(columns={"inpat": "inpat_next", "hl2gp": "gp_next"}),
                on=["pidp", "wave"], how="left")
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "dcsedw_dv", "sex"])
    death_wave = panel.groupby("pidp")["dcsedw_dv"].max() - 18
    sex = panel.groupby("pidp")["sex"].first()
    d = d.merge(death_wave.rename("death_wave"), left_on="pidp", right_index=True, how="left")
    d["died_next"] = np.where(d["wave"] >= 15, np.nan, (d["death_wave"] == d["wave"] + 1).astype(float))
    d.loc[d["death_wave"] <= d["wave"], "died_next"] = np.nan
    # predictors at t
    per = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet").set_index("pidp")
    d = d.join(per[["educ_group", "income_rank_mean"]], on="pidp")
    d["female"] = d["pidp"].map(sex).eq(2).astype(float)
    for g in EDUC:
        d[f"educ_{g}"] = (d["educ_group"] == g).astype(float)
    d["km"] = d["pidp"].map(L)
    for c in (1, 2):
        d[f"km_{c}"] = (d["km"] == c).astype(float)
    for tag in ("base", "ssm"):
        post = pd.read_parquet(DESC / f"paper_bayes_{tag}_posteriors.parquet")
        post = post[post["variant"] == V].set_index("pidp")[["class1", "class2"]]
        d = d.join(post.rename(columns={"class1": f"{tag}_c1", "class2": f"{tag}_c2"}), on="pidp")
    d["a"] = (d["age"] - 55) / 10
    obs = ["income_rank_mean", "female"] + [f"educ_{g}" for g in EDUC]
    SETS = {
        "age": [],
        "+ observables": obs,
        "+ K-means type": ["km_1", "km_2"],
        "+ baseline class": ["base_c1", "base_c2"],
        "+ AR(1)+ME class": ["ssm_c1", "ssm_c2"],
        f"+ {V}(t)": [V],
        f"+ {V}(t), {V}(t-1)": [V, "h_lag"],
        f"+ AR(1)+ME class, {V}(t)": ["ssm_c1", "ssm_c2", V],
        "+ all": ["ssm_c1", "ssm_c2", V] + obs,
    }
    OUT = [("h_next", f"${VL}_{{t+1}}$", "r2"), ("died_next", "death before $t+1$", "auc"),
           ("inpat_next", "in-patient stay, $t+1$", "auc"), ("gp_next", "GP visits, $t+1$", "r2")]
    rows = []
    for oc, olab, kind in OUT:
        base_cols = ["a", V] + obs + ["km_1", "base_c1", "ssm_c1"]
        sub = d.dropna(subset=[oc] + base_cols)
        test = ((pd.util.hash_pandas_object(sub["pidp"], index=False) % 10) >= 7).to_numpy()
        y = sub[oc].to_numpy(float)
        for name, cols in SETS.items():
            s2, t2, y2 = sub, test, y
            if "h_lag" in cols:
                keep = sub["h_lag"].notna().to_numpy()
                s2, t2, y2 = sub[keep], test[keep], y[keep]
            X = np.column_stack([np.ones(len(s2)), s2["a"], s2["a"] ** 2] + [s2[c].to_numpy(float) for c in cols])
            tr, te = ~t2, t2
            if kind == "auc":
                b = logistic_fit(X[tr], y2[tr]); p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
                score = auc(y2[te], p[te])
            else:
                b = np.linalg.lstsq(X[tr], y2[tr], rcond=None)[0]; p = X @ b
                score = float(1 - ((y2[te] - p[te]) ** 2).sum() / ((y2[te] - y2[te].mean()) ** 2).sum())
            rows.append({"outcome": oc, "outcome_label": olab, "metric": kind, "predictors": name, "n": len(s2),
                         "n_test": int(te.sum()), "events": float(y2.mean()) if kind == "auc" else np.nan, "score": score})
        print(f"{oc}: {len(sub):,} pairs, {sub['pidp'].nunique():,} people" + (f", event rate {y.mean():.3%}" if kind == "auc" else ""))
    t = pd.DataFrame(rows)
    t["gain"] = t["score"] - t.groupby("outcome")["score"].transform("first")
    t.to_csv(DESC / f"paper_prediction{SUFFIX}.csv", index=False)
    piv = t.pivot(index="predictors", columns="outcome_label", values="score").reindex(list(SETS))
    piv = piv[[o[1] for o in OUT]]
    print(piv.round(3).to_string())
    write_table(TAB / f"tab_prediction{SUFFIX}.tex", ["predictors at $t$"] + [f"{o[1]} ({'AUC' if o[2] == 'auc' else '$R^2$'})" for o in OUT],
                [[nm] + [f"{piv.loc[nm, o[1]]:.3f}" for o in OUT] for nm in piv.index], colspec="lrrrr")
    # figure: gain over age by predictor set, one panel per outcome
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    names = list(SETS)[1:]
    cols = [INK2, CLUSTER[2], CLUSTER[1], CLUSTER[0], ORANGE, VERM, PURPLE, GREEN]
    for ax, (oc, olab, kind) in zip(axes, OUT):
        q = t[t["outcome"] == oc].set_index("predictors").reindex(names)
        base = t[(t["outcome"] == oc) & (t["predictors"] == "age")]["score"].iloc[0]
        ax.barh(np.arange(len(names)), q["gain"], color=cols[: len(names)])
        ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.grid(True, axis="x")
        ax.set_title(f"{olab}\n{'AUC' if kind == 'auc' else '$R^2$'} gain over age only ({base:.2f})", fontsize=9)
    fig.text(0.01, -0.03, "Health contract people, adjacent-wave pairs; people split 70/30, fitted on the 70 and scored on the 30. "
             "Death: at risk at t, died before the next wave. In-patient and GP outcomes exist for waves 7-15 only.\n"
             f"Class posteriors and K-means types use the whole history, including t+1. The {V}(t), {V}(t-1) row is on the pairs "
             "with a lag observed.", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout(); fig.savefig(FIG / f"fig_prediction{SUFFIX}.png")
    print(f"wrote fig_prediction{SUFFIX}.png, tab_prediction{SUFFIX}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
