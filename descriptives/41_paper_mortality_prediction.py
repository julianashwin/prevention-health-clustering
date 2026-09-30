"""Mortality prediction on the full-sample fits: every person-wave at risk.

The held-out exercise (33_paper_prediction.py) can only predict death at the
last interview, because the held rows are the last two rows of every person.
This companion uses the full-sample fits and every adjacent-wave pair: at each
row t (wave < 15, not yet recorded dead) the outcome is death before the next
wave, so the population is everyone at risk and the event rate is the panel's
yearly death rate rather than a last-row selection.

What is and is not out of sample here. The outcome never enters the
single-channel fits, so their class posteriors are out of sample for death
(they use the person's whole health history, including waves after t). The
multidimensional fit includes the mortality hazard, and a decedent's event is
on their last row, so its class posterior taken from the full likelihood has
seen the outcome; it is shown twice: from the Gaussian channels only (the
class from physical and mental health alone, out of sample for death) and
from the full likelihood (in sample, an upper bound). The multidimensional
model's own hazard, the class-weighted Gompertz-Makeham hazard at the
person's age with the Gaussian-only posterior, enters as a single predictor.

Predictor sets, each with a quadratic in age; people split 70/30 on a fixed
hash, fitted on the 70, scored on the 30; AUC. Both measures, on the people in
both the health and the multidim contracts.

The figure's lower row is the held-out comparison from 33_paper_prediction.py
(paper_prediction_death_heldout.csv), which must run first.

Outputs: paper/figures/fig_prediction_mortality.png,
         artifacts/descriptives/paper_mortality_prediction.csv
"""

from __future__ import annotations

import importlib
import textwrap
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import CONTRACT, DESC, FIG, kalman_person_lp, load_measure  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, PURPLE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

M39 = importlib.import_module("39_paper_multidim")
K = 3
VARIANTS = ["h", "theta"]
VL = {"h": "h", "theta": r"\theta"}
EDUC = ["GCSE", "A-level", "degree or higher"]
MD_CHAINS = {"theta-ssm-mort": None, "h-ssm-mort": [1, 2, 3]}


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


def multidim_posteriors(v: str) -> pd.DataFrame:
    """Full-sample multidim posteriors from the Gaussian channels only and from the full
    likelihood, and the class-weighted yearly hazard at each row (Gaussian-only weights)."""
    tag = f"{v}-ssm-mort"; fit_dir = M39.MD / tag
    d = json.load(open(fit_dir / "stan_data.json")); summ = json.load(open(fit_dir / "run_summary.json"))
    m, _, _ = M39.chain_means(fit_dir, MD_CHAINS[tag])
    theta = np.array([m[f"theta.{k}"] for k in range(1, K + 1)])
    coef = np.array([[[m[f"coef.{c}.{k}.{p}"] for p in (1, 2, 3)] for k in range(1, K + 1)] for c in (1, 2)])
    sigma = np.array([m[f"sigma.1.{c}"] for c in (1, 2)])
    rho = np.array([[m[f"rho.{k}.{c}"] for c in (1, 2)] for k in range(1, K + 1)])
    smeas = np.array([m[f"sigma_meas.{c}"] for c in (1, 2)])
    log_b = np.array([m[f"log_b.{k}"] for k in range(1, K + 1)]); slope = np.array([m[f"gomp_slope.{k}"] for k in range(1, K + 1)])
    makeham = m["makeham.1"]
    X = np.asarray(d["X"]); Y = np.asarray(d["y"]); fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    per = np.zeros((len(fs), K))
    for c in range(2):
        per += kalman_person_lp(Y[c], X, coef[c], sigma[c], smeas[c], rho[:, c], np.asarray(d["age_gap"], float), fs, fe)
    def weights(lp):
        un = np.log(theta)[None, :] + lp
        w = np.exp(un - un.max(axis=1, keepdims=True)); return w / w.sum(axis=1, keepdims=True)
    w_g = weights(per); w_full = weights(per + M39.gompertz_person_lp(d, log_b, slope, makeham))
    long = pd.read_csv(M39.CONTRACT / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    assert len(long) == len(Y[0])
    person = np.repeat(np.arange(len(fs)), fe - fs + 1)
    age = long["age"].to_numpy(float)
    hz = np.vstack([makeham + np.exp(log_b[k] + slope[k] * (age - 55)) for k in range(K)]).T      # rows x K
    rows = pd.DataFrame({"pidp": long["pidp"], "wave": long["wave"],
                         "mdg_c1": w_g[person, 0], "mdg_c2": w_g[person, 1],
                         "mdf_c1": w_full[person, 0], "mdf_c2": w_full[person, 1],
                         "md_loghaz": np.log((w_g[person] * hz).sum(axis=1))})
    return rows


def main() -> int:
    apply_style()
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "dcsedw_dv", "sex"])
    death_wave = panel.groupby("pidp")["dcsedw_dv"].max() - 18
    sex = panel.groupby("pidp")["sex"].first()
    per = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet").set_index("pidp")
    health_people = pd.read_csv(CONTRACT / "long.csv", usecols=["pidp"])["pidp"].unique()
    md_people = pd.read_csv(M39.CONTRACT / "long.csv", usecols=["pidp"])["pidp"].unique()
    people = np.intersect1d(health_people, md_people)
    out_rows, aucs = [], {}
    for v in VARIANTS:
        d = load_measure([v]).sort_values(["pidp", "wave"])
        d = d[d["pidp"].isin(people)].copy()
        lag = d[["pidp", "wave", v]].rename(columns={v: "y_lag"}); lag["wave"] = lag["wave"] + 1
        d = d.merge(lag, on=["pidp", "wave"], how="left")
        d = d.merge(death_wave.rename("death_wave"), left_on="pidp", right_index=True, how="left")
        d["died_next"] = np.where(d["wave"] >= 15, np.nan, (d["death_wave"] == d["wave"] + 1).astype(float))
        d.loc[d["death_wave"] <= d["wave"], "died_next"] = np.nan
        d = d.join(per[["educ_group", "income_rank_mean"]], on="pidp")
        d["female"] = d["pidp"].map(sex).eq(2).astype(float)
        for g in EDUC:
            d[f"educ_{g}"] = (d["educ_group"] == g).astype(float)
        for tag in ("base", "ssm"):
            post = pd.read_parquet(DESC / f"paper_bayes_{tag}_posteriors.parquet")
            post = post[post["variant"] == v].set_index("pidp")[["class1", "class2"]]
            d = d.join(post.rename(columns={"class1": f"{tag}_c1", "class2": f"{tag}_c2"}), on="pidp")
        d = d.merge(multidim_posteriors(v), on=["pidp", "wave"], how="left")
        d["a"] = (d["age"] - 55) / 10
        obs = ["income_rank_mean", "female"] + [f"educ_{g}" for g in EDUC]
        SETS = {
            "age": [],
            "+ observables": obs,
            "+ iid class": ["base_c1", "base_c2"],
            '+ AR(1)+"spike" class': ["ssm_c1", "ssm_c2"],
            "+ multidim class (health channels only)": ["mdg_c1", "mdg_c2"],
            "+ multidim class (full likelihood, in sample)": ["mdf_c1", "mdf_c2"],
            "+ multidim model hazard": ["md_loghaz"],
            f"+ {v}(t)": [v],
            f"+ {v}(t), {v}(t-1)": [v, "y_lag"],
            f'+ AR(1)+"spike" class, {v}(t)': ["ssm_c1", "ssm_c2", v],
            f"+ multidim class (health only), {v}(t)": ["mdg_c1", "mdg_c2", v],
            "+ all": ["ssm_c1", "ssm_c2", v] + obs,
        }
        base_cols = ["died_next", "a", v] + obs + ["base_c1", "ssm_c1", "mdg_c1", "md_loghaz"]
        s = d.dropna(subset=base_cols)
        test = ((pd.util.hash_pandas_object(s["pidp"], index=False) % 10) >= 7).to_numpy()
        y = s["died_next"].to_numpy(float)
        print(f"{v}: {len(s):,} person-waves at risk, {s['pidp'].nunique():,} people, {int(y.sum()):,} deaths ({y.mean():.3%})")
        for name, cols in SETS.items():
            s2, t2, y2 = s, test, y
            if "y_lag" in cols:
                keep = s["y_lag"].notna().to_numpy(); s2, t2, y2 = s[keep], test[keep], y[keep]
            X = np.column_stack([np.ones(len(s2)), s2["a"], s2["a"] ** 2] + [s2[c].to_numpy(float) for c in cols])
            b = logistic_fit(X[~t2], y2[~t2]); p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
            score = auc(y2[t2], p[t2])
            out_rows.append({"variant": v, "predictors": name, "n": len(s2), "n_test": int(t2.sum()), "deaths": int(y2.sum()), "auc": score})
        aucs[v] = pd.DataFrame([r for r in out_rows if r["variant"] == v]).set_index("predictors")["auc"]
        print(aucs[v].round(3).to_string())
    t = pd.DataFrame(out_rows); t["gain"] = t["auc"] - t.groupby("variant")["auc"].transform("first")
    t.to_csv(DESC / "paper_mortality_prediction.csv", index=False)
    held = pd.read_csv(DESC / "paper_prediction_death_heldout.csv")      # from 33_paper_prediction.py
    for VSEL, sfx in ((['theta'], ''), (['h'], '_h')):          # theta for the paper, h for the appendix
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.6), gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.32}, squeeze=False)
        colmap = {"+ observables": INK2, "+ iid class": CLUSTER[2], '+ AR(1)+"spike" class': CLUSTER[1],
                  "+ multidim class (health channels only)": CLUSTER[0], "+ multidim class": CLUSTER[0],
                  "+ multidim class (full likelihood, in sample)": "#9e9ac8", "+ multidim model hazard": VERM,
                  '+ AR(1)+"spike" forecast': VERM, "+ multidim forecast": VERM, "+ all": GREEN}
        def colour(n):
            if n in colmap:
                return colmap[n]
            return PURPLE if "class" in n else ORANGE
        for c, v in enumerate(VSEL):
            for r, (tab, metric, title) in enumerate([
                    (t[t["variant"] == v].rename(columns={"auc": "score"}), "full", "every person-wave at risk, full-sample fits"),
                    (held[held["variant"] == v], "held", "death after the last interview vs leaving alive, held-out fits")]):
                ax = axes[r, c]
                names = [n for n in tab["predictors"] if n != "age"]
                q = tab.set_index("predictors").reindex(names)
                base = tab[tab["predictors"] == "age"]["score"].iloc[0]
                ax.barh(np.arange(len(names)), q["gain"].fillna(0), color=[colour(n) for n in names])
                for i, gval in enumerate(q["gain"]):
                    if np.isnan(gval):
                        ax.text(0.0005, i, "fit running", va="center", fontsize=7, color=INK2)
                ax.set_yticks(np.arange(len(names)))
                ax.set_yticklabels([n.replace(f"{v}(t)", f"${VL[v]}(t)$").replace(f"{v}(t-1)", f"${VL[v]}(t-1)$") for n in names], fontsize=8)
                ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.grid(True, axis="x")
                ax.set_title(f"({'ab'[r]}) on ${VL[v]}$, {title}\nAUC gain over age only ({base:.2f})", fontsize=9, loc="left")
        n = t[(t["variant"] == VSEL[0]) & (t["predictors"] == "age")].iloc[0]
        nh = held[(held["variant"] == VSEL[0]) & (held["predictors"] == "age")].iloc[0]
        footer = (f"Top: death before the next wave at every person-wave at risk ({n['n']:,} person-waves, {n['deaths']:,} deaths), full-sample fits, people in both the health and the multidim contracts. "
                 "Single-channel class posteriors never saw the death record;\nthe multidimensional class is shown from the physical and mental channels only (out of sample for death) and from the full likelihood, "
                 "which includes the decedent's event (in sample); the model hazard is the class-weighted Gompertz-Makeham hazard\nat the person's age with the health-only weights. "
                 f"Bottom: the held-out fits, predicting from the last fitted row t whether a person dies after their last interview (t+2) or leaves the panel alive ({nh['n']:,} people, "
                 f"{nh['events']:.1%} deaths); nobody dies before t+2 by construction.\nThe multidimensional posterior there uses survival through the fitted window only. Both: split 70/30 by person, fitted on the 70, scored on the 30.")
        footer = "\n".join(textwrap.wrap(footer.replace("\n", " "), 118))
        fig.text(0.01, 0.02, footer, fontsize=7.2, color=INK2, va="top")
        fig.subplots_adjust(top=0.95, bottom=0.2, left=0.42, right=0.98)
        fig.savefig(FIG / f"fig_prediction_mortality{sfx}.png")
    print("wrote fig_prediction_mortality{,_h}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
