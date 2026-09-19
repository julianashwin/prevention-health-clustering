"""How well does each model's class structure predict external outcomes?

Mortality and healthcare utilisation are not channels in any fitted model, so
this is criterion validity rather than held-out density: given a person's
class posterior, how well can we predict outcomes the model never saw?

Outcomes
  mortality    person-level: died during the panel (dcsedfl_dv == 1)
  in-patient   person-wave: any in-patient stay in the last 12 months
  GP visits    person-wave: visit band 0-4 (hl2gp, 'Visited GP in last 12
               months'); out-patient is hl2hop, a separate band

Predictor sets, each also carrying a quadratic in age so the comparison is
against what age alone already gives:
  age          age + age^2 only
  +class       age + the K-1 free class posteriors
  +measure     age + the model's own health measure (continuous), which is
               the fairer competitor: it is what the classes summarise
  +class+measure  both: what the classes add once the measure is known

Evaluation splits PEOPLE 70/30 on a fixed hash, so in-sample and
out-of-sample differ in who is scored, not in which waves. Binary outcomes
report AUC and log-loss; the GP visit band reports RMSE and R-squared.

IMPORTANT CAVEAT, recorded in the note: the class posteriors come from
mixtures fitted on all people, so the split tests the class-to-outcome
relationship out of sample, not the whole pipeline. A fully clean version
would refit each mixture on the training people.

Outputs: artifacts/descriptives/outcome_prediction.csv,
         measuring_health/figures/fig_outcome_prediction.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, UKHLS_PANEL_DIR)

MEASURE_OF = {
    "pcs-ar1": "sf12pcs_dv", "pcs-ar1-ho": "sf12pcs_dv",
    "physgrm-base": "theta_phys_full", "physgrm-ar1": "theta_phys_full",
    "physgrm-ar1-ho": "theta_phys_full",
    "combgrm-base": "theta_combined", "combgrm-ar1": "theta_combined",
    "combgrm-ar1-ho": "theta_combined",
    "grm-ssm": "grm_theta", "physfunc-ssm": "theta_phys_func", "physfull-ssm": "theta_phys_full",
    "multidim-baseline": "theta_phys_func", "multidim-holdout": "theta_phys_func",
    "multidim-ar1": "theta_phys_func", "multidim-ar1-holdout": "theta_phys_func",
}


def logistic_fit(X, y, iters=60, ridge=1e-4):
    """Newton-IRLS; small ridge keeps the Hessian invertible."""
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
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    apply_style()
    post = pd.read_parquet(ARTIFACTS_DIR / "descriptives" / "class_posteriors.parquet")
    panel = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
                            columns=["pidp", "wave", "age", "sf12pcs_dv",
                                     "theta_phys_full", "theta_combined",
                                     "theta_phys_func", "grm_theta"])
    util = pd.read_parquet(INTERIM_DATA_DIR / "utilisation_long.parquet")
    xw = pd.read_csv(UKHLS_PANEL_DIR / "xwavedat.tab", sep="\t",
                     usecols=["pidp", "dcsedfl_dv"], low_memory=False)
    xw["dcsedfl_dv"] = pd.to_numeric(xw["dcsedfl_dv"], errors="coerce")
    xw["died"] = np.where(xw["dcsedfl_dv"].isin([1, 2, 3]),
                          (xw["dcsedfl_dv"] == 1).astype(float), np.nan)

    base = (panel.merge(util[["pidp", "wave", "hosp", "hl2gp"]],
                        on=["pidp", "wave"], how="left")
            .merge(xw[["pidp", "died"]], on="pidp", how="left"))
    base["inpatient"] = np.where(base["hosp"].isna(), np.nan,
                                 (base["hosp"] == 1).astype(float))
    base["a"] = (base["age"] - 55) / 10

    rows = []
    for fit in post["fit"].unique():
        w = post[post["fit"] == fit]
        d = base.merge(w[["pidp", "class1", "class2", "class3"]],
                       on="pidp", how="inner")
        test = (pd.util.hash_pandas_object(d["pidp"], index=False) % 10) >= 7
        measure = MEASURE_OF[fit]

        for outcome, kind, unit in (("died", "binary", "person"),
                                    ("inpatient", "binary", "wave"),
                                    ("hl2gp", "count", "wave")):
            sub = d.dropna(subset=[outcome, "a", measure]).copy()
            if unit == "person":       # one row per person for mortality
                sub = sub.sort_values("age").groupby("pidp").last().reset_index()
                sub_test = (pd.util.hash_pandas_object(sub["pidp"], index=False) % 10) >= 7
            else:
                sub_test = (pd.util.hash_pandas_object(sub["pidp"], index=False) % 10) >= 7
            if len(sub) < 1000:
                continue
            y = sub[outcome].to_numpy(float)
            one = np.ones(len(sub))
            designs = {
                "age": np.column_stack([one, sub["a"], sub["a"] ** 2]),
                "+class": np.column_stack([one, sub["a"], sub["a"] ** 2,
                                           sub["class1"], sub["class2"]]),
                "+measure": np.column_stack([one, sub["a"], sub["a"] ** 2,
                                             sub[measure]]),
                "+class+measure": np.column_stack([one, sub["a"], sub["a"] ** 2,
                                                   sub["class1"], sub["class2"],
                                                   sub[measure]]),
            }
            tr, te = ~sub_test.to_numpy(), sub_test.to_numpy()
            for name, X in designs.items():
                if kind == "binary":
                    b = logistic_fit(X[tr], y[tr])
                    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
                    m = {"auc_in": auc(y[tr], p[tr]), "auc_out": auc(y[te], p[te]),
                         "logloss_in": float(-np.mean(
                             y[tr] * np.log(np.clip(p[tr], 1e-12, 1))
                             + (1 - y[tr]) * np.log(np.clip(1 - p[tr], 1e-12, 1)))),
                         "logloss_out": float(-np.mean(
                             y[te] * np.log(np.clip(p[te], 1e-12, 1))
                             + (1 - y[te]) * np.log(np.clip(1 - p[te], 1e-12, 1))))}
                else:
                    b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                    p = X @ b
                    m = {"rmse_in": float(np.sqrt(np.mean((y[tr] - p[tr]) ** 2))),
                         "rmse_out": float(np.sqrt(np.mean((y[te] - p[te]) ** 2))),
                         "r2_in": float(1 - ((y[tr] - p[tr]) ** 2).sum()
                                        / ((y[tr] - y[tr].mean()) ** 2).sum()),
                         "r2_out": float(1 - ((y[te] - p[te]) ** 2).sum()
                                         / ((y[te] - y[te].mean()) ** 2).sum())}
                rows.append({"fit": fit,
                             "family": "multidim" if fit.startswith("multidim")
                             else "univariate",
                             "outcome": outcome, "predictors": name,
                             "n": len(sub), "n_test": int(te.sum()), **m})
        print(f"  {fit}", flush=True)

    tab = pd.DataFrame(rows)
    out = ARTIFACTS_DIR / "descriptives" / "outcome_prediction.csv"
    tab.to_csv(out, index=False)

    # ---- figure: out-of-sample gain over the age-only baseline -------------
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.6))
    titles = {"died": "mortality\n(AUC gain over age)",
              "inpatient": "in-patient stay\n(AUC gain over age)",
              "hl2gp": "GP visit band\n($R^2$ gain over age)"}
    order = [f for f in tab["fit"].unique()]
    for ax, oc in zip(axes, titles):
        sub = tab[tab["outcome"] == oc]
        col = "auc_out" if oc != "hl2gp" else "r2_out"
        base = {f: sub[(sub.fit == f) & (sub.predictors == "age")][col].iloc[0]
                for f in order}
        x = np.arange(len(order))
        for j, (pred, colr) in enumerate((("+class", CLUSTER[0]),
                                          ("+measure", "#CC5500"),
                                          ("+class+measure", "#7B52AB"))):
            v = [sub[(sub.fit == f) & (sub.predictors == pred)][col].iloc[0]
                 - base[f] for f in order]
            ax.barh(x + (j - 1) * 0.27, v, height=0.25, color=colr,
                    label=pred if oc == "died" else None)
        ax.set_yticks(x, [f"{f}  ({base[f]:.2f})" for f in order], fontsize=6.4)
        ax.set_title(titles[oc], fontsize=9.5)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.grid(True, axis="x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, fontsize=8, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Predicting outcomes the models never saw, out of sample",
                 fontweight="bold", y=1.0)
    fig.text(0.01, -0.01,
             "Bars are the gain over an age-only quadratic, whose own out-of-sample value is in brackets\n"
             "beside each fit. People split 70/30, fitted on the 70 and scored on the 30. Mortality is\n"
             "person-level; the utilisation outcomes are person-wave over waves 7-15. Fits differ in sample,\n"
             "so the age baselines differ and the gains, not the levels, are what compare across rows.",
             fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    figpath = Path(ARTIFACTS_DIR).parent / "measuring_health" / "figures" / "fig_outcome_prediction.png"
    fig.savefig(figpath)
    print(f"\nwrote {out} and {figpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
