"""The K=3 mixtures on the paper's measure, against the K-means types.

For each variant (h, theta) a fit is read at its posterior
mean: class paths alpha_k + beta_k a + gamma_k a^2 on the standardised
channel with a = (age - 55)/10, mapped back to the variant's own units with
the contract's moments. Each person's class posterior follows from the fitted
rows, which gives the class composition of the person-waves observed at each
age and a person-by-person comparison with the partial K-means types of
22_paper_kmeans.py: the health contract (38,963 people), all of whom are in
the K-means sample.

Two fit sets, chosen on the command line:
  base   artifacts/health-base/   independent residuals (default)
  ssm    artifacts/health-ssm/    AR(1) latent state plus one-period
                                  measurement error; the per-class person
                                  likelihood is the same Kalman recursion the
                                  Stan program runs (mixture_gaussian_panel.stan,
                                  ar_mode 2), with age gaps
Outputs carry the set's name: fig_bayes_{base,ssm}.png (h and theta), tab_bayes_{base,ssm}.tex.

Figure (fig_bayes_base.png), one column per variant:
  row 1  Bayesian class paths (solid, width = share) and the K-means type
         means by age (dashed), in the variant's units
  row 2  class composition by age, Bayesian posteriors
  row 3  type composition by age, K-means
  row 4  the cross-tabulation: share of each K-means type in each Bayesian
         class (posterior-weighted)
Table (tab_bayes_base.tex): shares under both methods, the agreement.

`combined` on the command line draws the paper's figure instead (fig_bayes_mixtures.png, theta;
with --h, fig_bayes_mixtures_h.png for the appendix): independent residuals left, AR(1) plus
"spike" right, class paths against the K-means type means above the class composition. It
needs both fit sets and the K-means labels, and reads nothing the per-set runs write.
Also writes paper_bayes_base_classes.csv and paper_bayes_base_posteriors.parquet.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import (  # noqa: E402
    DESC, FIG, K, MAX_AGE, MENTAL_COL, MIN_AGE, MULTIDIM_CONTRACT, TAB, VARIANTS, bayes_fit, load_labels, load_measure,
    observed_class_means, write_table)
from _style import CLUSTER, INK2, apply_style  # noqa: E402

KMEANS_RED = ["#a50f15", "#ef3b2c", "#fc9272"]      # K-means types, worst -> best, in red so they read apart from the classes

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

CONTRACT = PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1"
SET = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("base", "ssm") else "base"
FITS = {v: f"health-{v}-{SET}" for v in ("h", "theta")}
FIT_DIR = ARTIFACTS_DIR / ("health-base" if SET == "base" else "health-ssm")
MIN_SUPPORT = 25


def posteriors(v: str):
    """The fit for one variant at its posterior mean (shared reader in _paper_common)."""
    return bayes_fit(FIT_DIR / FITS[v], v, K)


def main() -> int:
    apply_style()
    d = load_measure()
    ages = np.arange(MIN_AGE, MAX_AGE + 1); A = (ages - 55) / 10
    gk = {"height_ratios": [3.2, 0.7, 0.7, 1.9], "hspace": 0.5}
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 12), gridspec_kw={**gk, "wspace": 0.22})
    table, classes, posts = [], [], []
    for j, (v, lab) in enumerate(VARIANTS):
        r = posteriors(v)
        L = load_labels(v)
        post = r["post"].set_index("pidp")
        both = post.index.intersection(L.index)
        W = post.loc[both, [f"class{k + 1}" for k in range(K)]].to_numpy()
        modal = W.argmax(axis=1)
        km = L.loc[both].to_numpy()
        ari = adjusted_rand_score(km, modal)
        same = float((km == modal).mean())
        # cross-tab: posterior-weighted share of each K-means type in each Bayesian class
        ct = np.vstack([W[km == c].mean(axis=0) for c in range(K)])
        km_share = np.array([(L == c).mean() for c in range(K)])
        print(f"{v}: R-hat {r['rhat']:.4f}; shares Bayes {np.round(r['theta'], 3)} vs K-means {np.round(km_share, 3)}; "
              f"ARI {ari:.3f}, same class {same:.1%}, mean max posterior {W.max(axis=1).mean():.3f}")
        # ---- row 1: paths
        ax = axes[0, j]
        s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
        t = s.groupby(["cluster", "age"])[v].agg(["mean", "count"]).reset_index()
        obs = observed_class_means(r["post"], v, K)
        for k in range(K):
            mu = r["mom"]["mean"] + r["mom"]["sd"] * (r["coef"][k, 0] + r["coef"][k, 1] * A + r["coef"][k, 2] * A ** 2)
            ax.plot(ages, mu, color=CLUSTER[k], lw=1.2 + 4 * r["theta"][k], label=f"class {k + 1} ({r['theta'][k]:.0%})")
            ax.plot(obs.index, obs[f"class{k + 1}"].to_numpy(), color=CLUSTER[k], lw=1.0, ls=":")
            q = t[(t["cluster"] == k) & (t["count"] >= MIN_SUPPORT)]
            ax.plot(q["age"], q["mean"], color=KMEANS_RED[k], lw=1.1, ls="--", label=f"K-means type {k + 1} ({km_share[k]:.0%})")
            classes.append({"variant": v, "class": k + 1, "share_bayes": r["theta"][k], "share_kmeans": km_share[k],
                            "alpha": r["coef"][k, 0], "beta": r["coef"][k, 1], "gamma": r["coef"][k, 2],
                            "level_30": mu[10], "level_80": mu[60], "sigma": r["sigma"], "rhat": r["rhat"],
                            "rho": r["rho"][k], "sigma_meas": r["sigma_meas"], "set": SET})
        ax.set_title(f"{lab}\n{'AR(1) plus measurement error' if SET == 'ssm' else 'baseline (independent residuals)'}", loc="left", fontsize=9.5)
        ax.legend(fontsize=6.8, loc="lower left", ncols=2); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        # ---- rows 2-3: composition strips
        for row, comp, name in ((1, r["comp"], "Bayesian posteriors"),
                                (2, (s.groupby(["age", "cluster"]).size().unstack(fill_value=0).pipe(lambda c: c.div(c.sum(axis=1), axis=0))), "K-means")):
            ax = axes[row, j]
            comp = comp.reindex(columns=[f"class{k + 1}" for k in range(K)] if row == 1 else range(K))
            ax.stackplot(comp.index, *[comp[c] for c in comp.columns], colors=CLUSTER, alpha=0.9)
            ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(MIN_AGE, MAX_AGE); ax.spines["left"].set_visible(False)
            ax.set_title(f"composition by age, {name}", loc="left", fontsize=8.5)
        axes[2, j].set_xlabel("age")
        # ---- row 4: cross-tab
        ax = axes[3, j]
        im = ax.imshow(ct, cmap="Blues", vmin=0, vmax=1)
        for a in range(K):
            for b in range(K):
                ax.text(b, a, f"{ct[a, b]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if ct[a, b] > 0.6 else "black")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"class {k + 1}" for k in range(K)], fontsize=8)
        ax.set_yticks(range(K)); ax.set_yticklabels([f"type {k + 1}" for k in range(K)], fontsize=8)
        ax.set_xlabel("Bayesian class (posterior weight)"); ax.set_ylabel("K-means type")
        ax.set_title(f"agreement: ARI {ari:.2f}, same class {same:.0%}", loc="left", fontsize=9)
        ax.grid(False)
        table.append([lab, " / ".join(f"{x:.2f}" for x in r["theta"]), " / ".join(f"{x:.2f}" for x in km_share),
                      f"{ari:.2f}", f"{100 * same:.0f}\\%", f"{W.max(axis=1).mean():.2f}", f"{r['rhat']:.3f}"]
                     + ([" / ".join(f"{x:.2f}" for x in r["rho"]), f"{r['sigma_meas']:.3f}"] if SET == "ssm" else []))
        posts.append(r["post"].assign(variant=v))
    fig.text(0.01, -0.01,
             ("Baseline K = 3 quadratic growth mixtures, independent residuals," if SET == "base" else
              "K = 3 quadratic growth mixtures with AR(1) plus measurement error (an AR(1) latent state and a one-period measurement error),")
             + "\non the health contract (38,963 people, 334,194 person-ages), parameters at the posterior mean; class 1 is worst "
             "health.\nK-means types from 22_paper_kmeans.py on the same rows. Paths are in each variant's own units. Dotted: the "
             "observed class mean at each age,\nthe measure averaged over the people observed there weighted by their posterior class "
             "probabilities. Dashed red: the K-means type means.\nComposition: class shares of the person-waves observed at each age. "
             "Cross-tab rows sum to one.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / f"fig_bayes_{SET}.png")
    header = ["variant", "shares, mixture", "shares, K-means", "ARI", "same class", "mean max posterior", "max $\\hat R$"]
    if SET == "ssm":
        header += ["$\\rho$ by class", "$\\sigma_{\\text{meas}}$"]
    write_table(TAB / f"tab_bayes_{SET}.tex", header, table, colspec="lllrrrr" + ("lr" if SET == "ssm" else ""))
    pd.DataFrame(classes).to_csv(DESC / f"paper_bayes_{SET}_classes.csv", index=False)
    pd.concat(posts).to_parquet(DESC / f"paper_bayes_{SET}_posteriors.parquet", index=False)
    print(pd.DataFrame(classes).round(3).to_string(index=False))
    print(f"wrote fig_bayes_{SET}.png, tab_bayes_{SET}.tex")
    return 0


# --mental: the univariate fits on the mental GRM alone (clustering/runs/mental_queue.py), on the
# multidim contract's rows, with a third column for the fit with birth-decade shifts
MENTAL_FITS = [("base", "independent residuals", ARTIFACTS_DIR / "mental-health" / "mental-theta-base"),
               ("ssm", 'AR(1) + "spike"', ARTIFACTS_DIR / "mental-health" / "mental-theta-ssm"),
               ("ssm-cohort", 'AR(1) + "spike", cohort shifts', ARTIFACTS_DIR / "mental-health" / "mental-theta-ssm-cohort")]


def combined(v: str = "theta") -> int:
    """The paper's mixture figure: one variant, independent residuals (left) and AR(1) plus
    "spike" (right). Top: class paths (solid, width = share), observed class means (dotted)
    and the K-means type means (dashed red); bottom: the posterior class composition of the
    person-waves observed at each age. Reads both fit sets directly. ``v == "mental"`` draws
    the mental GRM's fits, with the cohort fit as a third column (paths at the 1950s)."""
    apply_style()
    mental = v == "mental"
    d = load_measure([v])
    ages = np.arange(MIN_AGE, MAX_AGE + 1); A = (ages - 55) / 10
    L = load_labels(v)
    s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    t = s.groupby(["cluster", "age"])[v].agg(["mean", "count"]).reset_index()
    km_share = np.array([(L == c).mean() for c in range(K)])
    lab = {"theta": r"$\theta$", "h": "$h$", "mental": r"mental $\theta$"}[v]
    if mental:
        fits = [(fs, name, fd) for fs, name, fd in MENTAL_FITS if (fd / "run_summary.json").exists()]
    else:
        fits = [(fs, name, ARTIFACTS_DIR / ("health-base" if fs == "base" else "health-ssm") / f"health-{v}-{fs}")
                for fs, name in (("base", "independent residuals"), ("ssm", 'AR(1) + "spike"'))]
    nc = len(fits)
    fig, axes = plt.subplots(2, nc, figsize=(5 * nc, 7.2), gridspec_kw={"height_ratios": [3.2, 0.7], "hspace": 0.32, "wspace": 0.2},
                             sharey="row", squeeze=False)
    for j, (fs, name, fit_dir) in enumerate(fits):
        r = (bayes_fit(fit_dir, v, K, contract=MULTIDIM_CONTRACT, col=MENTAL_COL) if mental else bayes_fit(fit_dir, v, K))
        obs = observed_class_means(r["post"], MENTAL_COL if mental else v, K, rows=r["obs_rows"])
        if mental and fs == "ssm":
            r["post"].to_parquet(DESC / "paper_bayes_mental_posteriors.parquet", index=False)
        ax = axes[0, j]
        for k in range(K):
            mu = r["mom"]["mean"] + r["mom"]["sd"] * (r["coef"][k, 0] + r["coef"][k, 1] * A + r["coef"][k, 2] * A ** 2)
            lk = f"class {k + 1} ({r['theta'][k]:.0%}" + (f", $\\rho$ {r['rho'][k]:.2f})" if fs.startswith("ssm") else ")")
            ax.plot(ages, mu, color=CLUSTER[k], lw=1.2 + 4 * r["theta"][k], label=lk)
            ax.plot(obs.index, obs[f"class{k + 1}"].to_numpy(), color=CLUSTER[k], lw=1.0, ls=":")
            q = t[(t["cluster"] == k) & (t["count"] >= MIN_SUPPORT)]
            ax.plot(q["age"], q["mean"], color=KMEANS_RED[k], lw=1.1, ls="--", label=f"K-means type {k + 1} ({km_share[k]:.0%})" if j == 0 else None)
        ax.set_title(f"({'abc'[j]}) {lab}, {name}", loc="left", fontsize=10)
        ax.legend(fontsize=7, loc="best" if mental else "lower left", ncols=2 if j == 0 else 1); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        ax = axes[1, j]
        comp = r["comp"].reindex(columns=[f"class{k + 1}" for k in range(K)])
        ax.stackplot(comp.index, *[comp[c] for c in comp.columns], colors=CLUSTER, alpha=0.9)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(MIN_AGE, MAX_AGE); ax.spines["left"].set_visible(False)
        ax.set_title("class composition by age", loc="left", fontsize=8.5); ax.set_xlabel("age")
        print(f"{v} {fs}: shares {np.round(r['theta'], 3)}" + (f", rho {np.round(r['rho'], 2)}" if fs.startswith("ssm") else "")
              + (f", rhat {r['rhat']:.3f}") + (f"; decade shifts (vs 1950s) {np.round(r['cohort_shift'], 2)}" if "cohort_shift" in r else ""))
    sample = ("the mental GRM on the multidim health contract (38,181 people with a mental score on every row, 325,822 person-ages)" if mental
              else f"{'theta' if v == 'theta' else 'h'} on the health contract (38,963 people, 334,194 person-ages)")
    footer = (f"K = 3 quadratic growth mixtures on {sample}, parameters at the posterior mean; "
             "class 1 is worst health. Left: independent residuals;\nmiddle" + ("" if mental else "/right") + ": an AR(1) latent state plus a one-period \"spike\", with the class persistence in the legend"
             + ("; right: the same with one level shift per birth decade, common to the classes, paths and observed means at the 1950s, every score net of its decade shift. " if mental else ". ")
             + "Line width is proportional to the class share. Dotted: the observed class mean at each age, the measure averaged over\nthe people observed there weighted by their "
             "posterior class probabilities. Dashed red: the partial K-means type means on the same rows. Bottom: the posterior class composition of the person-waves observed at each age.")
    if mental:   # wrapped to the figure's width
        footer = "\n".join(textwrap.wrap(footer.replace("\n", " "), 75 * nc))
    fig.text(0.01, 0.0, footer, fontsize=7.2, color=INK2, va="top")
    out = FIG / ("fig_bayes_mixtures.png" if v == "theta" else f"fig_bayes_mixtures_{v}.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    if "combined" in sys.argv:
        sys.exit(combined("mental" if "--mental" in sys.argv else "h" if "--h" in sys.argv else "theta"))
    sys.exit(main())
