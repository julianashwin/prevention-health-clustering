"""Class trajectories and shares on the paper's measure, by number of classes.

Main text (fig_class_trajectories_by_K.png): four rows by three columns. Rows are
h under independent residuals, h under AR(1) plus a one-period "spike", theta
under independent residuals, theta under AR(1) plus "spike"; columns are K = 3,
4 and 5. Each panel draws the fitted class paths in the variant's own units,
line width proportional to the class share, with the class share and (under
AR(1)) the class persistence in the legend, and the posterior class composition
of the observed person-waves as a strip beneath. Dotted lines are the observed
class means: at each age, the mean of the measure over the people observed at that
age weighted by their posterior class probabilities (drawn where the summed
weight is at least 25 people), so the eye can check the quadratic. A fit that has not finished
(no run_summary.json) leaves its panel blank and says so.
Appendix (fig_trajectories_fi10.png): K = 3 for the ten-deficit index.

K = 3 comes from what 32_paper_bayes_fits.py wrote (paper_bayes_{set}_classes.csv,
paper_bayes_{set}_posteriors.parquet); K = 4 and 5 are read straight from the fits
in artifacts/health-k45/ with _paper_common.bayes_fit, and their per-person
posteriors cached to paper_bayes_{set}_k{K}_{variant}.parquet.

Outputs: paper/figures/fig_class_trajectories_by_K.png, fig_trajectories_fi10.png,
         artifacts/descriptives/paper_trajectories.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, CONTRACT, DESC, FIG, K, MAX_AGE, MIN_AGE, bayes_fit, observed_class_means  # noqa: E402
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

K45_DIR = ARTIFACTS_DIR / "health-k45"
KS = (3, 4, 5)
SETS = [("base", "independent residuals"), ("ssm", 'AR(1) + "spike"')]
LABEL = {"h": "$h$", "theta": r"$\theta$", "fi10": "ten-deficit index"}
observed_means = observed_class_means
PALETTE = {3: CLUSTER, 4: ["#08306b", "#2171b5", "#4292c6", "#9ecae1"],
           5: ["#08306b", "#08519c", "#2171b5", "#4292c6", "#9ecae1"]}     # worst -> best health
A = (AGES - 55) / 10


def k3_panel(tag: str, v: str):
    """The K = 3 fit as 32 wrote it: shares, rho, paths in the variant's units, composition by age."""
    cl = pd.read_csv(DESC / f"paper_bayes_{tag}_classes.csv")
    cl = cl[cl["variant"] == v].sort_values("class").reset_index(drop=True)
    mom = json.load(open(CONTRACT / "manifest.json"))["metric_moments"][v]
    paths = np.vstack([mom["mean"] + mom["sd"] * (r["alpha"] + r["beta"] * A + r["gamma"] * A ** 2) for _, r in cl.iterrows()])
    post = pd.read_parquet(DESC / f"paper_bayes_{tag}_posteriors.parquet")
    post = post[post["variant"] == v][["pidp"] + [f"class{k + 1}" for k in range(K)]]
    long = pd.read_csv(CONTRACT / "long.csv", usecols=["pidp", "age"])
    comp = long.merge(post, on="pidp").groupby("age")[[f"class{k + 1}" for k in range(K)]].mean()
    return {"K": K, "share": cl["share_bayes"].to_numpy(), "rho": cl["rho"].to_numpy(), "sigma": cl["sigma"].to_numpy(),
            "sigma_meas": cl["sigma_meas"].to_numpy(), "paths": paths, "comp": comp, "rhat": float(cl["rhat"].iloc[0]),
            "observed": observed_means(post, v, K)}


# Chains read for a fit when the others sat in a separate mode. health-h-ssm-k5: chains
# 1, 2 and 4 agree to three decimals (split R-hat 1.002 over the three); chain 3 sits in a
# mode about 310 log-posterior units lower and is dropped.
CHAINS = {"health-h-ssm-k5": [1, 2, 4]}


def kn_panel(tag: str, v: str, Kc: int):
    """A K = 4 or 5 fit, if it has finished; None otherwise."""
    fit_dir = K45_DIR / f"health-{v}-{tag}-k{Kc}"
    if not (fit_dir / "run_summary.json").exists():
        return None
    f = bayes_fit(fit_dir, v, Kc, chains=CHAINS.get(fit_dir.name))
    f["post"].to_parquet(DESC / f"paper_bayes_{tag}_k{Kc}_{v}.parquet", index=False)
    paths = np.vstack([f["mom"]["mean"] + f["mom"]["sd"] * (c[0] + c[1] * A + c[2] * A ** 2) for c in f["coef"]])
    return {"K": Kc, "share": f["theta"], "rho": f["rho"], "sigma": np.full(Kc, f["sigma"]),
            "sigma_meas": np.full(Kc, f["sigma_meas"]), "paths": paths, "comp": f["comp"], "rhat": f["rhat"],
            "observed": observed_means(f["post"], v, Kc)}


def draw(ax, axc, fit, show_rho, ylabel=None):
    Kc = fit["K"]; cols = PALETTE[Kc]
    for k in range(Kc):
        lab = f"{k + 1}: {fit['share'][k]:.0%}" + (f", $\\rho$ {fit['rho'][k]:.2f}" if show_rho else "")
        ax.plot(AGES, fit["paths"][k], color=cols[k], lw=1.0 + 3.5 * fit["share"][k], label=lab)
        obs = fit["observed"][f"class{k + 1}"]
        ax.plot(obs.index, obs.to_numpy(), color=cols[k], lw=1.0, ls=":")
    ax.legend(fontsize=6.6, loc="lower left", ncols=1 if Kc == 3 else 2, handlelength=1.6, columnspacing=0.8)
    ax.grid(True, axis="y")
    ax.set_xlim(MIN_AGE, MAX_AGE)
    ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    comp = fit["comp"]
    axc.stackplot(comp.index, *[comp[c] for c in comp.columns], colors=cols, alpha=0.9)
    axc.set_ylim(0, 1); axc.set_yticks([]); axc.set_xlim(MIN_AGE, MAX_AGE)
    axc.spines["left"].set_visible(False)


def record(rows, tag, v, fit):
    for k in range(fit["K"]):
        rows.append({"set": tag, "variant": v, "K": fit["K"], "class": k + 1, "share": fit["share"][k],
                     "level_20": fit["paths"][k][0], "level_55": fit["paths"][k][35], "level_90": fit["paths"][k][-1],
                     "rho": fit["rho"][k], "sigma": fit["sigma"][k], "sigma_meas": fit["sigma_meas"][k], "rhat": fit["rhat"]})


def main() -> int:
    apply_style()
    rows, missing = [], []
    row_specs = [(v, tag, spec) for v in ("h", "theta") for tag, spec in SETS]
    fig = plt.figure(figsize=(13.5, 16.5))
    gs = fig.add_gridspec(2 * len(row_specs), len(KS), height_ratios=[3.0, 0.55] * len(row_specs), hspace=0.32, wspace=0.16,
                          top=0.97, bottom=0.04)
    for r, (v, tag, spec) in enumerate(row_specs):
        for c, Kc in enumerate(KS):
            ax = fig.add_subplot(gs[2 * r, c]); axc = fig.add_subplot(gs[2 * r + 1, c])
            fit = k3_panel(tag, v) if Kc == 3 else kn_panel(tag, v, Kc)
            if r == 0:
                ax.set_title(f"$K = {Kc}$", fontsize=11, loc="center")
            if fit is None:
                missing.append(f"health-{v}-{tag}-k{Kc}")
                ax.text(0.5, 0.5, "fit still running", ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([]); axc.set_xticks([]); axc.set_yticks([])
                for a_ in (ax, axc):
                    for sp in a_.spines.values():
                        sp.set_visible(False)
                if c == 0:
                    ax.set_ylabel(f"{LABEL[v]}, {spec}", fontsize=9)
                continue
            draw(ax, axc, fit, show_rho=(tag == "ssm"), ylabel=f"{LABEL[v]}, {spec}" if c == 0 else None)
            if r == len(row_specs) - 1:
                axc.set_xlabel("age")
            record(rows, tag, v, fit)
            print(f"{v} {tag} K={Kc}: rhat {fit['rhat']:.4f}, shares {np.round(fit['share'], 3)}"
                  + (f", rho {np.round(fit['rho'], 2)}" if tag == "ssm" else ""))
    dropped = [f"{name}: chains {', '.join(map(str, ch))} of 4" for name, ch in CHAINS.items() if f"health-{name.split('-')[1]}-{name.split('-')[2]}-{name.split('-')[3]}" not in missing]
    fig.text(0.01, 0.005,
             "Quadratic growth mixtures on the health contract (38,963 people, 334,194 person-ages), parameters at the posterior mean, in "
             "each variant's own units; class 1 is worst health. Line width is proportional to the class share; the legend gives\nthe share "
             "and, under AR(1), the class persistence. Dotted: the observed class mean at each age, the measure averaged over the people observed at that "
             "age weighted by their posterior class probabilities\n(where the summed weight is at least 25). The strip beneath each panel is the "
             "posterior class composition of the person-waves observed at each age. Rows: the variant and the stochastic specification,\nindependent "
             'residuals or an AR(1) latent state plus a one-period "spike"; columns: the number of classes.' + (f" Not yet fitted: {', '.join(missing)}." if missing else "")
             + (f" Posterior mean over the chains that share a mode for {'; '.join(dropped)} (the other chain sat in a worse mode)." if dropped else ""),
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_class_trajectories_by_K.png", bbox_inches="tight")
    plt.close(fig)

    # ---- appendix: the deficit index, K = 3 ---------------------------------------
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 0.62], hspace=0.3, wspace=0.2)
    for c, (tag, spec) in enumerate(SETS):
        fit = k3_panel(tag, "fi10")
        ax, axc = fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])
        draw(ax, axc, fit, show_rho=(tag == "ssm"))
        ax.set_title(f"{LABEL['fi10']}, {spec}", loc="left", fontsize=9.5)
        axc.set_xlabel("age")
        record(rows, tag, "fi10", fit)
    fig.text(0.01, -0.02, "As the main-text trajectory figure, for the ten-deficit index at $K = 3$.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_trajectories_fi10.png")
    plt.close(fig)

    t = pd.DataFrame(rows)
    t.to_csv(DESC / "paper_trajectories.csv", index=False)
    print(f"wrote fig_class_trajectories_by_K.png and fig_trajectories_fi10.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
