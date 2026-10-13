"""The between / within anatomy of the types, redrawn: the clustering version of Figures 2 and 3.

Two typologies on theta, K = 3: the partial K-means types (22_paper_kmeans.py) and the
AR(1) + "spike" classes (artifacts/health-ssm/health-theta-ssm). For each, at every age a:

  variance split   the cross-sectional variance of theta among the typology's people, split
                   into between types (the variance of the type means under the age-specific
                   composition) and within; for the mixture also the model's own within-class
                   variance, sigma^2 / (1 - rho_k^2) + sigma_meas^2 averaged over the classes
  slopes           d_{j,a}, the type path's slope at a in theta per year (K-means: the forward
                   five-year change of the smoothed path over five; mixture: the derivative of
                   the quadratic), its spread across types sd_j(d_{j,a}) and its correlation with
                   the level, Corr_j(hbar_{j,a}, d_{j,a}), with person-level type shares
  covariance rows  the between-type part of the covariance rows of Figure 2(c): for base ages
                   20, 30, ..., 80, Cov_j(hbar_{j,s}, hbar_{j,s+l}) for l = 0..9 under the
                   composition at s, against the observed rows (paper_exhibit1_moments.csv)

The linear-model layers of Figure 3 (Case 4 on theta, paper_ident_decomposition.csv) are drawn
over the slope panels as horizontal segments per age band: sqrt(V_d) against sd_j(d) and
Corr(H, d) = C_Hd / sqrt(V_H V_d) against Corr_j(hbar, d), so the type-based moments and the
moments-only fit can be read against each other.

Outputs: paper/figures/fig_decomp_alt.png (four rows by two typologies),
         paper/figures/fig_decomp_alt_compact.png (three panels, both typologies overlaid),
         artifacts/descriptives/paper_decomp_alt.csv.

    PYTHONPATH=src .venv/bin/python descriptives/52_paper_decomposition_alt.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, DESC, FIG, K, MAX_AGE, MIN_AGE, bayes_fit, load_labels, load_measure, load_trajectories, smooth_path  # noqa: E402
from _style import BLUE, GREEN, INK2, INK3, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

V = "theta"
MIN_SUPPORT = 25
BASE_AGES = [20, 30, 40, 50, 60, 70, 80]
LAGS = np.arange(0, 10)
BANDS = {"25-40": (25, 40), "40-60": (40, 60), "60-75": (60, 75), "75-90": (75, 90)}
STYLE = {"kmeans": dict(color=BLUE, ls="-", label="K-means types"), "ssm": dict(color=VERM, ls="--", label='AR(1) + "spike" classes')}


def kmeans_moments() -> dict:
    L = load_labels(V)
    d = load_measure([V])
    s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    prof = s.groupby("age")[V].agg(["var", "count"])
    cell = s.groupby(["age", "cluster"])[V].agg(["mean", "count"]).reset_index()
    cell["share"] = cell["count"] / cell.groupby("age")["count"].transform("sum")
    cell["gmean"] = cell.groupby("age").apply(lambda g: np.average(g["mean"], weights=g["share"]), include_groups=False).reindex(cell["age"]).to_numpy()
    between = cell.assign(b=lambda x: x["share"] * (x["mean"] - x["gmean"]) ** 2).groupby("age")["b"].sum().reindex(AGES)
    total = prof["var"].where(prof["count"] >= 100).reindex(AGES)
    traj = load_trajectories(V)
    paths = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(), min_count=MIN_SUPPORT).to_numpy()
                       for c in range(K) for t in [traj[traj["cluster"] == c]]])
    fwd = np.full_like(paths, np.nan)
    fwd[:, :-5] = (paths[:, 5:] - paths[:, :-5]) / 5                      # forward five-year slope, theta per year
    fwd[:, -5:] = fwd[:, [-6]]
    pi = L.value_counts(normalize=True).sort_index().reindex(range(K)).to_numpy()
    comp = cell.pivot(index="age", columns="cluster", values="share").reindex(AGES).ffill().bfill().to_numpy()
    return {"total": total.to_numpy(), "between": between.to_numpy(), "paths": paths, "slopes": fwd, "pi": pi, "comp": comp, "model_within": None, "n": len(L)}


def ssm_moments() -> dict:
    f = bayes_fit(ARTIFACTS_DIR / "health-ssm" / f"health-{V}-ssm", V, K)
    A = (AGES - 55) / 10
    m, sd = f["mom"]["mean"], f["mom"]["sd"]
    paths = np.vstack([m + sd * (c[0] + c[1] * A + c[2] * A ** 2) for c in f["coef"]])
    slopes = np.vstack([sd * (c[1] + 2 * c[2] * A) / 10 for c in f["coef"]])      # theta per year
    rows = f["obs_rows"]
    total = rows.groupby("age")[V].agg(["var", "count"])
    total = total["var"].where(total["count"] >= 100).reindex(AGES).to_numpy()
    comp = f["comp"][[f"class{k + 1}" for k in range(K)]].reindex(AGES).ffill().bfill().to_numpy()
    gmean = (comp * paths.T).sum(axis=1)
    between = (comp * (paths.T - gmean[:, None]) ** 2).sum(axis=1)
    within_k = sd ** 2 * (f["sigma"] ** 2 / (1 - f["rho"] ** 2) + f["sigma_meas"] ** 2)      # stationary within-class variance by class
    model_within = (comp * within_k[None, :]).sum(axis=1)
    return {"total": total, "between": between, "paths": paths, "slopes": slopes, "pi": f["theta"], "comp": comp, "model_within": model_within, "n": len(f["post"])}


def type_moments(mm: dict) -> dict:
    """sd of the slopes, the level-slope covariance and correlation across types, person-level shares."""
    w = mm["pi"]; lev, d = mm["paths"], mm["slopes"]
    ml, md = w @ lev, w @ d
    var_l = w @ (lev - ml) ** 2; var_d = w @ (d - md) ** 2
    cov = w @ ((lev - ml) * (d - md))
    return {"sd_d": np.sqrt(var_d), "cov": cov, "corr": cov / np.sqrt(var_l * var_d)}


def between_rows(mm: dict) -> dict:
    """Cov_j(hbar_s, hbar_{s+l}) under the composition at s, for the base ages."""
    out = {}
    for s0 in BASE_AGES:
        i = s0 - MIN_AGE; w = mm["comp"][i]
        ms = w @ mm["paths"][:, i]
        row = []
        for l in LAGS:
            j = i + l
            if j >= len(AGES):
                break
            mt = w @ mm["paths"][:, j]
            row.append(w @ ((mm["paths"][:, i] - ms) * (mm["paths"][:, j] - mt)))
        out[s0] = np.array(row)
    return out


def exhibit2_bands() -> pd.DataFrame:
    d = pd.read_csv(DESC / "paper_ident_decomposition.csv")
    # the csv stores each layer's contribution to the cell: V_H, (i + j) C_Hd and i j V_d, with i, j in years
    d = d[(d["variant"] == V) & (d["design"] == "main") & (d["case"] == "case4") & (d["cell"] == "V(1)")]
    d = d.assign(CHd=d["CHd"] / (d["i"] + d["j"]), Vd=d["Vd"] / (d["i"] * d["j"]))
    d = d.assign(sd_d=np.sqrt(d["Vd"].clip(lower=0)), corr=d["CHd"] / np.sqrt((d["VH"] * d["Vd"]).clip(lower=1e-12)))
    return d.set_index("band")[["VH", "CHd", "Vd", "sd_d", "corr"]]


def observed_rows() -> pd.DataFrame:
    d = pd.read_csv(DESC / "paper_exhibit1_moments.csv")
    return d[(d["variant"] == V) & (d["kind"] == "row")]


def draw_bands(ax, bands: pd.DataFrame, col: str, label: str):
    first = True
    for name, (lo, hi) in BANDS.items():
        if name in bands.index:
            ax.hlines(bands.loc[name, col], lo, hi, color=INK3, lw=2.2, alpha=0.9, label=label if first else None); first = False


def main() -> int:
    apply_style()
    mm = {"kmeans": kmeans_moments(), "ssm": ssm_moments()}
    tm = {k: type_moments(v) for k, v in mm.items()}
    br = {k: between_rows(v) for k, v in mm.items()}
    bands = exhibit2_bands(); obs = observed_rows()
    sm = lambda x: pd.Series(x).rolling(5, center=True, min_periods=1).mean().to_numpy()  # noqa: E731
    share = {k: sm(v["between"]) / sm(v["total"]) for k, v in mm.items()}
    recs = []
    for k, v in mm.items():
        for i, a in enumerate(AGES):
            recs.append({"typology": k, "age": int(a), "var_total": v["total"][i], "var_between": v["between"][i], "between_share": share[k][i],
                         "model_within": None if v["model_within"] is None else v["model_within"][i], "sd_slope": tm[k]["sd_d"][i], "cov_level_slope": tm[k]["cov"][i], "corr_level_slope": tm[k]["corr"][i]})
    pd.DataFrame(recs).to_csv(DESC / "paper_decomp_alt.csv", index=False)
    for k in mm:
        sgn = next((int(a) for a, c in zip(AGES[1:], tm[k]["corr"][1:]) if c < 0), None)
        print(f"{k}: between share at 30/50/70/85 = " + "/".join(f"{share[k][a - MIN_AGE]:.2f}" for a in (30, 50, 70, 85))
              + f"; sd of slopes min at {int(AGES[np.nanargmin(tm[k]['sd_d'])])} ({np.nanmin(tm[k]['sd_d']):.4f}/yr), max {np.nanmax(tm[k]['sd_d']):.4f}; corr(level, slope) turns negative at {sgn}")
    print("Exhibit 2 bands (Case 4):\n" + bands.round(3).to_string())

    # ---- figure 1: four rows by two typologies
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 14.5), gridspec_kw={"hspace": 0.42, "wspace": 0.22})
    for j, k in enumerate(("kmeans", "ssm")):
        v = mm[k]; name = STYLE[k]["label"]
        ax = axes[0, j]
        ax.fill_between(AGES, 0, sm(v["between"]), color=BLUE, alpha=0.55, lw=0, label="between types")
        ax.fill_between(AGES, sm(v["between"]), sm(v["total"]), color=GREEN, alpha=0.35, lw=0, label="within types")
        ax.plot(AGES, sm(v["total"]), color=VERM, lw=1.8, label="total")
        if v["model_within"] is not None:
            ax.plot(AGES, sm(v["between"]) + sm(v["model_within"]), color=INK2, lw=1.2, ls=":", label="between + model within")
        for a in (30, 50, 70, 85):
            i = a - MIN_AGE; ax.annotate(f"{share[k][i]:.0%}", (a, sm(v['between'])[i] / 2), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.set_title(f"({'ab'[j]}) {name}: variance of $\\theta$ by age (label: between share)", loc="left", fontsize=9.5)
        ax.legend(fontsize=7, loc="upper left"); ax.set_ylim(0, 1.0)
        ax = axes[1, j]
        ax.plot(AGES, sm(tm[k]["sd_d"]), color=BLUE, lw=1.8, label=f"sd$_j(d_{{j,a}})$ across types")
        draw_bands(ax, bands, "sd_d", r"$\sqrt{V_d}$, linear model by band (Figure 3)")
        ax.set_title(f"({'cd'[j]}) spread of the type slopes, $\\theta$ per year", loc="left", fontsize=9.5); ax.set_ylim(0, 0.06); ax.legend(fontsize=7, loc="upper left")
        ax = axes[2, j]
        ax.plot(AGES, sm(tm[k]["corr"]), color=BLUE, lw=1.8, label=r"Corr$_j(\bar h_{j,a}, d_{j,a})$ across types")
        draw_bands(ax, bands, "corr", r"Corr$(H,d)$, linear model by band (Figure 3)")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"({'ef'[j]}) level-slope correlation: diverging (+) or converging (-)", loc="left", fontsize=9.5); ax.legend(fontsize=7, loc="lower left")
        ax = axes[3, j]
        cmap = plt.get_cmap("viridis")
        for n, s0 in enumerate(BASE_AGES):
            c = cmap(0.9 * n / (len(BASE_AGES) - 1))
            q = obs[obs["start"] == s0].sort_values("lag")
            if len(q):
                ax.plot(s0 + q["lag"], q["cov"], color=c, lw=1.0, ls=":", marker="o", ms=2.5)
            ax.plot(s0 + LAGS[:len(br[k][s0])], br[k][s0], color=c, lw=1.8, marker="o", ms=3, label=f"base {s0}")
        ax.plot([], [], color=INK3, lw=1.0, ls=":", label="observed rows (Figure 2)")
        ax.plot([], [], color=INK3, lw=1.8, label="between-type part")
        ax.set_title(f"({'gh'[j]}) covariance rows: between-type part against observed", loc="left", fontsize=9.5); ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6.5, ncols=3, loc="upper left"); ax.set_xlabel("age")
        for ax in axes[:, j]:
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    footer = (f"Theta, K = 3. Left: the partial K-means types ({mm['kmeans']['n']:,} people; smoothed type paths, slopes the forward five-year change over five). Right: the AR(1) + \"spike\" classes "
              f"({mm['ssm']['n']:,} people; quadratic class paths at the posterior mean, slopes their derivative). (a), (b): the cross-sectional variance among the typology's people split into between types "
              "(variance of the type means under the age-specific composition) and within; the dotted line in (b) adds the mixture's own stationary within-class variance to the between part. "
              "(c)-(f): moments of the type slopes across types with person-level shares; grey segments are the linear model's layers of Figure 3 (Case 4) by age band, sqrt(V_d) and C_Hd / sqrt(V_H V_d). "
              "(g), (h): the between-type part of the covariance rows of Figure 2(c), Cov_j(hbar_s, hbar_s+l) under the composition at the base age, with the observed rows dotted. Five-year rolling means in (a)-(f).")
    fig.text(0.01, -0.005, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_decomp_alt.png", bbox_inches="tight")

    # ---- figure 2: compact, both typologies overlaid
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 10.5), gridspec_kw={"hspace": 0.4})
    ax = axes[0]
    for k in ("kmeans", "ssm"):
        ax.plot(AGES, share[k], lw=1.8, **STYLE[k])
    ax.set_ylim(0, 1); ax.set_title("(a) between-type share of the variance of $\\theta$", loc="left", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for k in ("kmeans", "ssm"):
        ax.plot(AGES, sm(tm[k]["sd_d"]), lw=1.8, **STYLE[k])
    draw_bands(ax, bands, "sd_d", r"$\sqrt{V_d}$, linear model by band (Figure 3)")
    ax.set_ylim(0, 0.06); ax.set_title("(b) spread of the type slopes, sd$_j(d_{j,a})$, $\\theta$ per year", loc="left", fontsize=10); ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    for k in ("kmeans", "ssm"):
        ax.plot(AGES, sm(tm[k]["corr"]), lw=1.8, **STYLE[k])
    draw_bands(ax, bands, "corr", r"Corr$(H,d)$, linear model by band (Figure 3)")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-1.05, 1.05)
    ax.set_title("(c) level-slope correlation across types, Corr$_j(\\bar h_{j,a}, d_{j,a})$", loc="left", fontsize=10); ax.legend(fontsize=8, loc="lower left"); ax.set_xlabel("age")
    for ax in axes:
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    footer = ("Theta, K = 3, the partial K-means types (solid) and the AR(1) + \"spike\" classes (dashed). (a) The variance of the type means under the age-specific composition as a share of the cross-sectional "
              "variance among the typology's people. (b), (c) The type slopes (K-means: forward five-year change of the smoothed path over five; mixture: the derivative of the quadratic class path), "
              "their spread across types and their correlation with the level, person-level type shares; grey segments: the linear model's layers of Figure 3 (Case 4) by age band. Five-year rolling means.")
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(footer, 120)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_decomp_alt_compact.png", bbox_inches="tight")
    print(f"wrote fig_decomp_alt.png and fig_decomp_alt_compact.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
