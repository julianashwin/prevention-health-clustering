"""The figures and the properties table for the health measure construction note.

Everything here reads only what data_cleaning/04_build_health.py and
05_build_cost_proxy.py write, so the note's numbers regenerate from the live
chain. The three reported variants (theta, h, fi10) and the appendix weighted
sum (ws) are shown side by side throughout, because the point of the note is
that the same answers carry all of them.

A fifth score is computed here rather than in the build: the linear factor
model on the same eight codes treated as continuous (measures/health.py,
one_factor_score), which is what a linear measurement system does with
ordinal items. It is reported beside the others to show which side of the
theta/h contrast a linear model lands on.

Outputs:
  measuring_health/figures/fig_health_measure.png     the scale and its age profile
  measuring_health/figures/fig_health_variants.png    the variants against each other
  measuring_health/figures/fig_health_cost.png        cost in each variant
  measuring_health/figures/fig_health_factor.png      the linear factor score against theta and h
  paper/figures/fig_health_scale.png                  the paper's two-panel version of the first
  artifacts/descriptives/health_measure_properties.csv
  artifacts/descriptives/health_measure_invariance.csv
  artifacts/descriptives/health_measure_factor.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import BLUE, GREEN, INK, INK2, INK3, ORANGE, PURPLE, apply_style  # noqa: E402

from prevention_health_clustering.config import (  # noqa: E402
    ARTIFACTS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
)
from prevention_health_clustering.measures.health import (  # noqa: E402
    build_health_items,
    one_factor_score,
)

MEAS = PROCESSED_DATA_DIR / "measures"
FIG = ROOT_DIR / "measuring_health" / "figures"
COST = "flat_cost_total"
VARIANTS = [("theta", r"$\theta$", BLUE), ("h", "$h$", ORANGE),
            ("fi10", "ten-deficit index", GREEN), ("ws", "weighted sum", PURPLE)]
FACTOR = ("fs", "linear factor score", INK2)
TABLE_VARIANTS = VARIANTS + [FACTOR]
BANDS = [(20, 34), (75, 90)]


def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with person-clustered (CR0) standard errors, as in 17_cost_age_interaction.py."""
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ (X.T @ y)
    u = y - X @ b
    idx = np.argsort(groups, kind="stable")
    Xs, us, gs = X[idx], u[idx], groups[idx]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    meat = sum(np.outer(v, v) for v in (Xs[a:z].T @ us[a:z] for a, z in zip(edges[:-1], edges[1:])))
    return b, np.sqrt(np.diag(XtX @ meat @ XtX))


WINDOWS = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.25, 0.75)]


def band_profile(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean and variance by single year of age, in pooled standard deviations."""
    z = (d[col] - d[col].mean()) / d[col].std()
    g = pd.DataFrame({"age": d["age"], "z": z}).groupby("age")["z"]
    out = pd.DataFrame({"mean": g.mean(), "var": g.var()}).reset_index()
    base = out.loc[out["age"].between(30, 34), "var"].mean()
    out["vrel"] = out["var"] / base
    return out


def main() -> int:
    apply_style()
    scores = pd.read_parquet(MEAS / "health_measure.parquet")
    bank, _ = build_health_items(pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet"),
                                 pd.read_parquet(MEAS / "chronic_conditions.parquet"))
    scores = scores.merge(bank.drop(columns="age"), on=["pidp", "wave"], how="inner", validate="1:1")
    scores["fs"], loadings, eig_share = one_factor_score(scores, orient=scores["theta"].to_numpy())
    items = pd.read_csv(MEAS / "health_measure_items.csv")
    prof = pd.read_csv(MEAS / "health_measure_age_profile.csv")
    cost = pd.read_parquet(MEAS / "cost_index.parquet", columns=["pidp", "wave", COST])
    d = scores.merge(cost, on=["pidp", "wave"], how="inner")
    d = d[d[COST].notna()]
    raw = d[COST].to_numpy(float)          # uncapped, as in 17
    cap = d[COST].quantile(0.99)
    d["cost"] = d[COST].clip(upper=cap)
    print(f"scores {len(scores):,} person-waves; cost sample {len(d):,} "
          f"({d['pidp'].nunique():,} people), mean £{d['cost'].mean():,.0f} capped at £{cap:,.0f}")

    # ---- figure 1: the scale itself -----------------------------------------
    # The note's version has three panels; the paper's (paper/figures/fig_health_scale.png)
    # drops the latent age profile, which the paper does not use.
    def scale_panels(ax_a, ax_b):
        ax = ax_a
        o = scores.sort_values("theta")
        ax.plot(o["theta"], o["h"], color=INK, lw=1.8)
        ax.set_xlim(-3.2, 2.2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("$h$")
        ax.set_title(r"(a) $h$ is the expected score at $\theta$", loc="left")
        tw = ax.twinx()
        tw.hist(scores["theta"], bins=80, color=BLUE, alpha=0.18, density=True)
        tw.set_yticks([])
        tw.spines["right"].set_visible(False)
        ax.grid(True, axis="y")
        ax = ax_b
        for _, r in items.iterrows():
            b = r[[c for c in items.columns if c.startswith("b")]].dropna().to_numpy(float)
            ax.plot(b, np.full(len(b), r["a"]), marker="|", ms=9, lw=0.8, color=INK3)
            ax.annotate(r["item"], (b.max(), r["a"]), textcoords="offset points",
                        xytext=(6, -2.5), ha="left", fontsize=7.6, color=INK2)
        ax.set_xlim(-5.1, 3.4)
        ax.set_xlabel("threshold location")
        ax.set_ylabel("discrimination $a$")
        ax.set_title("(b) where each item speaks", loc="left")
        ax.grid(True, axis="y")

    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3))
    scale_panels(axs[0], axs[1])
    ax = axs[2]
    ax.plot(prof["age"], prof["mu"], color=INK, lw=1.8)
    ax.fill_between(prof["age"], prof["mu"] - prof["sigma"], prof["mu"] + prof["sigma"],
                    color=BLUE, alpha=0.15, lw=0)
    ax.set_xlabel("age")
    ax.set_ylabel(r"$\mu_a$")
    ax.set_title(r"(c) latent age profile, $\pm\sigma_a$", loc="left")
    ax.grid(True, axis="y")
    fig.text(0.01, -0.03,
             "Left: the expected-score curve, with the pooled distribution of theta behind it. Middle: fitted discriminations "
             "against category\nthresholds. Right: the multigroup fit by single year of age — item parameters common "
             "across ages, the pooled latent distribution standardised.", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig_health_measure.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
    scale_panels(axs[0], axs[1])
    fig.text(0.01, -0.03,
             "Left: the expected-score curve that maps theta to h, with the pooled distribution of theta behind it. "
             "Right: each item's fitted\ndiscrimination against its category thresholds.", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "paper" / "figures" / "fig_health_scale.png")
    plt.close(fig)

    # ---- figure 2: the variants against each other -------------------------
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3))
    ax = axs[0]
    q = pd.qcut(scores["h"], 50, labels=False, duplicates="drop")
    for col, lab, c in VARIANTS:
        if col == "h":
            continue
        y = scores.groupby(q)[col].mean()
        y = (y - y.min()) / (y.max() - y.min())
        x = scores.groupby(q)["h"].mean()
        ax.plot(x, y, color=c, lw=1.6, label=lab)
    ax.plot([scores["h"].min(), 1], [0, 1], color=INK3, lw=0.9, ls=":")
    ax.set_xlabel("$h$ (percentile bin means)")
    ax.set_ylabel("variant, rescaled to [0, 1]")
    ax.set_title("(a) the variants are one ordering", loc="left")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")

    ax = axs[1]
    for col, lab, c in VARIANTS:
        p = band_profile(scores, col)
        ax.plot(p["age"], p["mean"], color=c, lw=1.6, label=lab)
    ax.set_xlabel("age")
    ax.set_ylabel("mean, pooled sd units")
    ax.set_title("(b) mean by age", loc="left")
    ax.legend(loc="lower left", ncols=2, fontsize=7.6)
    ax.grid(True, axis="y")

    ax = axs[2]
    for col, lab, c in VARIANTS:
        p = band_profile(scores, col)
        ax.plot(p["age"], p["vrel"], color=c, lw=1.6, label=lab)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xlabel("age")
    ax.set_ylabel("variance, 30–34 = 1")
    ax.set_title("(c) variance by age", loc="left")
    ax.grid(True, axis="y")
    fig.text(0.01, -0.03,
             "Panel (a) plots each variant's mean within fiftieths of h, rescaled to a common range; the dotted line is "
             "equality.\nPanels (b) and (c) standardise each variant on the pooled sample, so the curves are comparable "
             "across variants but not across scales.", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig_health_variants.png")
    plt.close(fig)

    # ---- figure 3: cost ----------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3))
    rows = []
    for ax, (col, lab, c) in zip(axs, [VARIANTS[0], VARIANTS[1], VARIANTS[2]]):
        v = pd.qcut(d[col], 20, labels=False, duplicates="drop")
        m = d.groupby(v).agg(x=(col, "mean"), y=("cost", "mean"))
        z = (m["x"] - d[col].mean()) / d[col].std()
        ax.plot(z, m["y"], marker="o", ms=3.2, color=c, lw=1.5)
        ax.set_xlabel(f"{lab}, pooled sd units")
        ax.set_title(f"cost in {lab}", loc="left")
        ax.grid(True, axis="y")
    axs[0].set_ylabel("mean annual cost, £ (capped at p99)")
    top = max(a.get_ylim()[1] for a in axs)
    for ax in axs:
        ax.set_ylim(0, top)
    fig.text(0.01, -0.03,
             f"Twentieths of each variant on its own cost sample ({len(d):,} person-waves, waves 7–15, ages 20–90). "
             "Costs capped at the 99th\npercentile. Convexity is the same fact in all three: the curve steepens as health "
             "worsens (leftwards).", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig_health_cost.png")
    plt.close(fig)

    # ---- the properties table ---------------------------------------------
    for col, lab, _ in TABLE_VARIANTS:
        z = ((d[col] - d[col].mean()) / d[col].std()).to_numpy()
        X = np.column_stack([np.ones(len(z)), z, z**2])
        b, *_ = np.linalg.lstsq(X, d["cost"].to_numpy(float), rcond=None)
        u = d["cost"].to_numpy(float) - X @ b
        XtX = np.linalg.inv(X.T @ X)
        idx = np.argsort(d["pidp"].to_numpy(), kind="stable")
        Xs, us, gs = X[idx], u[idx], d["pidp"].to_numpy()[idx]
        edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
        meat = sum(np.outer(s, s) for s in
                   (Xs[a:z2].T @ us[a:z2] for a, z2 in zip(edges[:-1], edges[1:])))
        se = np.sqrt(np.diag(XtX @ meat @ XtX))
        p = band_profile(scores, col)
        worst = scores[scores["age"].between(60, 84)]
        floor = (worst[col] <= scores[col].min() + 1e-12).sum()
        rows.append({"variant": lab.replace("$", ""), "distinct values": scores[col].nunique(),
                     "people at the worst value, 60-84": int(floor),
                     "var 30->60": p.set_index("age").loc[58:62, "vrel"].mean()
                     / p.set_index("age").loc[28:32, "vrel"].mean(),
                     "var 60->85": p.set_index("age").loc[83:87, "vrel"].mean()
                     / p.set_index("age").loc[58:62, "vrel"].mean(),
                     "cost curvature": b[2], "t": b[2] / se[2]})
    t = pd.DataFrame(rows)
    out = ARTIFACTS_DIR / "descriptives" / "health_measure_properties.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print("\n" + t.round(3).to_string(index=False))
    # ---- age-invariance of the gradient, in pounds (17's tests) ----------
    rows = []
    a = (d["age"].to_numpy(float) - 50) / 10.0
    g = d["pidp"].to_numpy()
    pct = d["theta"].rank(pct=True).to_numpy()      # the same on every variant
    for col, lab, _ in TABLE_VARIANTS:
        z = ((d[col] - d[col].mean()) / d[col].std()).to_numpy()
        rec = {"variant": lab.replace("$", "")}
        for lo, hi in BANDS:
            m = d["age"].between(lo, hi).to_numpy()
            b, _ = ols_cluster(np.column_stack([np.ones(m.sum()), z[m]]), raw[m], g[m])
            rec[f"{lo}-{hi}"] = -b[1]
        rec["ratio"] = rec["75-90"] / rec["20-34"]
        cs = []
        for w0, w1 in WINDOWS:
            sl = []
            for lo, hi in BANDS:
                m = (d["age"].between(lo, hi).to_numpy() & (pct >= w0) & (pct <= w1))
                b, _ = ols_cluster(np.column_stack([np.ones(m.sum()), z[m]]), raw[m], g[m])
                sl.append(-b[1])
            cs.append(sl[1] / sl[0])
        rec["support ratio min"], rec["support ratio max"] = min(cs), max(cs)
        X = np.column_stack([np.ones(len(z)), a, a**2, -z, z**2, -z * a])
        for yv, tag in ((raw, "uncapped"), (d["cost"].to_numpy(float), "capped")):
            b, se = ols_cluster(X, yv, g)
            rec[f"health x age, {tag}"], rec[f"t, {tag}"] = b[5], b[5] / se[5]
        rows.append(rec)
    inv = pd.DataFrame(rows)
    inv.to_csv(ARTIFACTS_DIR / "descriptives" / "health_measure_invariance.csv", index=False)
    print("\ncost gradient in pounds per sd sicker (uncapped), by age, and the health x age test "
          "(quadratic in health, age centred at 50, per decade):")
    print(inv.round(2).to_string(index=False))

    # ---- the linear factor model --------------------------------------------
    s2 = scores.copy()
    s2["logdef"] = -np.log(np.clip(1.0001 - s2["h"], 1e-4, None))
    print(f"\nlinear factor model on the eight codes: first eigenvalue {eig_share:.3f} of the trace")
    print("  loadings  " + "  ".join(f"{k} {v:.2f}" for k, v in loadings.items()))
    from scipy.stats import pearsonr, spearmanr
    print(f"  correlation with the appendix weighted sum: {pearsonr(s2['fs'], s2['ws'])[0]:.4f}")
    print("  Spearman with " + ", ".join(f"{c} {spearmanr(s2['fs'], s2[c]).statistic:.3f}"
                                       for c in ["theta", "h", "fi10"]))
    gaps = []
    for _, r in items.iterrows():
        b = r[[c for c in items.columns if c.startswith("b")]].dropna().to_numpy(float)
        if len(b) > 1:
            gaps.append((r["item"], np.diff(b).max() / np.diff(b).min()))
    print("  widest/narrowest threshold gap: " + ", ".join(f"{k} {v:.1f}" for k, v in gaps))
    rows = []
    for col, lab in [("theta", "theta"), ("h", "h"), ("fi10", "fi10"), ("ws", "weighted sum"),
                     ("fs", "linear factor score"), ("logdef", "-log(1-h)")]:
        z = (s2[col] - s2[col].mean()) / s2[col].std()
        p = band_profile(s2, col).set_index("age")
        rows.append({"variant": lab, "best pattern, sd above mean": z.max(), "skew": z.skew(),
                     "variance peak age": int(p.loc[25:88, "var"].idxmax()),
                     "share at best pattern": float((s2[col] >= s2[col].max() - 1e-9).mean())})
    fac = pd.DataFrame(rows)
    fac.to_csv(ARTIFACTS_DIR / "descriptives" / "health_measure_factor.csv", index=False)
    print(fac.round(3).to_string(index=False))

    fig, axs = plt.subplots(2, 4, figsize=(13.2, 5.6))
    for j, (col, lab, c) in enumerate([("theta", r"$\theta$, graded response", BLUE),
                                       ("h", "$h$, expected score", ORANGE),
                                       ("fs", "linear factor score", PURPLE),
                                       ("logdef", r"$-\log(1-h)$", GREEN)]):
        z = ((s2[col] - s2[col].mean()) / s2[col].std()).to_numpy()
        ax = axs[0, j]
        ax.hist(z, bins=120, color=c, alpha=0.75, lw=0)
        ax.axvline(z.max(), color=INK, lw=1.2, ls="--")
        ax.set_title(f"({'abcd'[j]}) {lab}", loc="left")
        ax.set_yticks([])
        ax.set_xlim(-4.2, 3.2)
        ax.set_xlabel("pooled sd units")
        ax.text(0.02, 0.93, f"skew {pd.Series(z).skew():+.2f}", transform=ax.transAxes,
                fontsize=7.6, color=INK2)
        ax = axs[1, j]
        p = band_profile(s2, col).set_index("age").loc[25:88, "var"]
        p = p / band_profile(s2, col).set_index("age").loc[30:34, "var"].mean()
        ax.plot(p.index, p.values, color=c, lw=1.8)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.axvline(p.idxmax(), color=INK, lw=0.9, ls=":")
        ax.set_xlabel("age")
        ax.grid(True, axis="y")
        ax.text(0.03, 0.9, f"peaks at {p.idxmax()}", transform=ax.transAxes, fontsize=7.6, color=INK2)
    axs[0, 0].set_ylabel("density")
    axs[1, 0].set_ylabel("variance, 30–34 = 1")
    fig.text(0.005, -0.02,
             "Top: each score's pooled distribution, standardised; the dashed line is the all-best answer pattern. "
             "Bottom: variance by single year of age.\nThe linear factor score is the one-factor model on the eight codes "
             "treated as continuous. It piles the healthy mass against a wall as h does;\ntheta and the log deficit "
             "put the same people further out.", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig_health_factor.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote four figures to {FIG} and three csv files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
