"""The figures and the properties table for the health measure construction note.

Everything here reads only what data_cleaning/04_build_health.py and
05_build_cost_proxy.py write, so the note's numbers regenerate from the live
chain. The three reported variants (theta, h, fi10) and the appendix weighted
sum (ws) are shown side by side throughout, because the point of the note is
that the same answers carry all of them.

Outputs:
  measuring_health/figures/fig_health_measure.png     the scale and its age profile
  measuring_health/figures/fig_health_variants.png    the variants against each other
  measuring_health/figures/fig_health_cost.png        cost in each variant
  artifacts/descriptives/health_measure_properties.csv
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
    PROCESSED_DATA_DIR,
    ROOT_DIR,
)

MEAS = PROCESSED_DATA_DIR / "measures"
FIG = ROOT_DIR / "measuring_health" / "figures"
COST = "flat_cost_total"
VARIANTS = [("theta", r"$\theta$", BLUE), ("h", "$h$", ORANGE),
            ("fi10", "ten-deficit index", GREEN), ("ws", "weighted sum", PURPLE)]


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
    items = pd.read_csv(MEAS / "health_measure_items.csv")
    prof = pd.read_csv(MEAS / "health_measure_age_profile.csv")
    cost = pd.read_parquet(MEAS / "cost_index.parquet", columns=["pidp", "wave", COST])
    d = scores.merge(cost, on=["pidp", "wave"], how="inner")
    d = d[d[COST].notna()]
    cap = d[COST].quantile(0.99)
    d["cost"] = d[COST].clip(upper=cap)
    print(f"scores {len(scores):,} person-waves; cost sample {len(d):,} "
          f"({d['pidp'].nunique():,} people), mean £{d['cost'].mean():,.0f} capped at £{cap:,.0f}")

    # ---- figure 1: the scale itself ---------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3))
    ax = axs[0]
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

    ax = axs[1]
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
    for col, lab, _ in VARIANTS:
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
    print(f"\nwrote three figures to {FIG} and {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
