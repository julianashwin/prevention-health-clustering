"""Stage 4b: does the health-cost relationship change with age?

The framework's question is what a unit of health improvement is worth, and
whether that depends on when in life it happens. Two versions, which can
give opposite answers and are routinely confused:

  ABSOLUTE     pounds avoided per SD of health. This is the cost-benefit
               object. It almost has to rise with age, because the base
               level of spending rises with age.
  PROPORTIONAL log points per SD, i.e. the percentage difference in cost
               between someone 1 SD sicker than average and the average.
               This asks whether health DISCRIMINATES more at older ages.

The decomposition is exact in the sense that
    absolute slope  ~=  mean cost at that age  x  proportional slope,
so if the proportional gradient is flat, every bit of age variation in the
absolute gradient is the rising base and nothing about health mattering more.

This comparison is only legitimate because the GRM was fitted multigroup by
single year of age against a pooled N(0,1), so a theta of -1 means the same
latent health at 30 as at 80. Age-standardised measures would build the
answer in.

A third quantity turns out to matter more than either. The cost-health curve
is convex, so a linear slope depends on where the mass sits. Older people sit
further into the steep region, which inflates the absolute slope without the
underlying relationship changing at all. The slope is therefore reported
twice: over the full range, and over the common support z in [-2, 1] where
every age band has substantial mass.

Standard errors are clustered on the person: the same people appear in up to
nine waves.

Outputs: measuring_health/figures/fig_cost_age_interaction.png,
         artifacts/descriptives/cost_age_interaction.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import BLUE, GREEN, INK2, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR)

BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]
LAB = ["20-34", "35-44", "45-54", "55-64", "65-74", "75-90"]
COST = "flat_cost_total"


def ols_cluster(X, y, groups):
    """OLS with cluster-robust (CR0) standard errors on `groups`."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    order = np.argsort(groups, kind="stable")
    Xs, us, gs = X[order], u[order], groups[order]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, z in zip(edges[:-1], edges[1:]):
        s = Xs[a:z].T @ us[a:z]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V))


def main() -> int:
    apply_style()
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet")
    d = d[d[COST].notna() & d["age"].notna() & d["theta_phys_full"].notna()]
    d = d[d["age"].between(20, 90)].copy()
    d["z"] = (d["theta_phys_full"] - d["theta_phys_full"].mean()) / d["theta_phys_full"].std()
    d["logc"] = np.log(d[COST] + 1.0)
    d["band"] = pd.cut(d["age"], [b[0] - 1 for b in BANDS] + [90], labels=LAB)
    print(f"{len(d):,} person-waves, {d['pidp'].nunique():,} people")

    # ---- gradients band by band -------------------------------------------
    SICK = float(np.quantile(d["z"], 0.10))
    rows = []
    for lab in LAB:
        s = d[d["band"] == lab]
        g = s["pidp"].to_numpy()
        X = np.column_stack([np.ones(len(s)), s["z"]])
        b_abs, se_abs = ols_cluster(X, s[COST].to_numpy(float), g)
        b_log, se_log = ols_cluster(X, s["logc"].to_numpy(float), g)
        c = s[s["z"].between(-2, 1)]
        Xc = np.column_stack([np.ones(len(c)), c["z"]])
        b_cs, se_cs = ols_cluster(Xc, c[COST].to_numpy(float),
                                  c["pidp"].to_numpy())
        rows.append({"band": lab, "n": len(s), "people": s["pidp"].nunique(),
                     "mean_cost": s[COST].mean(),
                     "abs_slope": -b_abs[1], "abs_se": se_abs[1],
                     "log_slope": -b_log[1], "log_se": se_log[1],
                     "cs_slope": -b_cs[1], "cs_se": se_cs[1],
                     "share_sickest": float((s["z"] <= SICK).mean())})
    t = pd.DataFrame(rows)
    t["implied_abs"] = t["mean_cost"] * t["log_slope"]
    print("\ngradient in cost per SD of WORSE physical health, by age band:")
    print(t[["band", "n", "mean_cost", "abs_slope", "abs_se", "log_slope",
             "log_se"]].round(3).to_string(index=False))
    print("\n  abs_slope = £ more per SD sicker;  log_slope = log points per SD")

    lo, hi = t["abs_slope"].iloc[0], t["abs_slope"].iloc[-1]
    llo, lhi = t["log_slope"].iloc[0], t["log_slope"].iloc[-1]
    clo, chi = t["cs_slope"].iloc[0], t["cs_slope"].iloc[-1]
    print(f"\n  ABSOLUTE gradient, full range:  £{lo:,.0f} at 20-34 -> "
          f"£{hi:,.0f} at 75-90  ({hi / lo:.1f}x)")
    print(f"  ABSOLUTE, common support:      £{clo:,.0f} -> £{chi:,.0f}"
          f"  ({chi / clo:.2f}x)")
    print(f"  PROPORTIONAL:                  {llo:.3f} -> {lhi:.3f} log points"
          f"  ({lhi / llo:.2f}x)")
    print(f"  share of the band in the sickest pooled decile: "
          f"{t['share_sickest'].iloc[0]:.1%} -> {t['share_sickest'].iloc[-1]:.1%}")

    # ---- formal interaction test ------------------------------------------
    a = (d["age"].to_numpy(float) - 50) / 10.0
    z = d["z"].to_numpy()
    g = d["pidp"].to_numpy()
    print("\ninteraction tests (age centred at 50, in decades; "
          "cluster-robust on person):")
    for name, y in (("cost, £", d[COST].to_numpy(float)),
                    ("log cost", d["logc"].to_numpy(float))):
        X = np.column_stack([np.ones(len(d)), a, a**2, z, z**2, z * a])
        b, se = ols_cluster(X, y, g)
        print(f"  {name:9s}  theta {b[3]:9.4f} ({b[3]/se[3]:6.1f})   "
              f"theta^2 {b[4]:8.4f} ({b[4]/se[4]:6.1f})   "
              f"theta x age {b[5]:8.4f} ({b[5]/se[5]:6.1f})")
        rows.append({"band": f"interaction:{name}", "abs_slope": b[5],
                     "abs_se": se[5]})

    # ---- the same person-type at every age --------------------------------
    print("\ncost at a fixed level of health, by age band "
          "(theta bins, pooled edges):")
    edges = np.quantile(d["z"], [0, .1, .25, .5, .75, .9, 1])
    d["zb"] = pd.cut(d["z"], edges, labels=["p0-10", "p10-25", "p25-50",
                                            "p50-75", "p75-90", "p90-100"],
                     include_lowest=True)
    piv = d.pivot_table(index="band", columns="zb", values=COST,
                        aggfunc="mean", observed=True)
    print(piv.round(0).to_string())
    gap = piv["p0-10"] / piv["p90-100"]
    print("\n  ratio of sickest tenth to healthiest tenth, by age band:")
    print("   " + "  ".join(f"{b}: {v:.1f}x" for b, v in gap.items()))

    pd.DataFrame(rows).to_csv(
        ARTIFACTS_DIR / "descriptives" / "cost_age_interaction.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    cmap = plt.cm.viridis(np.linspace(0.08, 0.88, len(LAB)))
    x = np.arange(len(LAB))
    bins = list(piv.columns)

    ax = axes[0, 0]
    for i, lab in enumerate(LAB):
        ax.plot(range(len(bins)), piv.loc[lab].to_numpy(), "o-", color=cmap[i],
                lw=1.9, ms=4, label=lab)
    ax.set_xticks(range(len(bins)), bins, fontsize=7, rotation=30)
    ax.set_xlabel("bin of $\\theta$ (pooled edges, low = sickest)")
    ax.set_ylabel("cost index, £ per person-year")
    ax.set_title("(a) One curve, not six", fontsize=10)
    ax.legend(fontsize=6.6, frameon=False, title="age", title_fontsize=6.6)
    ax.grid(True)

    ax = axes[0, 1]
    for k, b in enumerate(bins):
        ax.plot(x, piv[b].reindex(LAB).to_numpy(), "o-", lw=1.9, ms=4,
                color=plt.cm.plasma(k / (len(bins) - 1) * 0.85), label=b)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_yscale("log")
    ax.set_xlabel("age band"); ax.set_ylabel("cost index, £ (log scale)")
    ax.set_title("(b) At fixed health, cost barely moves with age", fontsize=10)
    ax.set_ylim(top=ax.get_ylim()[1] * 3.2)
    ax.legend(fontsize=6.4, frameon=False, ncol=3, loc="upper center",
              title="$\\theta$ bin", title_fontsize=6.4)
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(x, gap.reindex(LAB).to_numpy(), "o-", color=ORANGE, lw=2, ms=6)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("age band")
    ax.set_ylabel("sickest tenth / healthiest tenth")
    ax.set_title("(c) The health gap narrows slightly", fontsize=10)
    ax.grid(True)

    ax = axes[1, 0]
    ax.errorbar(x, t["abs_slope"], yerr=1.96 * t["abs_se"], fmt="o-",
                color=VERM, lw=2, ms=6, capsize=3, label="full range")
    ax.errorbar(x, t["cs_slope"], yerr=1.96 * t["cs_se"], fmt="s--",
                color=BLUE, lw=1.8, ms=5, capsize=3,
                label="common support, $z\\in[-2,1]$")
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("age band"); ax.set_ylabel("£ per SD sicker")
    ax.set_title(f"(d) The {hi/lo:.1f}x rise is mostly composition", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(True)

    ax = axes[1, 1]
    ax.errorbar(x, t["log_slope"], yerr=1.96 * t["log_se"], fmt="o-",
                color=GREEN, lw=2, ms=6, capsize=3)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("age band"); ax.set_ylabel("log points per SD sicker")
    ax.set_title("(e) Proportional gradient is flat", fontsize=10)
    ax.grid(True)

    ax = axes[1, 2]
    ax.bar(x, t["share_sickest"] * 100, color=INK2, width=0.62)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_xlabel("age band")
    ax.set_ylabel("% of the band in the sickest pooled decile")
    ax.set_title("(f) What ageing does: moves people down the curve",
                 fontsize=10)
    ax.grid(True, axis="y")

    fig.suptitle("Stage 4: the health-cost relationship is close to "
                 "age-invariant; the health distribution is not",
                 fontweight="bold", y=0.995)
    fig.text(0.01, -0.005,
             "Cost index, waves 7-15. Physical GRM theta is comparable across ages by construction (multigroup by single year of age\n"
             "against a pooled N(0,1)), so a fixed theta means the same latent health at 30 and at 80. Bands are 95% intervals from\n"
             "standard errors clustered on the person. Panel (d): the cost-health curve is convex, so a linear slope depends on where\n"
             "the mass sits; restricting to a range every age band populates removes most of the apparent steepening.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = ROOT_DIR / "measuring_health" / "figures" / "fig_cost_age_interaction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
