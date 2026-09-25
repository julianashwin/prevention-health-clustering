"""Stage 4b: does the health-cost relationship change with age? In pounds.

The framework's question is what a unit of health improvement is worth, and
whether that depends on when in life it happens. Cost is kept in pounds
throughout: convexity of cost in health is already the non-linearity the
framework cares about, and a log or proportional gradient would change the
estimand from mean cost to something else.

The pound gradient per SD of health almost has to rise with age over the full
range, for a reason that has nothing to do with health mattering more: the
cost-health curve is convex, and older people sit further into its steep
region. Three tests separate that composition effect from a genuine change in
the curve:

  1. cost at a fixed level of health, by age, in bins of the pooled health
     distribution. The bins are percentiles, so this test is identical on
     every ruler (theta, h, any monotone rescaling).
  2. the slope over a common support, a window of pooled percentiles every
     age band populates. The same people on every ruler, though the slope is
     per SD of the ruler. It is fragile: single answer patterns hold up to
     11% of the sample, so where the window edge falls moves the young band's
     slope. It is therefore reported across windows, not for one.
  3. a pooled regression of cost on age, age squared, health, health squared
     and health x age. The quadratic in health absorbs the convexity, so the
     interaction asks whether the curve itself tilts with age.

Health is the project's measure (data_cleaning/04_build_health.py), reported
on theta and on h. theta is comparable across ages by construction (fitted
against a pooled N(0,1), scored under the pooled prior), and h is a monotone
function of it, so both are. Standard errors are clustered on the person.

Outputs: measuring_health/figures/fig_cost_age_interaction.png and the same file in paper/figures/,
         artifacts/descriptives/cost_age_interaction.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import BLUE, INK2, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR)

BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]
LAB = ["20-34", "35-44", "45-54", "55-64", "65-74", "75-90"]
COST = "flat_cost_total"
SCALES = [("theta", r"$\theta$", BLUE), ("h", "$h$", ORANGE)]
SUPPORT = (0.10, 0.90)
WINDOWS = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.25, 0.75)]


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
    d = d[d[COST].notna() & d["age"].notna() & d["theta"].notna()]
    d = d[d["age"].between(20, 90)].copy()
    d["band"] = pd.cut(d["age"], [b[0] - 1 for b in BANDS] + [90], labels=LAB)
    d["pct"] = d["theta"].rank(pct=True)          # identical on theta and h
    cap = d[COST].quantile(0.99)
    d["cost_cap"] = d[COST].clip(upper=cap)
    print(f"{len(d):,} person-waves, {d['pidp'].nunique():,} people; "
          f"mean £{d[COST].mean():,.0f}, p99 cap £{cap:,.0f}")
    support = d["pct"].between(*SUPPORT)

    # ---- 2. slopes band by band, per SD of each ruler ----------------------
    rows = []
    for col, name, _ in SCALES:
        z = (d[col] - d[col].mean()) / d[col].std()
        d[f"z_{col}"] = z
        for lab in LAB:
            m = (d["band"] == lab).to_numpy()
            s = d[m]
            g = s["pidp"].to_numpy()
            X = np.column_stack([np.ones(m.sum()), z[m]])
            b, se = ols_cluster(X, s[COST].to_numpy(float), g)
            mc = m & support.to_numpy()
            Xc = np.column_stack([np.ones(mc.sum()), z[mc]])
            bc, sec = ols_cluster(Xc, d.loc[mc, COST].to_numpy(float), d.loc[mc, "pidp"].to_numpy())
            rows.append({"scale": col, "band": lab, "n": int(m.sum()), "mean_cost": s[COST].mean(),
                         "slope": -b[1], "se": se[1], "cs_slope": -bc[1], "cs_se": sec[1],
                         "share_sickest": float((s["pct"] <= 0.10).mean())})
    t = pd.DataFrame(rows)
    print("\n£ more per SD sicker, by age band: full range, and on the common support "
          f"(pooled percentiles {SUPPORT[0]:.0%}-{SUPPORT[1]:.0%}):")
    print(t.pivot(index="band", columns="scale", values=["slope", "cs_slope"]).round(0).to_string())
    for col, _, _ in SCALES:
        u = t[t["scale"] == col].set_index("band")
        print(f"  {col:5s} full range £{u.loc['20-34','slope']:,.0f} -> £{u.loc['75-90','slope']:,.0f} "
              f"({u.loc['75-90','slope']/u.loc['20-34','slope']:.2f}x);  common support "
              f"£{u.loc['20-34','cs_slope']:,.0f} -> £{u.loc['75-90','cs_slope']:,.0f} "
              f"({u.loc['75-90','cs_slope']/u.loc['20-34','cs_slope']:.2f}x)")
    win = []
    for lo_, hi_ in WINDOWS:
        rec = {"window": f"{lo_:.0%}-{hi_:.0%}", "width": hi_ - lo_}
        for col, _, _ in SCALES:
            sl = []
            for a0, a1 in (BANDS[0], BANDS[-1]):
                m = (d["age"].between(a0, a1) & d["pct"].between(lo_, hi_)).to_numpy()
                X = np.column_stack([np.ones(m.sum()), d.loc[m, f"z_{col}"]])
                b, _ = ols_cluster(X, d.loc[m, COST].to_numpy(float), d.loc[m, "pidp"].to_numpy())
                sl.append(-b[1])
            rec[col] = sl[1] / sl[0]
        win.append(rec)
    win = pd.DataFrame(win)
    print("\n  ratio of the 75-90 to the 20-34 slope, by common-support window:")
    print(win.round(2).to_string(index=False))
    sh = t[t["scale"] == "theta"].set_index("band")["share_sickest"]
    print(f"  share of the band in the sickest pooled decile: {sh.iloc[0]:.1%} -> {sh.iloc[-1]:.1%}")

    # ---- 3. the pooled interaction test ------------------------------------
    a = (d["age"].to_numpy(float) - 50) / 10.0
    g = d["pidp"].to_numpy()
    print("\ninteraction test: cost on age, age^2, health, health^2, health x age "
          "(age centred at 50, in decades; health oriented so higher = sicker):")
    inter = []
    for col, _, _ in SCALES:
        z = -d[f"z_{col}"].to_numpy()
        for yname in (COST, "cost_cap"):
            X = np.column_stack([np.ones(len(d)), a, a**2, z, z**2, z * a])
            b, se = ols_cluster(X, d[yname].to_numpy(float), g)
            tag = "uncapped" if yname == COST else "capped p99"
            print(f"  {col:5s} {tag:10s} health £{b[3]:7.1f} ({b[3]/se[3]:5.1f})  "
                  f"health^2 £{b[4]:6.1f} ({b[4]/se[4]:5.1f})  health x age £{b[5]:6.1f} ({b[5]/se[5]:5.1f})")
            inter.append({"scale": col, "cost": tag, "health": b[3], "health_t": b[3] / se[3],
                          "health2": b[4], "health2_t": b[4] / se[4],
                          "health_x_age": b[5], "health_x_age_t": b[5] / se[5]})

    # ---- 1. cost at a fixed level of health --------------------------------
    print("\ncost at a fixed level of health, by age band (pooled percentile bins, "
          "the same on every ruler):")
    d["zb"] = pd.cut(d["pct"], [0, .1, .25, .5, .75, .9, 1],
                     labels=["p0-10", "p10-25", "p25-50", "p50-75", "p75-90", "p90-100"],
                     include_lowest=True)
    piv = d.pivot_table(index="band", columns="zb", values=COST, aggfunc="mean", observed=True)
    print(piv.round(0).to_string())
    gap = piv["p0-10"] / piv["p90-100"]
    print("\n  ratio of sickest tenth to healthiest tenth, by age band:")
    print("   " + "  ".join(f"{b}: {v:.1f}x" for b, v in gap.items()))

    out = pd.concat([t.assign(table="slopes"), pd.DataFrame(inter).assign(table="interaction"),
                     win.assign(table="windows")])
    out.to_csv(ARTIFACTS_DIR / "descriptives" / "cost_age_interaction.csv", index=False)

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
    ax.set_xlabel("pooled percentile of health (low = sickest)")
    ax.set_ylabel("cost index, £ per person-year")
    ax.set_title("(a) One curve, not six", fontsize=10)
    ax.legend(fontsize=6.6, frameon=False, title="age", title_fontsize=6.6)
    ax.grid(True)

    ax = axes[0, 1]
    for k, b in enumerate(bins):
        ax.plot(x, piv[b].reindex(LAB).to_numpy(), "o-", lw=1.9, ms=4,
                color=plt.cm.plasma(k / (len(bins) - 1) * 0.85), label=b)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_xlabel("age band"); ax.set_ylabel("cost index, £ per person-year")
    ax.set_title("(b) At fixed health, cost barely moves with age", fontsize=10)
    ax.set_ylim(0, piv.to_numpy().max() * 1.3)
    ax.legend(fontsize=6.4, frameon=False, ncol=3, loc="upper center",
              title="health percentile", title_fontsize=6.4)
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(x, gap.reindex(LAB).to_numpy(), "o-", color=VERM, lw=2, ms=6)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("age band")
    ax.set_ylabel("sickest tenth / healthiest tenth")
    ax.set_title("(c) The health gap narrows slightly", fontsize=10)
    ax.grid(True)

    for ax, key, ttl in ((axes[1, 0], "slope", "(d) Full range: the slope rises with age"),):
        for j, (col, name, c) in enumerate(SCALES):
            u = t[t["scale"] == col]
            se = "se" if key == "slope" else "cs_se"
            ax.errorbar(x + (j - 0.5) * 0.12, u[key], yerr=1.96 * u[se], fmt="o-", color=c,
                        lw=1.9, ms=5, capsize=3, label=f"per SD of {name}")
        ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("age band"); ax.set_ylabel("£ per SD sicker")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7, frameon=False); ax.grid(True)
    ax = axes[1, 1]
    for col, name, c in SCALES:
        ax.plot(win["width"] * 100, win[col], "o-", color=c, lw=1.9, ms=5, label=f"per SD of {name}")
    ax.axhline(1, color=INK2, lw=0.9)
    ax.set_xticks(win["width"] * 100, win["window"], fontsize=7.5)
    ax.set_xlabel("common-support window, pooled percentiles")
    ax.set_ylabel("slope at 75-90 / slope at 20-34")
    ax.set_title("(e) On a common support: near one, window-sensitive", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(True)

    ax = axes[1, 2]
    ax.bar(x, sh.reindex(LAB).to_numpy() * 100, color=INK2, width=0.62)
    ax.set_xticks(x, LAB, fontsize=7.5, rotation=20)
    ax.set_xlabel("age band")
    ax.set_ylabel("% of the band in the sickest pooled decile")
    ax.set_title("(f) What ageing does: moves people down the curve", fontsize=10)
    ax.grid(True, axis="y")

    fig.suptitle("Stage 4: in pounds, the health-cost relationship is close to "
                 "age-invariant; the health distribution is not",
                 fontweight="bold", y=0.995)
    fig.text(0.01, -0.005,
             "Cost index in pounds, waves 7-15. Panels (a)-(c) and (f) use percentiles of the pooled health distribution, so they are the same on every\n"
             "health scale. Panel (d) gives the pound slope per standard deviation of theta and of h over the full range; panel (e) the ratio of the\n"
             "oldest to the youngest band's slope within windows of the pooled distribution, the same people on both scales. 95% intervals clustered on the person.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    for out in (ROOT_DIR / "measuring_health" / "figures" / "fig_cost_age_interaction.png",   # the construction note
                ROOT_DIR / "paper" / "figures" / "fig_cost_age_interaction.png"):             # the paper
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
