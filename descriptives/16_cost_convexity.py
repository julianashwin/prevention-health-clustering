"""Stage 4a: convexity with pounds on the vertical axis, and what it rests on.

Cost stays in pounds throughout. Convexity of cost in health is the
non-linearity the framework is about, and it is a statement about mean cost in
pounds; logging cost would change the estimand to something like a geometric
mean and would need an arbitrary constant for the 23% of person-waves at zero.

Convexity is the quadratic term in the standardised measure, fitted on the
rows, with standard errors clustered on the person. The sign survives any
positive linear rescaling of pounds, which is the only rescaling a cost in
pounds admits. Its size is per standard deviation of the health ruler, so it
differs across rulers; its sign is what the sensitivity grid tests.

Measures: the health measure on theta and on h (04_build_health.py, carried by
the cost index), and for comparison the UKHLS PCS, the SF-6D utility and the
mental-health theta. Then the sensitivities the plan set out, all on theta:
spell rule, maternity, winsorising, condition weights and the top band.

Outputs: measuring_health/figures/fig_cost_convexity.png and the same file in paper/figures/,
         artifacts/descriptives/cost_convexity.csv,
         artifacts/descriptives/cost_sensitivity.csv
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
from prevention_health_clustering.measures.cost import UnitCosts, build_cost_index

MEASURES = {"theta": "health $\\theta$",
            "h": "health $h$",
            "sf12pcs_dv": "UKHLS PCS",
            "sf6d_utility": "SF-6D utility",
            "theta_ment": "MENT $\\theta$"}
COLORS = {"theta": BLUE, "h": VERM,
          "sf12pcs_dv": ORANGE, "sf6d_utility": GREEN, "theta_ment": "#7B4EA8"}


def quad_levels(z, y, groups):
    """Quadratic in z fitted on the rows; coefficient and person-clustered t."""
    X = np.column_stack([np.ones(len(z)), z, z**2])
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ (X.T @ y)
    u = y - X @ b
    idx = np.argsort(groups, kind="stable")
    Xs, us, gs = X[idx], u[idx], groups[idx]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    meat = sum(np.outer(v, v) for v in (Xs[a:e].T @ us[a:e] for a, e in zip(edges[:-1], edges[1:])))
    se = np.sqrt(np.diag(XtX @ meat @ XtX))
    return float(b[2]), float(b[2] / se[2])


def decile_means(z, y, k=10):
    """Mean cost in pounds within k bins of z."""
    b = np.clip((pd.Series(z).rank(pct=True) * k).astype(int), 0, k - 1)
    return pd.DataFrame({"z": z, "y": y, "b": b}).groupby("b").agg(
        z=("z", "mean"), y=("y", "mean"))


def main() -> int:
    apply_style()
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet")
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=["pidp", "wave",
                 "sf12pcs_dv", "sf6d_utility", "theta_ment"])
    d = d.merge(panel, on=["pidp", "wave"], how="left")
    d = d[d["flat_cost_total"].notna() & d["age"].notna()]
    d = d[d["age"].between(20, 90)]
    keep = d[list(MEASURES)].notna().all(axis=1)
    d = d[keep].copy()
    print(f"common sample: {len(d):,} person-waves, {d['pidp'].nunique():,} "
          f"people; mean index £{d['flat_cost_total'].mean():,.0f}")

    # ---- 1. convexity in pounds, across measures ---------------------------
    rows, curves = [], {}
    y = d["flat_cost_total"].to_numpy(float)
    gid = d["pidp"].to_numpy()
    ycap = np.minimum(y, np.quantile(y, 0.99))
    for m, lab in MEASURES.items():
        z = ((d[m] - d[m].mean()) / d[m].std()).to_numpy()
        q, t = quad_levels(z, y, gid)
        qc, tc = quad_levels(z, ycap, gid)
        g = decile_means(z, y)
        curves[m] = g
        rows.append({"measure": m, "label": lab, "quad_level": q, "t_level": t,
                     "quad_capped": qc, "t_capped": tc,
                     "bottom_decile": g["y"].iloc[0], "top_decile": g["y"].iloc[-1],
                     "ratio": g["y"].iloc[0] / g["y"].iloc[-1]})
    tab = pd.DataFrame(rows)
    print("\nconvexity of the cost index in pounds, by measure (t clustered on the person):")
    print(tab[["label", "quad_level", "t_level", "quad_capped", "t_capped",
               "bottom_decile", "top_decile", "ratio"]].round(2).to_string(index=False))
    print("\n  quad_level > 0 = convex in pounds: an extra SD of poor health costs more when already unwell")

    # ---- 2. sensitivity, on the headline measure ---------------------------
    z0 = ((d["theta"] - d["theta"].mean()) / d["theta"].std()).to_numpy()
    w = d["cond_weight_filled"] if "cond_weight_filled" in d else None
    variants = [
        ("base: one spell, no maternity, flat", dict()),
        ("spells: two if stay > 5 nights", dict(spell_rule="two")),
        ("spells: one per 5 nights", dict(spell_rule="per_los")),
        ("maternity included", dict(exclude_maternity=False)),
        ("condition-weighted spells", dict(condition_weights=w)),
        ("top band = 30 contacts", dict(top_band=30.0)),
    ]
    srows = []
    for name, kw in variants:
        c = build_cost_index(d, **kw)["cost_total"].to_numpy(float)
        for wins, tag in ((False, ""), (True, " + winsorised p99")):
            cc = c.copy()
            if wins:
                cc = np.minimum(cc, np.quantile(cc, 0.99))
            q, t = quad_levels(z0, cc, gid)
            g = decile_means(z0, cc)
            srows.append({"variant": name + tag, "mean": cc.mean(),
                          "quad_level": q, "t_level": t,
                          "ratio": g["y"].iloc[0] / g["y"].iloc[-1]})
    sens = pd.DataFrame(srows)
    print("\nsensitivity of the convexity conclusion (health measure theta):")
    print(sens.round(3).to_string(index=False))
    print(f"\n  convex in pounds in every variant: {(sens['quad_level'] > 0).all()}, "
          f"smallest t {sens['t_level'].min():.1f}")
    print(f"  health ratio spans {sens['ratio'].min():.1f}x to "
          f"{sens['ratio'].max():.1f}x across variants")

    out_dir = ARTIFACTS_DIR / "descriptives"
    tab.to_csv(out_dir / "cost_convexity.csv", index=False)
    sens.to_csv(out_dir / "cost_sensitivity.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))

    ax = axes[0]
    for m in MEASURES:
        g = curves[m]
        ax.plot(np.arange(1, 11) * 10 - 5, g["y"], "o-", ms=4, lw=1.8,
                color=COLORS[m], label=MEASURES[m])
    ax.set_xlabel("percentile of the measure (low = worst health)")
    ax.set_ylabel("cost index, £ per person-year")
    ax.set_title("(a) Cost against health, five measures", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(True)

    ax = axes[1]
    for m in MEASURES:
        g = curves[m]
        ax.plot(g["z"], g["y"], "o-", ms=4, lw=1.8, color=COLORS[m], label=MEASURES[m])
    ax.set_xlabel("standardised measure (pooled sd units)")
    ax.set_ylabel("cost index, £ per person-year")
    ax.set_title("(b) The same curves per SD: the ruler sets the bend", fontsize=10)
    ax.grid(True)

    ax = axes[2]
    yy = np.arange(len(sens))[::-1]
    cols = [VERM if "winsor" in v else BLUE for v in sens["variant"]]
    ax.barh(yy, sens["quad_level"], color=cols, height=0.7)
    ax.set_yticks(yy, [v.replace(" + winsorised p99", " (w)")
                       for v in sens["variant"]], fontsize=6.4)
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set_xlabel("quadratic term, £ per SD$^2$ of $\\theta$")
    ax.set_title("(c) Convexity survives every costing variant", fontsize=10)
    ax.grid(True, axis="x")

    fig.suptitle("Stage 4: is the cost index convex in health, and does the "
                 "conclusion depend on the assumptions?",
                 fontweight="bold", y=1.005)
    fig.text(0.01, -0.02,
             "Cost index, waves 7-15, ages 20-90, common sample on all five measures. Panel (c) varies the spell rule,\n"
             "maternity treatment, condition weighting and top-band value; (w) marks the winsorised-at-p99 twin of each.\n"
             "A positive bar means an extra SD of poor health costs more, in pounds, when already unwell. All cost in pounds.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    for out in (ROOT_DIR / "measuring_health" / "figures" / "fig_cost_convexity.png",   # the construction note
                ROOT_DIR / "paper" / "figures" / "fig_cost_convexity.png"):             # the paper
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
