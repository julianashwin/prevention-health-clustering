"""Stage 4a: convexity with pounds on the vertical axis, and what it rests on.

Section 8 established that convexity in health is scale-dependent: on the
probability scale every measure looked convex, and on log-odds the PCS and
the TCC flipped sign. The complaint there was that a binary admission
indicator has no natural vertical scale. The cost index supplies one. Pounds
are the quantity the conceptual framework is actually about, so the question
"is healthcare use convex in health" can now be asked without the answer
being an artefact of the link function.

Two scales are still reported, and they mean different things:
  levels   quadratic in the standardised measure, fitted on the row data.
           Convex = an extra unit of poor health costs more when you are
           already unwell. This is the cost-benefit object.
  logs     quadratic in log mean cost across deciles. Convex here means the
           PROPORTIONAL gradient steepens, which is a much stronger claim
           and the one that survives any monotone rescaling of pounds.

Then the sensitivities the plan set out, all on the headline physical GRM:
spell rule, maternity, winsorising, condition weights and the top band.

Outputs: measuring_health/figures/fig_cost_convexity.png,
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

MEASURES = {"theta_phys_full": "P-FULL $\\theta$",
            "grmh_phys_full": "P-FULL TCC (grmh)",
            "sf12pcs_dv": "UKHLS PCS",
            "sf6d_utility": "SF-6D utility",
            "theta_ment": "MENT $\\theta$"}
COLORS = {"theta_phys_full": BLUE, "grmh_phys_full": VERM,
          "sf12pcs_dv": ORANGE, "sf6d_utility": GREEN, "theta_ment": "#7B4EA8"}


def quad_levels(z, y):
    """Quadratic in z fitted on the rows; returns coefficient and t."""
    X = np.column_stack([np.ones(len(z)), z, z**2])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = (r**2).sum() / (len(y) - 3)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
    return float(b[2]), float(b[2] / se)


def quad_log_deciles(z, y, k=10):
    """Quadratic in log mean cost across k bins of z, plus the bin means."""
    b = np.clip((pd.Series(z).rank(pct=True) * k).astype(int), 0, k - 1)
    g = pd.DataFrame({"z": z, "y": y, "b": b}).groupby("b").agg(
        z=("z", "mean"), y=("y", "mean"))
    lg = np.log(g["y"].to_numpy() + 1.0)
    X = np.column_stack([np.ones(k), g["z"], g["z"]**2])
    c, *_ = np.linalg.lstsq(X, lg, rcond=None)
    return float(c[2]), g


def main() -> int:
    apply_style()
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet")
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=["pidp", "wave", "grmh_phys_full", "sf12pcs_dv",
                 "sf6d_utility", "theta_ment"])
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
    for m, lab in MEASURES.items():
        z = ((d[m] - d[m].mean()) / d[m].std()).to_numpy()
        q, t = quad_levels(z, y)
        ql, g = quad_log_deciles(z, y)
        curves[m] = g
        rows.append({"measure": m, "label": lab, "quad_level": q, "t_level": t,
                     "quad_log_decile": ql,
                     "bottom_decile": g["y"].iloc[0], "top_decile": g["y"].iloc[-1],
                     "ratio": g["y"].iloc[0] / g["y"].iloc[-1]})
    tab = pd.DataFrame(rows)
    print("\nconvexity of the cost index (pounds), by measure:")
    print(tab[["label", "quad_level", "t_level", "quad_log_decile",
               "bottom_decile", "top_decile", "ratio"]].round(3).to_string(index=False))
    print("\n  quad_level > 0  = convex in pounds")
    print("  quad_log_decile > 0 = the PROPORTIONAL gradient steepens too")

    # ---- 2. sensitivity, on the headline measure ---------------------------
    z0 = ((d["theta_phys_full"] - d["theta_phys_full"].mean())
          / d["theta_phys_full"].std()).to_numpy()
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
            q, t = quad_levels(z0, cc)
            ql, g = quad_log_deciles(z0, cc)
            srows.append({"variant": name + tag, "mean": cc.mean(),
                          "quad_level": q, "t_level": t,
                          "quad_log_decile": ql,
                          "ratio": g["y"].iloc[0] / g["y"].iloc[-1]})
    sens = pd.DataFrame(srows)
    print("\nsensitivity of the convexity conclusion (P-FULL theta):")
    print(sens.round(3).to_string(index=False))
    sgn = (sens["quad_level"] > 0).all(), (sens["quad_log_decile"] > 0).all()
    print(f"\n  convex in levels in every variant: {sgn[0]}")
    print(f"  convex in logs   in every variant: {sgn[1]}")
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
        ax.plot(g["z"], np.log(g["y"] + 1), "o-", ms=4, lw=1.8,
                color=COLORS[m], label=MEASURES[m])
    ax.set_xlabel("standardised measure")
    ax.set_ylabel("log cost index")
    ax.set_title("(b) The same curves in logs: still bending", fontsize=10)
    ax.grid(True)

    ax = axes[2]
    yy = np.arange(len(sens))[::-1]
    cols = [VERM if "winsor" in v else BLUE for v in sens["variant"]]
    ax.barh(yy, sens["quad_log_decile"], color=cols, height=0.7)
    ax.set_yticks(yy, [v.replace(" + winsorised p99", " (w)")
                       for v in sens["variant"]], fontsize=6.4)
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set_xlabel("quadratic in log cost across deciles")
    ax.set_title("(c) Convexity in logs survives every variant", fontsize=10)
    ax.grid(True, axis="x")

    fig.suptitle("Stage 4: is the cost index convex in health, and does the "
                 "conclusion depend on the assumptions?",
                 fontweight="bold", y=1.005)
    fig.text(0.01, -0.02,
             "Cost index, waves 7-15, ages 20-90, common sample on all five measures. Panel (c) varies the spell rule,\n"
             "maternity treatment, condition weighting and top-band value; (w) marks the winsorised-at-p99 twin of each.\n"
             "A positive bar means the proportional cost gradient steepens as health worsens.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "measuring_health" / "figures" / "fig_cost_convexity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
