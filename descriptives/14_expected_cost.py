"""If the object of interest is expected cost, what shape does health imply?

A binary outcome forces a bounded scale, and a bounded scale makes curvature
partly mechanical (section 8.1). Expected cost does not have that problem: it
is a ratio quantity with a meaningful zero and meaningful units, so
"convex in pounds" is a substantive claim rather than a convention. That
removes the VERTICAL ambiguity. It does not remove the horizontal one -- a
unit of health still has to be defined -- but it makes the vertical axis
honest.

UKHLS carries no spend, but it carries the physical quantities that spend is
bought with: nights as an in-patient (hospd) and out-patient attendance bands
(hl2gp), waves 7-15. With unit costs that do NOT vary with health, expected
cost is a fixed linear combination of expected nights and expected
attendances, so the SHAPE can be studied in physical units without inventing
prices -- multiplying by a constant cannot change convexity.

The decomposition that matters is
    E[nights] = P(admitted) x E[nights | admitted]
because the two margins may both move with health. If they do, expected cost
is more convex than the admission probability alone, which is all section 8
measured.

Outputs: docs/figures/fig_expected_cost.png,
         artifacts/descriptives/expected_cost.csv
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
    ARTIFACTS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ROOT_DIR)

MEASURE = "theta_phys_full"


def main() -> int:
    apply_style()
    panel = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
                            columns=["pidp", "wave", "age", MEASURE, "grmh_phys_full"])
    util = pd.read_parquet(INTERIM_DATA_DIR / "utilisation_long.parquet")
    d = panel.merge(util[["pidp", "wave", "hosp", "hospd", "hl2gp"]],
                    on=["pidp", "wave"], how="inner")
    d = d[d["hosp"].notna() & d[MEASURE].notna()].copy()
    # nights is asked only of those admitted; a non-admission is zero nights
    d["nights"] = np.where(d["hosp"] == 1, d["hospd"], 0.0)
    d["admitted"] = (d["hosp"] == 1).astype(float)
    d = d[d["nights"].notna()]
    d["z"] = (d[MEASURE] - d[MEASURE].mean()) / d[MEASURE].std()
    print(f"sample {len(d):,} person-waves; admitted {d['admitted'].mean():.3f}; "
          f"mean nights {d['nights'].mean():.3f}; "
          f"mean nights | admitted {d.loc[d.admitted==1,'nights'].mean():.2f}")

    # deciles of health, on the measure's own scale
    d["bin"] = pd.qcut(d["z"], 10, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(z=("z", "mean"), p_adm=("admitted", "mean"),
                             nights=("nights", "mean"),
                             nights_if=("nights", lambda x: x[x > 0].mean()),
                             op=("hl2gp", "mean"), n=("z", "size")).reset_index()
    g["ratio_to_best"] = g["nights"] / g["nights"].iloc[-1]
    print("\nby decile of health (10 = healthiest):")
    print(g.round(3).to_string(index=False))

    # how much of the gradient is each margin? log decomposition
    lo, hi = g.iloc[0], g.iloc[-1]
    tot = np.log(lo["nights"] / hi["nights"])
    ext = np.log(lo["p_adm"] / hi["p_adm"])
    inten = np.log(lo["nights_if"] / hi["nights_if"])
    print(f"\nbottom vs top decile, in logs:")
    print(f"  expected nights   {tot:+.3f}  ({np.exp(tot):.1f}x)")
    print(f"  = admission rate  {ext:+.3f}  ({np.exp(ext):.1f}x)   "
          f"[{100*ext/tot:.0f}% of the gap]")
    print(f"  + nights if admitted {inten:+.3f}  ({np.exp(inten):.1f}x)   "
          f"[{100*inten/tot:.0f}%]")

    # curvature of expected nights, on levels and on logs
    rows = []
    for name, y in (("expected nights", d["nights"].to_numpy()),
                    ("P(admitted)", d["admitted"].to_numpy()),
                    ("out-patient band", d["hl2gp"].fillna(0).to_numpy())):
        z = d["z"].to_numpy()
        X = np.column_stack([np.ones(len(z)), z, z**2])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        se = np.sqrt((r**2).sum() / (len(y) - 3) * np.linalg.inv(X.T @ X)[2, 2])
        # log scale, on the decile means (levels contain zeros)
        gz = g["z"].to_numpy()
        gy = np.log(g["nights"] if name == "expected nights"
                    else g["p_adm"] if name == "P(admitted)" else g["op"])
        Xg = np.column_stack([np.ones(len(gz)), gz, gz**2])
        bg, *_ = np.linalg.lstsq(Xg, gy, rcond=None)
        rows.append({"outcome": name, "quad_level": b[2], "t_level": b[2] / se,
                     "quad_log_decile": bg[2]})
    tab = pd.DataFrame(rows)
    tab.to_csv(ARTIFACTS_DIR / "descriptives" / "expected_cost.csv", index=False)
    print("\ncurvature:")
    print(tab.round(4).to_string(index=False))

    # ---------------- figure -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ax = axes[0]
    ax.plot(g["z"], g["nights"], "o-", color=BLUE, lw=2, ms=5)
    ax.set_xlabel(r"$\theta$, standardised"); ax.set_ylabel("expected nights per year")
    ax.set_title("(a) Expected nights, the cost-relevant quantity", fontsize=10)
    ax.grid(True)

    ax = axes[1]
    ax.plot(g["z"], g["p_adm"] / g["p_adm"].iloc[-1], "o-", color=ORANGE,
            lw=2, ms=5, label="admission rate (extensive)")
    ax.plot(g["z"], g["nights_if"] / g["nights_if"].iloc[-1], "s-", color=VERM,
            lw=2, ms=5, label="nights if admitted (intensive)")
    ax.plot(g["z"], g["nights"] / g["nights"].iloc[-1], "^-", color=BLUE,
            lw=2, ms=5, label="their product")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$, standardised")
    ax.set_ylabel("relative to the healthiest decile")
    ax.set_title("(b) Both margins move with health", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True)

    ax = axes[2]
    ax.plot(g["z"], np.log(g["nights"]), "o", color=BLUE, ms=6, label="observed")
    zc = np.linspace(g["z"].min(), g["z"].max(), 100)
    bb = np.polyfit(g["z"], np.log(g["nights"]), 1)
    ax.plot(zc, np.polyval(bb, zc), color=VERM, lw=2, label="log-linear")
    ax.set_xlabel(r"$\theta$, standardised")
    ax.set_ylabel("log expected nights")
    ax.set_title("(c) Curvature survives on the log scale", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True)

    fig.suptitle("Expected cost: both margins move, and the curvature survives logs",
                 fontweight="bold", y=1.0)
    fig.text(0.01, -0.01,
             "In-patient nights and out-patient bands, waves 7-15, against the physical GRM theta. Unit costs\n"
             "that do not vary with health only rescale these curves, so the shape is the shape of expected\n"
             "cost. Panel (b) is on a log axis, where a straight line is a constant proportional gradient.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "figures" / "fig_expected_cost.png"
    fig.savefig(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
