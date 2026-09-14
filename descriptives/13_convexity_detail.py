"""How much of the apparent convexity in healthcare use is real?

Figure 7 plots utilisation against PERCENTILES of each measure. Percentiles
are a rank transform, so equal horizontal steps are equal shares of people,
not equal amounts of health. For a roughly normal theta the bottom decile
spans far more of the health scale than the fifth does, and compressing it
into the same horizontal width makes the curve look steeper on the left than
it is. The formal test in that section is already on the measure's own scale,
so only the picture is affected -- but the picture is what carries the claim.

A second and larger issue is the vertical scale. In-patient stays run at
about 8%, and the logistic curve is convex everywhere below 50%. So a
relationship that is exactly LINEAR in the log-odds still appears convex in
probability. Any claim that health enters convexly needs to say convex in
what.

This script separates the three:
  (a) equal-count (percentile) against equal-width theta bins, same data
  (b) the observed decile pattern against what a LINEAR-in-logit model
      predicts -- if they coincide, the curvature is the link function
  (c) the quadratic coefficient on the probability scale and on the logit
      scale, for several cardinalisations of the same underlying ranking

Outputs: docs/measurement/figures/fig_convexity_detail.png,
         artifacts/descriptives/convexity_detail.csv
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

MEASURES = {
    "theta_phys_full": "GRM physical, theta",
    "grmh_phys_full": "GRM physical, TCC scale",
    "sf12pcs_dv": "UKHLS PCS",
    "sf6d_utility": "SF-6D utility",
}


def logistic_fit(X, y, iters=80, ridge=1e-6):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


def main() -> int:
    apply_style()
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=["pidp", "wave", "age"] + list(MEASURES))
    util = pd.read_parquet(INTERIM_DATA_DIR / "utilisation_long.parquet")
    d = panel.merge(util[["pidp", "wave", "hosp", "hl2gp"]],
                    on=["pidp", "wave"], how="inner")
    d["inpatient"] = np.where(d["hosp"].isna(), np.nan,
                              (d["hosp"] == 1).astype(float))
    d = d.dropna(subset=["inpatient"] + list(MEASURES))
    print(f"sample: {len(d):,} person-waves, in-patient rate "
          f"{d['inpatient'].mean():.3f}")

    rows = []
    for m, label in MEASURES.items():
        z = ((d[m] - d[m].mean()) / d[m].std()).to_numpy()
        y = d["inpatient"].to_numpy()
        one = np.ones(len(z))
        # probability scale (the linear probability model of section 8)
        Xp = np.column_stack([one, z, z**2])
        bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        rp = y - Xp @ bp
        sp = np.sqrt((rp**2).sum() / (len(y) - 3)
                     * np.linalg.inv(Xp.T @ Xp)[2, 2])
        # logit scale
        bl, sel = logistic_fit(Xp, y)
        bl1, _ = logistic_fit(np.column_stack([one, z]), y)
        rows.append({
            "measure": m, "label": label,
            "quad_prob": bp[2], "t_prob": bp[2] / sp,
            "quad_logit": bl[2], "t_logit": bl[2] / sel[2],
            "lin_logit_slope": bl1[1],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(ARTIFACTS_DIR / "descriptives" / "convexity_detail.csv", index=False)
    print(tab.round(4).to_string(index=False))

    # ---------------- figure -------------------------------------------------
    m = "theta_phys_full"
    z = ((d[m] - d[m].mean()) / d[m].std()).to_numpy()
    y = d["inpatient"].to_numpy()
    one = np.ones(len(z))
    b_lin, _ = logistic_fit(np.column_stack([one, z]), y)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    # (a) equal-count vs equal-width bins
    ax = axes[0]
    r = pd.Series(z).rank(pct=True).to_numpy()
    qb = np.clip((r * 10).astype(int), 0, 9)
    g1 = pd.DataFrame({"b": qb, "y": y, "z": z}).groupby("b").agg(
        y=("y", "mean"), z=("z", "mean"))
    ax.plot(np.arange(1, 11) * 10 - 5, g1["y"], "o-", color=BLUE, lw=1.9,
            ms=4, label="equal-count bins (percentiles)")
    edges = np.linspace(z.min(), z.max(), 11)
    wb = np.clip(np.digitize(z, edges[1:-1]), 0, 9)
    g2 = pd.DataFrame({"b": wb, "y": y, "z": z}).groupby("b").agg(
        y=("y", "mean"), z=("z", "mean"), n=("y", "size"))
    g2 = g2[g2["n"] >= 200]
    pctile = [100 * (z < zz).mean() for zz in g2["z"]]
    ax.plot(pctile, g2["y"], "s--", color=ORANGE, lw=1.9, ms=4,
            label="equal-width bins in $\\theta$, placed at their percentile")
    ax.set_xlabel("percentile of the measure")
    ax.set_ylabel("P(in-patient stay)")
    ax.set_title("(a) On a percentile axis: a sharp elbow", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True)

    # (b) observed vs a LINEAR-in-logit model
    ax = axes[1]
    zc = np.linspace(z.min(), z.max(), 200)
    ax.plot(zc, 1 / (1 + np.exp(-(b_lin[0] + b_lin[1] * zc))), color=VERM,
            lw=2.0, label="linear in log-odds")
    ax.plot(g2["z"], g2["y"], "o", color=BLUE, ms=5, label="observed")
    ax.set_xlabel(r"$\theta$, standardised")
    ax.set_ylabel("P(in-patient stay)")
    ax.set_title(r"(b) The same data on the $\theta$ axis: a smooth decay",
                 fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(True)

    # (c) the same data on the log-odds scale
    ax = axes[2]
    obs_logit = np.log(np.clip(g2["y"], 1e-6, 1) / np.clip(1 - g2["y"], 1e-6, 1))
    ax.plot(g2["z"], obs_logit, "o", color=BLUE, ms=5, label="observed")
    ax.plot(zc, b_lin[0] + b_lin[1] * zc, color=VERM, lw=2.0,
            label="linear in log-odds")
    ax.set_xlabel(r"$\theta$, standardised")
    ax.set_ylabel("log-odds of an in-patient stay")
    ax.set_title("(c) On the log-odds scale it is close to straight",
                 fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(True)

    fig.suptitle("How much of the convexity is real?", fontweight="bold", y=1.0)
    fig.text(0.01, -0.01,
             "In-patient stays, waves 7-15, against the physical GRM theta. Panels (a) and (b) show the SAME\n"
             "relationship on two horizontal scales: the binning scheme barely matters (the two series in (a)\n"
             "coincide), but the axis does -- the elbow in (a) is the percentile transform stretching the tails.\n"
             "In (b) and (c) the curve is a logistic with NO quadratic term, so all curvature in (b) is the link.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "measurement" / "figures" / "fig_convexity_detail.png"
    fig.savefig(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
