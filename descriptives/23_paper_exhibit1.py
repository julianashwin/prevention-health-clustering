"""Exhibit 1: the health measure's moments over the life course, model-free.

The framework's Figure 3 on the project's measure: the mean path, the
cross-sectional variance profile, and the rows of the age-covariance matrix,
each row starting on the diagonal at a five-year base-age bin and running nine
years. Two versions, h and theta, one row of panels each, in the measure's own
units. Five-year centred rolling means; cells with n < 100 dropped. Variance
and covariance panels share a vertical scale within a variant, so the drop
from the profile to the row leaving it is the one-year spike.

A second figure (fig_exhibit1_balanced.png) redraws the covariance rows on
balanced samples, each row over the people observed at all ten of its ages,
against the pairwise rows above them, for the appendix.

The paper's figure is theta alone, one column, so the age axis lines up down the panels;
--measures draws the appendix version on h, the frailty index and log frailty, one column each
(fig_exhibit1_measures.png). --frailty draws the main figure on the 31-deficit frailty index and its shifted
log, log(frailty + 1/31), which keeps the zeros at the floor, instead, to
fig_exhibit1_frailty.png, and skips the balanced appendix figure.

Outputs: paper/figures/fig_exhibit1.png, fig_exhibit1_balanced.png
         artifacts/descriptives/paper_exhibit1_moments.csv (kind = profile / row / row_balanced)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import FRAILTY_SUFFIX, MAX_AGE, MIN_AGE, draw_exhibit1_row, profile_rows, profile_rows_balanced, read_scores  # noqa: E402
from _style import INK2, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402

FRAIL = "--frailty" in sys.argv
RULERS = "--measures" in sys.argv
LAB = {"h": "$h$ (expected score, 0-1)", "theta": r"$\theta$ (latent, pooled $N(0,1)$)",
       "frailty": "frailty index (share of 31 deficits)", "logfrailty": "log(frailty + 1/31)"}
# the paper's Exhibit 1 is theta, one column; --measures the appendix version on the other three measures;
# --frailty the two frailty indices; the balanced-rows appendix figure stays on h and theta
VARIANTS = [(v, LAB[v]) for v in (["h", "frailty", "logfrailty"] if RULERS else ["frailty", "logfrailty"] if FRAIL else ["theta"])]
BAL_VARIANTS = [(v, LAB[v]) for v in ("h", "theta")]
SUFFIX = "_measures" if RULERS else FRAILTY_SUFFIX if FRAIL else ""
EXTRA = RULERS or FRAIL
MIN_CELL, ROW_LEN = 100, 9


def main() -> int:
    apply_style()
    d = read_scores(sorted({v for v, _ in VARIANTS + (BAL_VARIANTS if not EXTRA else [])})).drop(columns="wave")
    d = d[d["age"].between(20, 90)].assign(age=lambda x: x["age"].astype(int))
    nc = len(VARIANTS)
    fig, axes = plt.subplots(3, nc, figsize=(5.0 * nc, 9), gridspec_kw={"hspace": 0.33, "wspace": 0.22}, squeeze=False)
    out_rows = []
    for j, (v, label) in enumerate(VARIANTS):
        smooth, rows = profile_rows(d[["pidp", "age", v]].dropna(), v, row_len=ROW_LEN, min_cell=MIN_CELL)
        out_rows.append(smooth.reset_index().assign(variant=v, kind="profile"))
        out_rows.append(rows.assign(variant=v, kind="row"))
        letters = "abc" if nc == 1 else ["adg", "beh", "cfi"][j] if nc == 3 else ["ace", "bdf"][j]
        titles = [f"({letters[0]}) {label}: mean", f"({letters[1]}) variance",
                  f"({letters[2]}) rows of the covariance matrix"]
        draw_exhibit1_row(axes[:, j], smooth, rows, label, legend=(j == 0), titles=titles)
    for ax in axes[2]:
        ax.set_xlabel("age")
    fig.text(0.01, -0.01,
             ("Frailty: the share of 31 equal-weight deficits (six SF-12 physical items graded 0-1, eight impairment areas, the ever-diagnosed conditions), higher = frailer; "
              f"log frailty is log(frailty + 1/31), one whole deficit added, so the {(d['frailty'] == 0).mean():.0%} of person-waves with no deficit sit at the floor.\n" if EXTRA else "") +
             (f"{len(d):,} person-waves, {d['pidp'].nunique():,} people, ages 20-90. Profiles are five-year centred "
              "rolling means; cells with fewer than 100 person-waves\nare dropped. Each covariance row starts on the "
              "diagonal at its five-year base-age bin (dotted: the variance profile)\nand runs nine years. Rows two and "
              "three share a vertical scale within each column." if nc > 1 else
              f"{len(d):,} person-waves, {d['pidp'].nunique():,} people, ages 20-90.\nProfiles are five-year centred rolling means;\n"
              "cells with fewer than 100 person-waves are dropped.\nEach covariance row starts on the diagonal at its\n"
              "five-year base-age bin (dotted: the variance profile)\nand runs nine years. Panels (b) and (c) share a vertical scale."),
             fontsize=7.4, color=INK2, va="top")
    out = ROOT_DIR / "paper" / "figures" / f"fig_exhibit1{SUFFIX}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    if EXTRA:
        pd.concat(out_rows).to_csv(ARTIFACTS_DIR / "descriptives" / f"paper_exhibit1_moments{SUFFIX}.csv", index=False)
        print(f"wrote {out}")
        return 0

    # ---- appendix: the covariance rows, pairwise against balanced ------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), gridspec_kw={"hspace": 0.35, "wspace": 0.22})
    cmap = plt.get_cmap("viridis")
    npeople = {}
    for c, (v, label) in enumerate(BAL_VARIANTS):
        smooth, pair = profile_rows(d, v, row_len=ROW_LEN, min_cell=MIN_CELL)
        bal = profile_rows_balanced(d, v, row_len=ROW_LEN, min_cell=MIN_CELL)
        out_rows.append(bal.assign(variant=v, kind="row_balanced"))
        npeople[v] = bal.groupby("start")["n"].first()
        for r, (name, rr) in enumerate([("pairwise, as in Exhibit 1", pair), ("balanced: observed at all ten ages of the row", bal)]):
            ax = axes[r, c]
            starts = sorted(rr["start"].unique())
            for k, s0 in enumerate(starts):
                seg = rr[rr["start"] == s0].sort_values("lag")
                ax.plot(seg["later_age"], seg["cov"], color=cmap(0.9 * k / max(len(starts) - 1, 1)), lw=1.0, marker="o", ms=2)
            ax.plot(smooth.index, smooth["var"], color=VERM, lw=0.8, ls=":", alpha=0.7)
            ax.set_title(f"({'abcd'[2 * r + c]}) {label.split(' (')[0]}: {name}", loc="left", fontsize=9.5)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if r == 1:
                ax.set_xlabel("age")
        lo = min(pair["cov"].min(), bal["cov"].min(), 0)
        hi = max(pair["cov"].max(), bal["cov"].max(), smooth["var"].max()) * 1.05
        for r in range(2):
            axes[r, c].set_ylim(lo, hi)
    n = npeople["h"]
    fig.text(0.01, -0.01,
             "Rows of the age-covariance matrix leaving the diagonal at five-year base-age bins and running nine years; dotted red is "
             "the smoothed variance profile. Top: each cell over the people\nobserved at both of its ages, as in Exhibit 1. Bottom: each "
             "row over the people observed at every one of its ten ages, which leaves "
             f"{int(n.min()):,} to {int(n.max()):,} people per row and nothing from 80.\nCells with fewer than 100 people dropped; cells pooled "
             "into bins by count. Panels share a vertical scale within each column.",
             fontsize=7.4, color=INK2, va="top")
    out2 = ROOT_DIR / "paper" / "figures" / "fig_exhibit1_balanced.png"
    fig.savefig(out2, bbox_inches="tight")
    pd.concat(out_rows).to_csv(ARTIFACTS_DIR / "descriptives" / "paper_exhibit1_moments.csv", index=False)
    print(f"wrote {out} and {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
