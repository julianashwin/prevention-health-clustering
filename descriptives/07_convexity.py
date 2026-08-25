"""Convexity of healthcare use in each health metric.

Is utilisation convex in health — falling steeply among the ill, flat among
the healthy — and does each metric's cardinalisation carry that shape?
Follows the sandbox's convexity test (quadratic in the standardised measure;
slope asymmetry; binned shapes) extended to the full metric suite.

Outcomes (waves 7-15): P(in-patient stay, last 12 months) from ``hosp``;
out-patient visit band 0-4 from ``hl2gp``. NOTE the UKDA label: ``hl2gp`` is
"Hosp or clinic OUT-PATIENT last 12 months" — the sandbox's "GP band" name
was wrong. UKHLS mainstage has no GP-visit count; the closest is
``servuse1`` ("service use: your local doctor", yes/no, waves 4/6/10/14/15),
included as a supplementary outcome.

All metrics are oriented so higher = better health (GHQ flipped), so on a
declining relationship a POSITIVE quadratic term = convex.

Outputs: docs/figures/fig_convexity.png,
         artifacts/descriptives/convexity_tests.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
    UKHLS_PANEL_DIR,
)

MEASURES = {
    "sf12pcs_dv": "UKHLS PCS",
    "sf12mcs_dv": "UKHLS MCS",
    "PCS_phys_only": "physical-only composite",
    "sf6d_utility": "SF-6D utility",
    "ghq_flipped": "GHQ-12 (flipped)",
    "grm_theta": "GRM original, theta",
    "grmh": "GRM original, TCC",
    "theta_phys_func": "P-FUNC theta",
    "theta_phys_full": "P-FULL theta",
    "grmh_phys_full": "P-FULL, TCC",
    "theta_ment": "MENT theta",
    "grmh_ment": "MENT, TCC",
    "theta_combined": "COMBINED theta",
}
HEADLINE = ["sf12pcs_dv", "sf6d_utility", "grm_theta", "theta_phys_func",
            "theta_phys_full", "theta_ment", "theta_combined"]
COLORS = {"sf12pcs_dv": "#12395B", "sf6d_utility": "#3D9970",
          "grm_theta": "#CC5500", "theta_phys_func": "#2E86C1",
          "theta_phys_full": "#7B52AB", "theta_ment": "#CC79A7",
          "theta_combined": "#E69F00"}


def extract_utilisation() -> pd.DataFrame:
    cache = INTERIM_DATA_DIR / "utilisation_long.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for wi, w in enumerate("abcdefghijklmno", start=1):
        path = UKHLS_PANEL_DIR / f"{w}_indresp.tab"
        header = set(pd.read_csv(path, sep="\t", nrows=0).columns)
        cols = {f"{w}_{s}": s for s in ("hl2gp", "hosp", "servuse1")
                if f"{w}_{s}" in header}
        if not cols:
            continue
        df = pd.read_csv(path, sep="\t", usecols=["pidp"] + list(cols),
                         low_memory=False).rename(columns=cols)
        for c in cols.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan
        df["wave"] = wi
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def main() -> int:
    apply_style()
    cols = ["pidp", "wave", "age", "ghq_likert"] + [
        m for m in MEASURES if m != "ghq_flipped"]
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=list(dict.fromkeys(cols)))
    panel["ghq_flipped"] = -panel["ghq_likert"]
    util = extract_utilisation()
    d = panel.merge(util, on=["pidp", "wave"], how="inner")
    d["inpatient"] = np.where(d["hosp"].isna(), np.nan,
                              (d["hosp"] == 1).astype(float))
    d["gp_used"] = np.where(d["servuse1"].isna(), np.nan, d["servuse1"])

    complete = d[list(MEASURES)].notna().all(axis=1)
    OUTCOMES = {"inpatient": "P(in-patient stay)",
                "hl2gp": "out-patient band (0-4)",
                "gp_used": "P(GP used, 5 waves)"}
    rows = []
    for out, oname in OUTCOMES.items():
        s = d[complete & d[out].notna()]
        print(f"\n=== {oname}: common sample n = {len(s):,}, "
              f"mean = {s[out].mean():.4f} ===")
        for m, label in MEASURES.items():
            z = ((s[m] - s[m].mean()) / s[m].std()).to_numpy()
            y = s[out].to_numpy(float)
            X2 = np.column_stack([np.ones(len(z)), z, z**2])
            b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
            r2 = y - X2 @ b2
            sigma2 = (r2**2).sum() / (len(y) - 3)
            se = np.sqrt(sigma2 * np.linalg.inv(X2.T @ X2)[2, 2])
            lo, hi = z < -1, z > 1
            def _slope(mask):
                A = np.column_stack([np.ones(mask.sum()), z[mask]])
                return float(np.linalg.lstsq(A, y[mask], rcond=None)[0][1])
            slope_lo, slope_hi = _slope(lo), _slope(hi)
            ratio = slope_lo / slope_hi if abs(slope_hi) > 1e-12 else np.inf
            bot = s[out][z <= np.quantile(z, 0.01)].mean()
            top = s[out][z >= np.quantile(z, 0.99)].mean()
            rows.append({"outcome": out, "measure": m, "label": label,
                         "n": len(s), "linear": b2[1], "quadratic": b2[2],
                         "t_quad": b2[2] / se, "slope_bottom": slope_lo,
                         "slope_top": slope_hi, "slope_ratio": ratio,
                         "mean_bottom1pct": bot, "mean_top1pct": top})
        sub = pd.DataFrame([r for r in rows if r["outcome"] == out])
        print(sub[["label", "linear", "quadratic", "t_quad", "slope_ratio",
                   "mean_bottom1pct", "mean_top1pct"]]
              .round(4).to_string(index=False))

    tab = pd.DataFrame(rows)
    out_dir = ARTIFACTS_DIR / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / "convexity_tests.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, out, oname in ((axes[0], "inpatient", "P(in-patient stay)"),
                           (axes[1], "hl2gp", "out-patient band (0-4)")):
        s = d[complete & d[out].notna()]
        for m in HEADLINE:
            r = s[m].rank(pct=True)
            b = np.clip((r * 10).astype(int), 0, 9)
            g = s.groupby(b)[out].mean()
            ax.plot(np.arange(1, 11) * 10 - 5, g, marker="o", ms=3.5, lw=1.7,
                    color=COLORS[m], label=MEASURES[m])
        ax.set_xlabel("percentile of the measure (low = worst health)")
        ax.set_title(oname)
        ax.grid(True, axis="y")
    axes[0].legend(fontsize=6.6)
    ax = axes[2]
    sub = tab[tab["outcome"] == "inpatient"].set_index("measure").loc[HEADLINE]
    ax.barh(np.arange(len(sub))[::-1], sub["quadratic"],
            color=[COLORS[m] for m in sub.index], height=0.62)
    ax.set_yticks(np.arange(len(sub))[::-1],
                  [MEASURES[m] for m in sub.index], fontsize=7.2)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_title("quadratic term, P(in-patient)")
    ax.set_xlabel("coefficient on $z^2$")
    fig.suptitle("Convexity of healthcare use in each health metric "
                 "(waves 7–15, common sample)", fontweight="bold", y=1.02)
    fig.text(0.01, -0.04,
             "Measures oriented so higher = better; on a declining relationship a positive "
             "quadratic = convex. UKHLS records utilisation, not expenditure; hl2gp is "
             "out-patient attendance (the survey has no GP-visit count).",
             fontsize=7.4, color=INK2)
    fig.tight_layout()
    figpath = ROOT_DIR / "docs" / "figures" / "fig_convexity.png"
    fig.savefig(figpath)
    print(f"\nwrote {figpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
