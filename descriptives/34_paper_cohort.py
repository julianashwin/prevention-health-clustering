"""Birth-decade cohort effects in the AR(1) plus measurement error fits.

For each variant (h, theta, deficit index), from artifacts/health-ssm-cohort/:
  row 1  class paths of the cohort fit at the 1950s reference (solid) against
         the fit without cohort effects (dashed), in the variant's units
  row 2  observed mean of the variant by age within birth decades, raw
  row 3  the same profiles net of the estimated decade effect: what the
         model attributes to age and type once the level shifts are removed
  row 4  the decade effects relative to the 1950s, posterior mean and 95%
         interval, in the variant's units
The model has one level shift per decade shared across classes and no period
term, so a decade's shift is identified from the overlap of cohorts at each
age. The fit's own reference decade is the earliest (the 1900s, two people);
everything here is re-expressed relative to the 1950s.

Outputs: paper/figures/fig_cohort.png, artifacts/descriptives/paper_cohort_effects.csv
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, DESC, FIG, K, MAX_AGE, MIN_AGE, VARIANTS, load_measure  # noqa: E402
from _style import CLUSTER, INK2, INK3, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

CONTRACT = PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1"
DECADES = [1900, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
REF = 1950
MIN_CELL = 50


def paths(summary_path, mom, A):
    p = json.load(open(summary_path))["params"]; g = lambda n: p[n]["mean"]
    theta = np.array([g(f"theta[{k}]") for k in range(1, K + 1)])
    coef = np.array([[g(f"coef[1,{k},{j}]") for j in (1, 2, 3)] for k in range(1, K + 1)])
    return theta, np.vstack([mom["mean"] + mom["sd"] * (coef[k, 0] + coef[k, 1] * A + coef[k, 2] * A ** 2) for k in range(K)])


def main() -> int:
    apply_style()
    d = load_measure()
    birthy = pd.read_csv(CONTRACT / "long.csv", usecols=["pidp", "birthy"]).groupby("pidp")["birthy"].first()
    d = d[d["pidp"].isin(birthy.index)].copy()
    d["decade"] = (d["pidp"].map(birthy) // 10 * 10).astype(int)
    mom_all = json.load(open(CONTRACT / "manifest.json"))["metric_moments"]
    A = (AGES - 55) / 10
    cmap = plt.get_cmap("viridis")
    dec_col = {dec: cmap(i / (len(DECADES) - 2)) for i, dec in enumerate(DECADES[1:])}
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), gridspec_kw={"hspace": 0.42, "wspace": 0.28})
    rows = []
    for j, (v, lab) in enumerate(VARIANTS):
        mom = mom_all[v]
        tag = f"health-{v}-ssm-cohort"
        files = sorted(glob.glob(str(ARTIFACTS_DIR / "health-ssm-cohort" / tag / "chains" / "*.csv")))
        ce = pd.concat([pd.read_csv(f, comment="#", usecols=[f"cohort_effect.1.{k}" for k in range(1, 11)]) for f in files]).to_numpy()
        ce = ce - ce[:, DECADES.index(REF)][:, None]            # relative to the 1950s, draw by draw
        eff = {dec: (ce[:, i].mean(), *np.percentile(ce[:, i], [2.5, 97.5])) for i, dec in enumerate(DECADES)}
        for dec, (m, lo, hi) in eff.items():
            rows.append({"variant": v, "decade": dec, "effect_std": m, "lo_std": lo, "hi_std": hi,
                         "effect_units": m * mom["sd"], "lo_units": lo * mom["sd"], "hi_units": hi * mom["sd"]})
        theta_c, P_c = paths(ARTIFACTS_DIR / "health-ssm-cohort" / tag / "run_summary.json", mom, A)
        theta_n, P_n = paths(ARTIFACTS_DIR / "health-ssm" / f"health-{v}-ssm" / "run_summary.json", mom, A)
        # class paths at the reference decade: the fit's intercept is at its own reference (1900s), so shift
        shift_ref = mom["sd"] * (-ce[:, DECADES.index(1900)].mean())   # 1950s effect relative to 1900s, in units
        ax = axes[0, j]
        for k in range(K):
            ax.plot(AGES, P_c[k] + shift_ref, color=CLUSTER[k], lw=1.2 + 4 * theta_c[k], label=f"class {k + 1} ({theta_c[k]:.0%}), with cohort")
            ax.plot(AGES, P_n[k], color=CLUSTER[k], lw=1.1, ls="--", label=f"class {k + 1} ({theta_n[k]:.0%}), without")
        ax.set_title(f"{lab}\nclass paths: with cohort effects, at the 1950s (solid); without (dashed)", loc="left", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower left", ncols=2); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        # raw and net decade profiles
        s = d[["pidp", "age", "decade", v]].copy()
        s["net"] = s[v] - s["decade"].map({dec: eff[dec][0] * mom["sd"] for dec in DECADES})
        for row, col, title in ((1, v, "observed mean by birth decade"), (2, "net", "net of the estimated decade effect")):
            ax = axes[row, j]
            for dec in DECADES[1:]:
                q = s[s["decade"] == dec].groupby("age")[col].agg(["mean", "count"])
                q = q[q["count"] >= MIN_CELL]
                ax.plot(q.index, q["mean"].rolling(3, center=True, min_periods=1).mean(), color=dec_col[dec], lw=1.3,
                        label=f"{dec}s" if j == 0 else None)
            ax.set_title(title, loc="left", fontsize=9); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if j == 0 and row == 1:
                ax.legend(fontsize=6.5, ncols=3, loc="lower left")
        ax = axes[3, j]
        x = [dec for dec in DECADES if dec != 1900]
        m = [eff[dec][0] * mom["sd"] for dec in x]; lo = [eff[dec][1] * mom["sd"] for dec in x]; hi = [eff[dec][2] * mom["sd"] for dec in x]
        ax.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o-", color=INK3, lw=1.2, capsize=2, ms=4)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title("decade shift relative to the 1950s (95% interval)", loc="left", fontsize=9)
        ax.set_xlabel("birth decade"); ax.grid(True, axis="y")
        ax.set_ylabel(f"{lab} units")
    for ax in axes[2]:
        ax.set_xlabel("age")
    fig.text(0.01, -0.01,
             "AR(1) plus measurement error K = 3 fits with one level shift per birth decade shared across classes, on the health "
             "contract (38,963 people). Decade effects are re-expressed relative to the 1950s;\nthe 1900s (two people) are "
             "omitted from the bottom row. Rows 2 and 3: age-specific means within decades, cells of at least 50 person-waves, "
             "three-year rolling means; row 3 subtracts each decade's estimated shift.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_cohort.png")
    pd.DataFrame(rows).to_csv(DESC / "paper_cohort_effects.csv", index=False)
    print(pd.DataFrame(rows).pivot(index="decade", columns="variant", values="effect_units").round(3).to_string())
    print("wrote fig_cohort.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
