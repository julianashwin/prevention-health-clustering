"""Benchmark each holdout fit's held-out density against age-only nulls.

Raw held-out log densities are not comparable across channels with different
scales, so each fit is scored against two nulls evaluated on ITS OWN
held-out rows: the marginal N(0,1) (channels are standardised), and an
age-quadratic Gaussian whose coefficients and residual sd are estimated on
that fit's own fitted rows. The gain over the age-quadratic null is the
comparable quantity reported in the note (section 9).

    python clustering/heldout_benchmark.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR

JOBS = [
    ("pcs-ar1-ho", "pcs_lifecycle_20_89_minobs3_v1", "sf12pcs_dv"),
    ("physgrm-ar1-ho", "physgrm_lifecycle_20_89_minobs3_v1", "theta_phys_full"),
    ("combgrm-ar1-ho", "combgrm_lifecycle_20_89_minobs3_v1", "theta_combined"),
]
HOLD_K, MIN_OBS = 2, 5


def main() -> int:
    rows = []
    for tag, contract, channel in JOBS:
        long = pd.read_csv(
            PROCESSED_DATA_DIR / "contracts" / contract / "long.csv"
        ).sort_values(["pidp", "age"])
        n = long.groupby("pidp")["age"].transform("size")
        keep = long[n >= MIN_OBS].copy()
        keep["z"] = ((keep[channel] - keep[channel].mean())
                     / keep[channel].std(ddof=1))
        rank = keep.groupby("pidp").cumcount(ascending=False)
        held, fitted = keep[rank < HOLD_K], keep[rank >= HOLD_K]

        z = held["z"].to_numpy()
        null_marginal = float((-0.5 * np.log(2 * np.pi) - 0.5 * z**2).mean())

        a = (fitted["age"].to_numpy() - 55) / 10
        X = np.column_stack([np.ones_like(a), a, a**2])
        beta, *_ = np.linalg.lstsq(X, fitted["z"].to_numpy(), rcond=None)
        sd = (fitted["z"].to_numpy() - X @ beta).std(ddof=3)
        ah = (held["age"].to_numpy() - 55) / 10
        Xh = np.column_stack([np.ones_like(ah), ah, ah**2])
        null_age = float((-0.5 * np.log(2 * np.pi) - np.log(sd)
                          - 0.5 * ((z - Xh @ beta) / sd) ** 2).mean())

        summary = json.loads(
            (ARTIFACTS_DIR / "overnight" / tag / "run_summary.json").read_text())
        model = summary["heldout_mean_lpd"] / HOLD_K
        rows.append({"fit": tag, "channel": channel, "held_rows": len(held),
                     "model_lpd_per_obs": model, "null_marginal": null_marginal,
                     "null_age_quadratic": null_age,
                     "gain_over_age_null": model - null_age})
    tab = pd.DataFrame(rows)
    out = ARTIFACTS_DIR / "descriptives" / "heldout_benchmark.csv"
    tab.to_csv(out, index=False)
    print(tab.round(4).to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
