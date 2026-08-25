"""Licence-free single-index composites over the eight SF-12 subscales.

* Farivar et al. oblique composites (PCS_c / MCS_c): correlated summaries that
  avoid the varimax sign flips — no negative mental weights in the physical
  score. Applied to UK-normed subscale z-scores, T-anchored in the pooled
  sample, as in the sandbox (11_sf6d_and_composites.py).
* Equal-weight composite: the plain mean of the eight 0-100 subscales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.sf12 import SCALES

# Farivar, Cunningham & Hays (2007) correlated (oblique) scoring coefficients.
FARIVAR: dict[str, dict[str, float]] = {
    "PCS_c": {"PF": 0.20, "RP": 0.31, "BP": 0.23, "GH": 0.20,
              "VT": 0.13, "SF": 0.11, "RE": 0.03, "MH": -0.03},
    "MCS_c": {"PF": -0.02, "RP": 0.03, "BP": 0.04, "GH": 0.10,
              "VT": 0.29, "SF": 0.14, "RE": 0.20, "MH": 0.35},
}


def farivar_composites(
    sub: pd.DataFrame, norms: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """PCS_c/MCS_c on UK-normed z-scores, anchored 50/10 in the pooled sample."""
    z = np.column_stack(
        [(sub[s].to_numpy(float) - norms[s][0]) / norms[s][1] for s in SCALES]
    )
    out = pd.DataFrame(index=sub.index)
    for name, coef in FARIVAR.items():
        c = np.array([coef[s] for s in SCALES])
        raw = z @ c
        out[f"{name}_farivar"] = 50.0 + 10.0 * (raw - np.nanmean(raw)) / np.nanstd(raw)
    return out


def equal_weight_composite(sub: pd.DataFrame) -> pd.Series:
    """Mean of the eight 0-100 subscales; NaN unless all eight are present."""
    values = sub[list(SCALES)].to_numpy(float)
    complete = ~np.isnan(values).any(axis=1)
    out = np.full(len(sub), np.nan)
    out[complete] = values[complete].mean(axis=1)
    return pd.Series(out, index=sub.index, name="HEALTH_equal8")


__all__ = ["FARIVAR", "equal_weight_composite", "farivar_composites"]
