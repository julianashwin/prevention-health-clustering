"""SF-6D (SF-12) health-state classification and UK utility tariff.

Brazier & Roberts (2004), Medical Care 42(9):851-859: the UK standard-gamble
value set for the SF-12-derived SF-6D. Coefficients transcribed from Table 5-1
of Leelahavarong (2018, University of Glasgow PhD thesis), which reproduces the
published model with a worked example; ``validate_tariff`` checks that example
(0.657), the ceiling (1.000) and the floor (0.345) and is run by the build
driver every time.

Version caveat (documented in the construction note): the classification was
defined on SF-12 v1, UKHLS administers v2. The v2 5-point role items are
dichotomised at "none of the time"; the 5-level SF/MH/VIT alignments are
assumed 1:1. Research-grade utilities — obtain the IQVIA/Sheffield licence
before publishing utility values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIMS: tuple[str, ...] = ("PF", "RL", "SF", "PAIN", "MH", "VIT")
LEVELS: dict[str, int] = {"PF": 3, "RL": 4, "SF": 5, "PAIN": 5, "MH": 5, "VIT": 5}

# Decrements from a constant of 1.000.
TARIFF: dict[str, dict[int, float]] = {
    "PF":   {1: 0.000, 2: 0.000, 3: 0.045},
    "RL":   {1: 0.000, 2: 0.063, 3: 0.063, 4: 0.063},
    "SF":   {1: 0.000, 2: 0.063, 3: 0.066, 4: 0.081, 5: 0.093},
    "PAIN": {1: 0.000, 2: 0.000, 3: 0.042, 4: 0.077, 5: 0.137},
    "MH":   {1: 0.000, 2: 0.059, 3: 0.059, 4: 0.113, 5: 0.134},
    "VIT":  {1: 0.000, 2: 0.078, 3: 0.078, 4: 0.078, 5: 0.106},
}
CONSTANT = 1.000
MOST_SEVERE = 0.077
# Levels starred "most severe" in the source table. Not simply each dimension's
# top level: SF, PAIN and MH star their top TWO levels; RL stars 3 and 4.
MOST_LEVELS: dict[str, set[int]] = {
    "PF": {3}, "RL": {3, 4}, "SF": {4, 5},
    "PAIN": {4, 5}, "MH": {4, 5}, "VIT": {5},
}


def classify(items: pd.DataFrame) -> pd.DataFrame:
    """Map SF-12v2 items onto the six SF-6D dimensions (1 = best).

    Framing asymmetry, faithful to the source: sf6b (vitality) is POSITIVELY
    worded so its level maps directly, while negatively-worded items reverse.
    """
    d = items
    out = pd.DataFrame(index=d.index)
    out["PF"] = 4 - d.sf2a                    # 1 limited a lot .. 3 not limited
    phys = (d.sf3a < 5).astype(float)         # v2 role items dichotomised at
    emot = (d.sf4a < 5).astype(float)         # "none of the time"
    out["RL"] = 1 + phys + 2 * emot           # 1 none, 2 phys, 3 emot, 4 both
    out.loc[(phys == 1) & (emot == 1), "RL"] = 4
    out["SF"] = 6 - d.sf7                     # 1 all .. 5 none -> reverse
    out["PAIN"] = d.sf5                       # already ascending severity
    out["MH"] = 6 - d.sf6c                    # downhearted: reverse
    out["VIT"] = d.sf6b                       # positively worded: direct
    return out


def utility(states: pd.DataFrame) -> pd.Series:
    """Apply the Brazier & Roberts tariff to a frame of dimension levels."""
    u = pd.Series(CONSTANT, index=states.index, dtype=float)
    at_most = pd.Series(False, index=states.index)
    for dim in DIMS:
        lv = states[dim]
        u -= lv.map(TARIFF[dim]).astype(float)
        at_most |= lv.isin(MOST_LEVELS[dim])
    u -= at_most.astype(float) * MOST_SEVERE
    return u.where(states[list(DIMS)].notna().all(axis=1))


def state_string(states: pd.DataFrame, complete: pd.Series) -> pd.Series:
    """Six-digit state label, e.g. '111111', where classification is complete."""
    usable = complete & states[list(DIMS)].notna().all(axis=1)
    rows = states.loc[usable, list(DIMS)].astype(int).astype(str)
    label = rows[DIMS[0]].str.cat([rows[d] for d in DIMS[1:]])
    return label.reindex(states.index)


def validate_tariff() -> list[tuple[str, bool, float]]:
    """The published worked example, the ceiling, and the floor."""
    worked = pd.DataFrame([{"PF": 2, "RL": 4, "SF": 3, "PAIN": 2, "MH": 3, "VIT": 4}])
    best = pd.DataFrame([{d: 1 for d in DIMS}])
    worst = pd.DataFrame([{d: LEVELS[d] for d in DIMS}])
    checks = []
    for name, frame, expected in (
        ("worked example PF2 RL4 SF3 PAIN2 MH3 VIT4", worked, 0.657),
        ("ceiling (all best)", best, 1.000),
        ("floor (all worst)", worst, 0.345),
    ):
        got = float(utility(frame).iloc[0])
        checks.append((name, bool(abs(got - expected) < 5e-4), got))
    return checks


__all__ = [
    "CONSTANT",
    "DIMS",
    "LEVELS",
    "MOST_LEVELS",
    "MOST_SEVERE",
    "TARIFF",
    "classify",
    "state_string",
    "utility",
    "validate_tariff",
]
