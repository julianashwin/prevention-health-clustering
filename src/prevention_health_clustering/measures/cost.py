"""A secondary- and primary-care cost index from UKHLS utilisation.

Not a cost: a bed-day-weighted index in pounds, built from the physical
quantities UKHLS records. It omits prescribing, A&E, community services and
social care entirely, and every input is self-reported over a twelve-month
recall window. Named accordingly wherever it is reported.

Three tiers, all banded or counted per person-wave over waves 7-15:

    hl2gp    GP visits           band 0-4
    hl2hop   out-patient         band 0-4
    hosp     any in-patient stay, with hospd nights and hospch a maternity flag

UNIT COSTS are national averages from the NHS National Cost Collection,
which publishes free annual figures at aggregate and HRG level. The values
below are indicative and are deliberately kept in one editable table: they
should be replaced with the exact figures for a chosen price year before any
result is published, and every function takes the table as an argument so
that swapping it is a one-line change.

THE SPELL PROBLEM. UKHLS records total nights but not the number of
admissions, and twenty nights as one spell or four differ in cost by about
threefold. ``spell_rule`` selects an assumption and exists so that the
sensitivity can be reported rather than hidden: "one" treats the year as a
single spell, "per_los" assumes one spell per ``included_los`` nights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Band midpoints for "none / one or two / three to five / six to ten /
# more than ten". The top band is open-ended; 15 is a deliberate,
# conservative choice and is varied in the sensitivity analysis.
BAND_MIDPOINTS = np.array([0.0, 1.5, 4.0, 8.0, 15.0])


@dataclass
class UnitCosts:
    """National average unit costs, in pounds of a single price year."""

    spell: float = 1590.0          # non-elective in-patient episode
    excess_bed_day: float = 313.0  # each night beyond the included stay
    outpatient: float = 120.0      # one out-patient attendance
    gp: float = 42.0               # one GP consultation
    maternity_spell: float = 2000.0
    included_los: float = 5.0      # nights already inside the spell price
    price_year: str = "indicative"
    # multipliers on the spell price by condition group; 1.0 is the average
    # non-elective admission. Indicative, and varied in sensitivity analysis.
    condition_weight: dict[str, float] = field(default_factory=lambda: {
        "cvd": 1.35, "resp": 1.10, "cancer": 1.45, "msk": 1.25,
        "metab": 0.95, "neuro": 1.50, "renal": 1.40, "mental": 0.85,
        "other": 1.00,
    })


# UKHLS condition code -> cost group. Codes follow the hcond numbering, which
# the later waves extend well beyond the seventeen of the measurement note.
CONDITION_GROUP: dict[int, str] = {
    1: "resp", 8: "resp", 11: "resp", 21: "resp", 78: "resp",
    3: "cvd", 4: "cvd", 5: "cvd", 6: "cvd", 16: "cvd",
    7: "neuro", 15: "neuro", 19: "neuro", 70: "neuro", 81: "neuro",
    82: "neuro", 83: "neuro", 84: "neuro",
    13: "cancer", 26: "cancer", 27: "cancer", 28: "cancer", 29: "cancer",
    30: "cancer", 31: "cancer", 79: "cancer",
    2: "msk", 23: "msk", 24: "msk",
    14: "metab", 9: "metab", 10: "metab", 33: "metab", 34: "metab", 35: "metab",
    12: "other", 80: "renal", 86: "other", 87: "other",
    38: "mental", 39: "mental", 40: "mental", 41: "mental", 42: "mental",
    66: "mental", 67: "mental", 68: "mental", 69: "mental", 71: "mental",
    72: "mental", 73: "mental", 74: "mental", 75: "mental", 89: "mental",
}


def band_to_count(band: pd.Series, midpoints: np.ndarray = BAND_MIDPOINTS,
                  top: float | None = None) -> pd.Series:
    """Map a 0-4 frequency band to an implied count."""
    mp = midpoints.copy()
    if top is not None:
        mp[-1] = top
    idx = band.round().astype("Int64")
    out = pd.Series(np.nan, index=band.index, dtype=float)
    valid = idx.notna() & idx.between(0, len(mp) - 1)
    out[valid] = mp[idx[valid].astype(int)]
    return out


def n_spells(nights: np.ndarray, admitted: np.ndarray, rule: str,
             included_los: float) -> np.ndarray:
    """Implied number of in-patient spells under an explicit assumption."""
    if rule == "one":
        return admitted.astype(float)
    if rule == "per_los":
        return np.where(admitted == 1,
                        np.maximum(1.0, np.ceil(nights / included_los)), 0.0)
    if rule == "two":
        return np.where((admitted == 1) & (nights > included_los), 2.0,
                        admitted.astype(float))
    raise ValueError(f"unknown spell rule {rule!r}")


def inpatient_cost(frame: pd.DataFrame, costs: UnitCosts, *,
                   spell_rule: str = "one",
                   weights: pd.Series | None = None,
                   exclude_maternity: bool = True) -> pd.Series:
    """Spell price (optionally condition-weighted) plus excess bed days."""
    admitted = (frame["hosp"] == 1).to_numpy()
    nights = frame["hospd"].fillna(0).to_numpy(float)
    birth = (frame.get("hospch", pd.Series(np.nan, index=frame.index)) == 1).to_numpy()
    if exclude_maternity:
        admitted = admitted & ~birth
        nights = np.where(birth, 0.0, nights)
    spells = n_spells(nights, admitted, spell_rule, costs.included_los)
    w = np.ones(len(frame)) if weights is None else weights.to_numpy(float)
    base = spells * costs.spell * w
    excess = np.maximum(nights - spells * costs.included_los, 0.0)
    return pd.Series(base + excess * costs.excess_bed_day, index=frame.index)


def build_cost_index(frame: pd.DataFrame, costs: UnitCosts | None = None, *,
                     spell_rule: str = "one",
                     condition_weights: pd.Series | None = None,
                     exclude_maternity: bool = True,
                     top_band: float | None = None) -> pd.DataFrame:
    """Per person-wave cost index and its three components."""
    costs = costs or UnitCosts()
    out = pd.DataFrame(index=frame.index)
    out["cost_inpatient"] = inpatient_cost(
        frame, costs, spell_rule=spell_rule, weights=condition_weights,
        exclude_maternity=exclude_maternity)
    out["n_outpatient"] = band_to_count(frame["hl2hop"], top=top_band)
    out["n_gp"] = band_to_count(frame["hl2gp"], top=top_band)
    out["cost_outpatient"] = out["n_outpatient"] * costs.outpatient
    out["cost_gp"] = out["n_gp"] * costs.gp
    out["cost_total"] = (out["cost_inpatient"].fillna(0)
                         + out["cost_outpatient"].fillna(0)
                         + out["cost_gp"].fillna(0))
    # a row is usable only if the in-patient question was answered
    out.loc[frame["hosp"].isna(), ["cost_inpatient", "cost_total"]] = np.nan
    return out


__all__ = ["BAND_MIDPOINTS", "CONDITION_GROUP", "UnitCosts", "band_to_count",
           "build_cost_index", "inpatient_cost", "n_spells"]
