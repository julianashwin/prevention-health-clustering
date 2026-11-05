"""A secondary- and primary-care cost index from UKHLS utilisation.

Not a cost: a bed-day-weighted index in pounds, built from the physical
quantities UKHLS records. It omits prescribing, A&E, community services and
social care entirely, and every input is self-reported over a twelve-month
recall window. Named accordingly wherever it is reported.

Three tiers, all banded or counted per person-wave over waves 7-15:

    hl2gp    GP visits           band 0-4
    hl2hop   out-patient         band 0-4
    hosp     any in-patient stay, with hospd nights and hospch a maternity flag

UNIT COSTS are national averages in 2024/25 prices, kept in one editable
table that every function takes as an argument. The hospital rates come from
NHS England's National Cost Collection 2024/25 (published November 2025):
a stay of 0-1 nights is priced at the non-elective short-stay average per
episode, and a stay of 2+ nights at the non-elective long-stay cost per bed
day for every night. The 2024/25 collection no longer reports excess bed days
separately. The GP rate is PSSRU's Unit Costs of Health and Social Care 2025,
Table 9.4.2. None of the rates varies with age: NHS currencies separate
children from adults but have no older-age bands, and they rise with severity
(the HRG's complication and comorbidity score), which a flat index cannot see.

THE SPELL PROBLEM. UKHLS records total nights but not the number of
admissions. Under per-night pricing this matters only through ``stay_fixed``,
the fixed cost per stay in the two-part check (zero in the main index).
``spell_rule`` selects an assumption so that the sensitivity can be reported
rather than hidden: "one" treats the year as a single spell, "per_los" assumes
one spell per ``included_los`` nights.
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

    # NCC 2024/25, non-elective short stay (NES), average per episode
    short_stay: float = 826.0
    # NCC 2024/25 patient-level costs, non-elective long stay, per bed day
    # (written answer 100046, 22 Dec 2025); NEL averages £5,449 per episode
    bed_day: float = 791.0
    # fixed cost per stay of 2+ nights on top of its bed days; 0 in the main
    # index, positive only in the two-part check
    stay_fixed: float = 0.0
    short_stay_nights: float = 1.0  # stays up to this long take short_stay
    # NCC 2024/25, consultant-led and non-consultant-led attendances and
    # out-patient procedures, average per attendance
    outpatient: float = 191.0
    # PSSRU 2025 Table 9.4.2: surgery consultation of 10 minutes, including
    # direct care staff, excluding qualification costs (£48 with them)
    gp: float = 40.0
    included_los: float = 5.0       # nights per spell for the "per_los" and "two" rules
    price_year: str = "2024/25"
    # multipliers on the in-patient cost by condition group; 1.0 is the
    # average admission. Indicative, and varied in sensitivity analysis.
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
    """Short-stay price, or bed days plus any fixed cost per stay, times an
    optional condition weight."""
    admitted = (frame["hosp"] == 1).to_numpy()
    nights = frame["hospd"].fillna(0).to_numpy(float)
    birth = (frame.get("hospch", pd.Series(np.nan, index=frame.index)) == 1).to_numpy()
    if exclude_maternity:
        admitted = admitted & ~birth
        nights = np.where(birth, 0.0, nights)
    spells = n_spells(nights, admitted, spell_rule, costs.included_los)
    w = np.ones(len(frame)) if weights is None else weights.to_numpy(float)
    stay = np.where(nights <= costs.short_stay_nights, costs.short_stay,
                    spells * costs.stay_fixed + nights * costs.bed_day)
    return pd.Series(np.where(admitted, stay * w, 0.0), index=frame.index)


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
