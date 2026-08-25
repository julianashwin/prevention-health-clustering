"""Item banks for the two-dimensional GRM: physical and mental health.

Assembles the every-wave item sets proposed in docs/health_measures_note.md.
Every item is coded so category 1 is WORST health, matching the original
physical GRM. Assignments of the cross-loading SF-12 items follow this repo's
own UK wave-1 varimax loadings (measures build): PF .86/.14, RP .85/.25,
BP .75/.17, GH .72/.27 physical; MH .09/.89, RE .26/.82, SF .48/.64 mental.
VT (sf6b, .55/.44) is genuinely bidimensional and enters neither bank.

PHYSICAL (P-FUNC adds FUNC to the validated four testlets; P-FULL adds the
chronic-condition groups; the recency variant dates conditions and drops
diagnoses more than ten years old):

    GH PF RP BP        the original testlets (grm.build_testlets)
    FUNC               Equality Act physical limitations {mobility, lifting,
                       dexterity, coordination, personal care}: 0/1/2/3+,
                       structural zero when health == 2 (no long-standing
                       illness -> the module is skipped by design)
    CVD METAB RESP     ever-diagnosis group counts (measures/chronic.py),
    MSK CANCER OTHER   capped: 2+/2/2+ then binary for the last three

MENTAL:

    MH                 (6 - sf6a) + sf6c - 1, 9 levels (calm + downhearted)
    RE                 sf4a + sf4b - 1, 9 levels (emotional role)
    SF                 sf7, 5 levels (social functioning; weakest assignment,
                       flagged for the drop-if-DIF check)
    GHQ a..l           12 items, 4 levels, reversed to 1 = worst
    DEPR               ever clinical depression (hcond17), binary
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.grm import build_testlets
from prevention_health_clustering.measures.items import GHQ_ITEM_STEMS

PHYS_DISDIF = (1, 2, 3, 10, 11)   # mobility, lifting, dexterity, coordination, care

# name -> (source kind, categories)
PHYS_CONDITION_ITEMS: dict[str, int] = {
    "CVD": 3, "METAB": 3, "RESP": 3, "MSK": 2, "CANCER": 2, "OTHER": 2,
}
PHYS_CORE: tuple[str, ...] = ("GH", "PF", "RP", "BP", "FUNC")
PHYS_NCAT: dict[str, int] = {"GH": 5, "PF": 5, "RP": 9, "BP": 5, "FUNC": 4,
                             **PHYS_CONDITION_ITEMS}

MENT_ITEMS: tuple[str, ...] = ("MH", "RE", "SF") + tuple(
    s.upper() for s in GHQ_ITEM_STEMS) + ("DEPR",)
MENT_NCAT: dict[str, int] = {"MH": 9, "RE": 9, "SF": 5, "DEPR": 2,
                             **{s.upper(): 4 for s in GHQ_ITEM_STEMS}}


def _reverse_count(count: pd.Series, k: int) -> pd.Series:
    """Cap a 0-based deficit count at k-1 and reverse so 1 = worst."""
    capped = count.clip(upper=k - 1)
    return (k - capped).where(count.notna())


def build_func_item(items: pd.DataFrame) -> pd.Series:
    """The disdif physical-limitation testlet, 1 = worst (3+ limitations).

    Routing: disdif is asked only when health == 1 (long-standing illness).
    health == 2 respondents are valid zeros by design. A health == 1 row where
    no disdif item was answered is missing, not zero.
    """
    disdif = items[[f"disdif{i}" for i in PHYS_DISDIF]]
    answered = items[[f"disdif{i}" for i in range(1, 13)] + ["disdif96"]]
    asked = answered.notna().any(axis=1)
    count = disdif.eq(1).sum(axis=1).astype(float)
    count = count.where((items["health"] == 1) & asked)
    count = count.mask(items["health"] == 2, 0.0)
    return _reverse_count(count, 4).rename("FUNC")


def build_physical_items(
    items: pd.DataFrame,
    chronic: pd.DataFrame | None = None,
    *,
    conditions: str | None = "ever",
    age_range: tuple[int, int] = (20, 90),
) -> pd.DataFrame:
    """Person-wave physical item bank; complete cases on the requested items.

    ``conditions``: "ever" uses n_{group}, "recent10" uses nrec10_{group},
    None builds the functioning-only bank (P-FUNC).
    """
    core = build_testlets(items, age_range=age_range)
    func = pd.concat(
        [items[["pidp", "wave"]], build_func_item(items)], axis=1)
    out = core.merge(func, on=["pidp", "wave"], how="left")
    names = list(PHYS_CORE)
    if conditions is not None:
        prefix = {"ever": "n_", "recent10": "nrec10_"}[conditions]
        cols = ["pidp", "wave"] + [f"{prefix}{g.lower()}"
                                   for g in PHYS_CONDITION_ITEMS]
        merged = out.merge(chronic[cols], on=["pidp", "wave"], how="left")
        for g, k in PHYS_CONDITION_ITEMS.items():
            merged[g] = _reverse_count(merged[f"{prefix}{g.lower()}"], k)
        out = merged
        names += list(PHYS_CONDITION_ITEMS)
    out = out[["pidp", "wave", "age"] + names].dropna(subset=names)
    for n in names:
        out[n] = out[n].astype(int)
    return out.reset_index(drop=True)


def build_mental_items(
    items: pd.DataFrame,
    chronic: pd.DataFrame,
    *,
    age_range: tuple[int, int] = (20, 90),
) -> pd.DataFrame:
    """Person-wave mental item bank, complete cases, 1 = worst."""
    d = items.copy()
    d = d[d["age"].notna() & d["age"].between(*age_range)]
    for v, k in (("sf4a", 5), ("sf4b", 5), ("sf6a", 5), ("sf6c", 5), ("sf7", 5)):
        d[v] = d[v].where(d[v].isin(range(1, k + 1)))
    out = pd.DataFrame({
        "pidp": d["pidp"], "wave": d["wave"], "age": d["age"].astype(int),
        "MH": (6 - d["sf6a"]) + d["sf6c"] - 1,
        "RE": d["sf4a"] + d["sf4b"] - 1,
        "SF": d["sf7"],
    })
    for stem in GHQ_ITEM_STEMS:
        raw = d[stem].where(d[stem].isin([1, 2, 3, 4]))
        out[stem.upper()] = 5 - raw          # reverse: 1 = worst
    depr = chronic[["pidp", "wave", "ever_17"]]
    out = out.merge(depr, on=["pidp", "wave"], how="left")
    out["DEPR"] = np.where(out["ever_17"].isna(), np.nan, 2 - out["ever_17"])
    names = list(MENT_ITEMS)
    out = out[["pidp", "wave", "age"] + names].dropna(subset=names)
    for n in names:
        out[n] = out[n].astype(int)
    return out.reset_index(drop=True)


GHQ_POSITIVE = ("SCGHQA", "SCGHQC", "SCGHQD", "SCGHQG", "SCGHQH", "SCGHQL")
GHQ_NEGATIVE = ("SCGHQB", "SCGHQE", "SCGHQF", "SCGHQI", "SCGHQJ", "SCGHQK")


def collapse_ghq_wording(ment: pd.DataFrame) -> pd.DataFrame:
    """Sum the GHQ items into two wording testlets (each 6..24 -> 1..19).

    The pre-specified remedy if item-level GHQ shows positive within-wording
    residual dependence (method effects; Hankins 2008) — the same cure the
    original GRM applied to the PF and RP pairs.
    """
    out = ment[["pidp", "wave", "age", "MH", "RE", "SF", "DEPR"]].copy()
    out["GHQPOS"] = ment[list(GHQ_POSITIVE)].sum(axis=1) - 5
    out["GHQNEG"] = ment[list(GHQ_NEGATIVE)].sum(axis=1) - 5
    return out


MENT_TESTLET_NCAT: dict[str, int] = {"MH": 9, "RE": 9, "SF": 5,
                                     "GHQPOS": 19, "GHQNEG": 19, "DEPR": 2}

__all__ = [
    "GHQ_NEGATIVE",
    "GHQ_POSITIVE",
    "MENT_ITEMS",
    "MENT_NCAT",
    "MENT_TESTLET_NCAT",
    "PHYS_CONDITION_ITEMS",
    "PHYS_CORE",
    "PHYS_DISDIF",
    "PHYS_NCAT",
    "build_func_item",
    "build_mental_items",
    "build_physical_items",
    "collapse_ghq_wording",
]
