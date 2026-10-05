"""Item banks for the graded-response models beyond the paper's health measure.

The MENTAL bank, the multidimensional model's second channel and the mental
GRM of the mental-health appendix (data_cleaning/04c_build_mental_grm.py).
Every item is coded so category 1 is WORST health. Assignments of the
cross-loading SF-12 items follow the UK wave-1 varimax loadings: MH .09/.89,
RE .26/.82, SF .48/.64 mental; VT (sf6b, .55/.44) is genuinely bidimensional
and enters neither the physical nor the mental bank.

    MH                 (6 - sf6a) + sf6c - 1, 9 levels (calm + downhearted)
    RE                 sf4a + sf4b - 1, 9 levels (emotional role)
    SF                 sf7, 5 levels (social functioning)
    GHQ a..l           12 items, 4 levels, reversed to 1 = worst; collapsed
                       into two wording testlets when Yen's Q3 shows
                       within-wording dependence (collapse_ghq_wording)
    DEPR               ever clinical depression (hcond17), binary; the paper's
                       bank drops it (include_depr=False)

Also the limitation-count helpers that the paper's health measure
(measures/health.py) builds its functioning testlets with: the disdif areas
mentioned, with the wave 1-7 routing applied in every wave, and the top-category
merge rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.items import GHQ_ITEM_STEMS

MENT_ITEMS: tuple[str, ...] = ("MH", "RE", "SF") + tuple(
    s.upper() for s in GHQ_ITEM_STEMS) + ("DEPR",)
MENT_NCAT: dict[str, int] = {"MH": 9, "RE": 9, "SF": 5, "DEPR": 2,
                             **{s.upper(): 4 for s in GHQ_ITEM_STEMS}}


def _reverse_count(count: pd.Series, k: int) -> pd.Series:
    """Cap a 0-based deficit count at k-1 and reverse so 1 = worst."""
    capped = count.clip(upper=k - 1)
    return (k - capped).where(count.notna())


def build_mental_items(
    items: pd.DataFrame,
    chronic: pd.DataFrame,
    *,
    age_range: tuple[int, int] = (20, 90),
    include_depr: bool = True,
) -> pd.DataFrame:
    """Person-wave mental item bank, complete cases, 1 = worst.

    ``include_depr`` False drops the ever-diagnosed-depression item BEFORE
    the complete-case rule is applied, which is the point of dropping it: the
    item carries the condition inventory's BHPS exclusion, so keeping it in
    the completeness rule costs 22% of person-waves for one weak item.
    """
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
    names = list(MENT_ITEMS)
    if include_depr:
        depr = chronic[["pidp", "wave", "ever_17"]]
        out = out.merge(depr, on=["pidp", "wave"], how="left")
        out["DEPR"] = np.where(out["ever_17"].isna(), np.nan,
                               2 - out["ever_17"])
    else:
        names = [n for n in names if n != "DEPR"]
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



# limitation-count helpers for the paper's health measure
# ---------------------------------------------------------------------------

DISDIF_LISTED: tuple[str, ...] = tuple(f"disdif{i}" for i in range(1, 13)) + ("disdif96",)

# a top count category holding less than this share of fit-sample
# person-waves merges into the category below, fixed before any fitting
MIN_TOP_SHARE = 0.005


def limitation_count(items: pd.DataFrame, codes: tuple[int, ...]) -> pd.Series:
    """Areas mentioned among ``codes``, with FUNC's routing in every wave.

    health == 2 counts as no limitation in every wave (the wave 1-7 routing
    applied throughout); health == 1 with no disdif answer is missing.
    """
    answered = items[list(DISDIF_LISTED)].notna().any(axis=1)
    count = items[[f"disdif{k}" for k in codes]].eq(1).sum(axis=1).astype(float)
    count = count.where((items["health"] == 1) & answered)
    return count.mask(items["health"] == 2, 0.0)


def top_category(count: pd.Series, min_share: float = MIN_TOP_SHARE) -> int:
    """Largest count kept as its own category under the merge rule."""
    share = count.dropna().value_counts(normalize=True).sort_index()
    cap = int(share.index.max())
    while cap > 1 and share[share.index >= cap].sum() < min_share:
        cap -= 1
    return cap



__all__ = [
    "DISDIF_LISTED",
    "GHQ_NEGATIVE",
    "GHQ_POSITIVE",
    "MENT_ITEMS",
    "MENT_NCAT",
    "MENT_TESTLET_NCAT",
    "MIN_TOP_SHARE",
    "build_mental_items",
    "collapse_ghq_wording",
    "limitation_count",
    "top_category",
]
