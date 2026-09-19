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


TIMED_CONDITION_ITEMS: dict[str, int] = {}
for _g, _k in PHYS_CONDITION_ITEMS.items():
    TIMED_CONDITION_ITEMS[f"{_g}R"] = _k     # diagnosed within the last 10 years
    TIMED_CONDITION_ITEMS[f"{_g}S"] = _k     # diagnosed more than 10 years ago
PHYS_TIMED_NCAT: dict[str, int] = {**{c: PHYS_NCAT[c] for c in PHYS_CORE},
                                   **TIMED_CONDITION_ITEMS}


def build_physical_timed_items(
    items: pd.DataFrame,
    chronic: pd.DataFrame,
    *,
    age_range: tuple[int, int] = (20, 90),
) -> pd.DataFrame:
    """P-TIMED: each condition group split into recent (<= 10y) and stale
    (> 10y) items with separate discriminations — the model then ESTIMATES the
    recency weighting instead of assuming it (the in-framework analogue of a
    weighted-cumulative-exposure weight function).
    """
    core = build_testlets(items, age_range=age_range)
    func = pd.concat([items[["pidp", "wave"]], build_func_item(items)], axis=1)
    out = core.merge(func, on=["pidp", "wave"], how="left")
    cols = ["pidp", "wave"] + [f"n_{g.lower()}" for g in PHYS_CONDITION_ITEMS]         + [f"nrec10_{g.lower()}" for g in PHYS_CONDITION_ITEMS]
    out = out.merge(chronic[cols], on=["pidp", "wave"], how="left")
    names = list(PHYS_CORE)
    for g, k in PHYS_CONDITION_ITEMS.items():
        recent = out[f"nrec10_{g.lower()}"]
        stale = out[f"n_{g.lower()}"] - recent
        out[f"{g}R"] = _reverse_count(recent, k)
        out[f"{g}S"] = _reverse_count(stale, k)
        names += [f"{g}R", f"{g}S"]
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

COMBINED_NCAT: dict[str, int] = {**PHYS_NCAT, **MENT_TESTLET_NCAT}


def build_combined_items(phys_full: pd.DataFrame, ment_testlet: pd.DataFrame) -> pd.DataFrame:
    """One bank over every good-coverage item: the P-FULL physical bank plus
    the mental testlet bank (GHQ pre-collapsed by wording, as decided by the
    two-bank Q3 diagnostics), inner-joined on person-wave."""
    m = ment_testlet.drop(columns=["age"])
    out = phys_full.merge(m, on=["pidp", "wave"], how="inner")
    return out.reset_index(drop=True)



# ---------------------------------------------------------------------------
# limitation-count banks: the four SF-12 testlets plus counts over disdif areas
# ---------------------------------------------------------------------------

DISDIF_LISTED: tuple[str, ...] = tuple(f"disdif{i}" for i in range(1, 13)) + ("disdif96",)

# name -> disdif areas counted
LIMITATION_GROUPS: dict[str, tuple[int, ...]] = {
    "LIM_PF": (1, 2, 3, 10, 11),   # mobility, lifting, dexterity, coordination, personal care
    "LIM_SC": (4, 5, 6),           # continence, hearing, sight
    "LIM_MOB": (1, 2),             # mobility, lifting
    "LIM_DEX": (3, 10, 11),        # dexterity, coordination, personal care
    "LIM_CONT": (4,),              # continence
    "LIM_SENS": (5, 6),            # hearing, sight
    "LIM_FL": (1, 2, 3, 10),       # functional limitations
    "LIM_SELF": (4, 11),           # self-care: continence, personal care
    "LIM_OTHER": (12,),            # other health problem or disability
}

# bank -> limitation testlets added to GH, PF, RP, BP
LIMITATION_BANKS: dict[str, tuple[str, ...]] = {
    "P-FUNC": ("FUNC",),
    "P-LIM1": ("LIM_PF",),
    "P-LIM": ("LIM_PF", "LIM_SC"),
    "P-LIM4": ("LIM_MOB", "LIM_DEX", "LIM_CONT", "LIM_SENS"),
    "P-LIM3": ("LIM_FL", "LIM_SELF", "LIM_SENS"),
    "P-LIM3+O": ("LIM_FL", "LIM_SELF", "LIM_SENS", "LIM_OTHER"),
}

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


def build_limitation_items(
    items: pd.DataFrame, age_range: tuple[int, int] = (20, 90)
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Every testlet the limitation banks use, on their common sample, 1 = worst.

    The sample is P-FUNC's: the four SF-12 testlets complete and the disdif
    module usable. Every limitation group shares that missingness, so all six
    banks are fitted on identical person-waves. Returns the frame and the
    number of categories per testlet after the merge rule.
    """
    core = build_testlets(items, age_range=age_range)
    counts = pd.DataFrame({g: limitation_count(items, c) for g, c in LIMITATION_GROUPS.items()})
    counts["FUNC"] = build_func_item(items)
    counts[["pidp", "wave"]] = items[["pidp", "wave"]]
    out = core.merge(counts, on=["pidp", "wave"], how="left")
    out = out.dropna(subset=["FUNC", *LIMITATION_GROUPS])
    ncat = {"GH": 5, "PF": 5, "RP": 9, "BP": 5, "FUNC": 4}
    for g in LIMITATION_GROUPS:
        cap = top_category(out[g])
        out[g] = _reverse_count(out[g], cap + 1)
        ncat[g] = cap + 1
    for c in ncat:
        out[c] = out[c].astype(int)
    return out.reset_index(drop=True), ncat


# bank suffix -> condition testlets: CC the total ever-diagnosed count of the
# sixteen physical conditions, CG P-FULL's six group items
CONDITION_CODINGS: dict[str, tuple[str, ...]] = {
    "CC": ("COND",),
    "CG": tuple(PHYS_CONDITION_ITEMS),
}


def add_condition_items(
    bank_items: pd.DataFrame, chronic: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """The condition testlets alongside the limitation testlets, 1 = worst.

    COND sums the six ever-diagnosed group counts, which is the count over the
    sixteen physical conditions, with the top category set by the merge rule on
    the rows that have a condition inventory; the group items are coded exactly
    as in P-FULL. Rows without an inventory keep NaN condition items.
    """
    groups = [f"n_{g.lower()}" for g in PHYS_CONDITION_ITEMS]
    out = bank_items.merge(chronic[["pidp", "wave", *groups]], on=["pidp", "wave"], how="left")
    total = out[groups].sum(axis=1, min_count=len(groups)).where(out[groups].notna().all(axis=1))
    cap = top_category(total)
    out["COND"] = _reverse_count(total, cap + 1)
    ncat = {"COND": cap + 1}
    for g, k in PHYS_CONDITION_ITEMS.items():
        out[g] = _reverse_count(out[f"n_{g.lower()}"], k)
        ncat[g] = k
    return out.drop(columns=groups), ncat


__all__ = [
    "COMBINED_NCAT",
    "CONDITION_CODINGS",
    "DISDIF_LISTED",
    "LIMITATION_BANKS",
    "LIMITATION_GROUPS",
    "MIN_TOP_SHARE",
    "GHQ_NEGATIVE",
    "GHQ_POSITIVE",
    "MENT_ITEMS",
    "MENT_NCAT",
    "MENT_TESTLET_NCAT",
    "PHYS_CONDITION_ITEMS",
    "PHYS_CORE",
    "PHYS_DISDIF",
    "PHYS_NCAT",
    "PHYS_TIMED_NCAT",
    "TIMED_CONDITION_ITEMS",
    "add_condition_items",
    "build_combined_items",
    "build_func_item",
    "build_limitation_items",
    "limitation_count",
    "top_category",
    "build_physical_timed_items",
    "build_mental_items",
    "build_physical_items",
    "collapse_ghq_wording",
]
