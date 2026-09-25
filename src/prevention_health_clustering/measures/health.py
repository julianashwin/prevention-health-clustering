"""The project's health measure: its testlets, and the variants it is reported on.

The bank is the four validated SF-12 physical testlets, three counts over the
UKHLS impairment areas, and one count of ever-diagnosed physical conditions:

    GH        sf1, reversed                                    5 categories
    PF        sf2a + sf2b, moderate activities and stairs      5
    RP        sf3a + sf3b, accomplished less, limited in kind  9
    BP        sf5, pain interference, reversed                 5
    LIM_FL    functional limitations: mobility, lifting,
              dexterity, co-ordination                         0-4
    LIM_SELF  self-care: personal care, continence             0-2
    LIM_SENS  sensory: hearing, sight                          0-2
    COND      ever-diagnosed physical conditions                0-5, 6+

Every item is coded so category 1 is worst health. Two coding rules hold
throughout: someone without a long-standing illness counts as having no
limitation in every wave (the wave 1-7 routing applied to all waves), and a top
count category holding less than 0.5% of person-waves merges into the one
below, decided on the sample before any fitting.

The measure is reported three ways, all monotone in the same answers:
``theta`` from the graded-response model, ``h`` from its expected-score curve
on [0, 1], and ``fi10``, a ten-deficit index anyone can compute by hand. The
weighted sum that reproduces ``h`` is in ``projection_weights``.

The search that chose this bank is archived: see the measures note and
redo_concepts/baseline_measure_comparison.xlsx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.grm import NCAT, TESTLETS, build_testlets
from prevention_health_clustering.measures.grm2 import (
    DISDIF_LISTED,
    _reverse_count,
    limitation_count,
    top_category,
)

# testlet -> impairment areas counted (disdif codes)
HEALTH_GROUPS: dict[str, tuple[int, ...]] = {
    "LIM_FL": (1, 2, 3, 10),     # mobility, lifting, dexterity, co-ordination
    "LIM_SELF": (4, 11),         # continence, personal care
    "LIM_SENS": (5, 6),          # hearing, sight
}
CONDITION_GROUPS: tuple[str, ...] = ("cvd", "metab", "resp", "msk", "cancer", "other")
HEALTH_ITEMS: tuple[str, ...] = TESTLETS + tuple(HEALTH_GROUPS) + ("COND",)

# the six SF-12 items behind the testlets, graded 0 (none) to 1 (full deficit)
SF_DEFICITS: dict[str, tuple[str, int, bool]] = {
    "gh": ("sf1", 5, False), "pf_moderate": ("sf2a", 3, True), "pf_stairs": ("sf2b", 3, True),
    "rp_less": ("sf3a", 5, True), "rp_kind": ("sf3b", 5, True), "pain": ("sf5", 5, False),
}
DEFICIT_COND_CAP = 6


def build_health_items(
    items: pd.DataFrame, chronic: pd.DataFrame, age_range: tuple[int, int] = (20, 90)
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Person-waves with the eight testlets, complete cases, 1 = worst.

    Category caps follow the merge rule: the limitation counts are capped on the
    SF-12 sample and the condition count on the rows that also have a condition
    inventory, which is the sample the measure is fitted on.
    """
    core = build_testlets(items, age_range=age_range)
    counts = pd.concat([items[["pidp", "wave"]]]
                       + [limitation_count(items, c).rename(g) for g, c in HEALTH_GROUPS.items()], axis=1)
    out = core.merge(counts, on=["pidp", "wave"], how="left").dropna(subset=list(HEALTH_GROUPS))
    ncat = dict(NCAT)
    for g in HEALTH_GROUPS:                      # capped on the SF-12 sample, as in the search
        cap = top_category(out[g])
        out[g] = _reverse_count(out[g], cap + 1)
        ncat[g] = cap + 1
    ngroups = [f"n_{g}" for g in CONDITION_GROUPS]
    out = out.merge(chronic[["pidp", "wave", *ngroups]], on=["pidp", "wave"], how="left")
    total = out[ngroups].sum(axis=1).where(out[ngroups].notna().all(axis=1))
    out = out.assign(COND=total).dropna(subset=["COND"]).drop(columns=ngroups)
    cap = top_category(out["COND"])              # capped on the condition sample
    out["COND"] = _reverse_count(out["COND"], cap + 1)
    ncat["COND"] = cap + 1
    for c in HEALTH_ITEMS:
        out[c] = out[c].astype(int)
    return out[["pidp", "wave", "age", *HEALTH_ITEMS]].reset_index(drop=True), {
        c: ncat[c] for c in HEALTH_ITEMS}


def build_deficit_index(items: pd.DataFrame, chronic: pd.DataFrame,
                        age_range: tuple[int, int] = (20, 90)) -> pd.DataFrame:
    """The ten-deficit index, one minus the mean deficit so 1 is no deficits.

    Deficits: the six SF-12 physical items graded 0 to 1 by response step; the
    three impairment groups as the share of their areas mentioned; and the
    condition count capped at six and scaled to 0 to 1. Equal weights, which is
    only defensible once the areas are grouped: scored area by area the index
    stops being convex in healthcare cost (archived measures note).
    """
    d = items[items["age"].notna() & items["age"].between(*age_range)].copy()
    D = pd.DataFrame(index=d.index)
    for name, (col, k, reverse) in SF_DEFICITS.items():
        x = d[col].where(d[col].isin(range(1, k + 1)))
        D[name] = (k - x) / (k - 1) if reverse else (x - 1) / (k - 1)
    for g, codes in HEALTH_GROUPS.items():
        D[g] = pd.concat([limitation_count(d, (c,)) for c in codes], axis=1).mean(axis=1)
    ngroups = [f"n_{g}" for g in CONDITION_GROUPS]
    ch = d[["pidp", "wave"]].merge(chronic[["pidp", "wave", *ngroups]], on=["pidp", "wave"], how="left")
    total = ch[ngroups].sum(axis=1).where(ch[ngroups].notna().all(axis=1))
    D["conditions"] = (total.clip(upper=DEFICIT_COND_CAP) / DEFICIT_COND_CAP).to_numpy()
    fi = D.mean(axis=1).where(D.notna().all(axis=1))
    return pd.DataFrame({"pidp": d["pidp"].to_numpy(), "wave": d["wave"].to_numpy(),
                         "fi10": (1 - fi).to_numpy()})


def projection_weights(codes: pd.DataFrame, score: np.ndarray) -> tuple[pd.Series, float]:
    """Weights of the transparent twin: the score regressed on standardised codes.

    Returns weights per standard deviation, normalised to sum to one, and the
    R-squared, which says how close the measure is to a weighted sum.
    """
    X = codes[list(HEALTH_ITEMS)].to_numpy(float)
    Z = np.column_stack([np.ones(len(X)), (X - X.mean(axis=0)) / X.std(axis=0)])
    b, *_ = np.linalg.lstsq(Z, score, rcond=None)
    r2 = 1 - ((score - Z @ b) ** 2).sum() / ((score - score.mean()) ** 2).sum()
    return pd.Series(b[1:] / b[1:].sum(), index=list(HEALTH_ITEMS)), float(r2)


def one_factor_score(codes: pd.DataFrame, orient: np.ndarray | None = None
                     ) -> tuple[np.ndarray, pd.Series, float]:
    """The linear factor model on the eight codes, treated as continuous.

    One factor fitted to the correlation matrix of the standardised codes by
    iterated principal axes (the least-squares fit, equal here to maximum
    likelihood at the precision that matters), scored by regression
    (Thurstone) weights and standardised. This is what a linear measurement
    system of the skill-formation kind does with ordinal items: it keeps the
    loadings but imposes equal spacing between adjacent categories.

    Returns the score (higher = better health when ``orient`` is given and
    positively ordered), the loadings, and the first eigenvalue's share of
    the trace.
    """
    X = codes[list(HEALTH_ITEMS)].to_numpy(float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    R = np.corrcoef(Z, rowvar=False)
    psi = np.full(len(HEALTH_ITEMS), 0.5)
    for _ in range(1000):
        w, v = np.linalg.eigh(R - np.diag(psi))
        lam = v[:, -1] * np.sqrt(max(w[-1], 1e-12))
        new = np.clip(1 - lam**2, 1e-3, 1.0)
        if np.max(np.abs(new - psi)) < 1e-10:
            psi = new
            break
        psi = new
    fs = Z @ np.linalg.solve(R, lam)
    if orient is not None and np.corrcoef(fs, orient)[0, 1] < 0:
        fs, lam = -fs, -lam
    share = float(np.linalg.eigvalsh(R)[-1] / len(HEALTH_ITEMS))
    return (fs - fs.mean()) / fs.std(), pd.Series(lam, index=list(HEALTH_ITEMS)), share


__all__ = [
    "CONDITION_GROUPS",
    "HEALTH_GROUPS",
    "HEALTH_ITEMS",
    "build_deficit_index",
    "build_health_items",
    "one_factor_score",
    "projection_weights",
]
