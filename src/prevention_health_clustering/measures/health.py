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

The measure is reported two ways, both monotone in the same answers:
``theta`` from the graded-response model and ``h`` from its expected-score
curve on [0, 1]. The weighted sum that reproduces ``h`` is in
``projection_weights``; the 31-deficit frailty index and its shifted log, the
paper's robustness measures, are ``build_frailty_index``.

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
# one whole deficit, 1/31: the shift in log(frailty + c). Half the smallest positive value, 0.125/31, was tried
# first and put the zeros so far below the next value up that they drove the young-age variance.
FRAILTY_SHIFT = 1 / 31


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



def build_frailty_index(items: pd.DataFrame, chronic: pd.DataFrame,
                        age_range: tuple[int, int] = (20, 90)) -> pd.DataFrame:
    """A Rockwood-style frailty index: the share of deficits a person has, 0 = none.

    Every deficit counts once, with equal weight and no grouping, as in the
    frailty-index literature (Searle et al. 2008): the six SF-12 physical items
    graded 0 to 1 by response step, each of the eight physical impairment areas
    as a 0/1 deficit, and each ever-diagnosed condition in the chronic inventory
    as a 0/1 deficit (31 deficits on the current inventory). A person-wave is
    scored only when every deficit is observed. Higher is frailer, the opposite
    orientation of ``h`` and ``theta``.

    ``log_frailty`` is the log of the index shifted by ``FRAILTY_SHIFT``, one whole
    deficit (1/31), so the 7% of person-waves with no deficit stay in at the floor
    rather than being dropped. ``log_frailty_pos`` is the log on positive
    frailty only, as in Hosseini, Kopecky and Zhao (2022), who model the zeros
    with a separate probit; it is kept for reference.
    """
    d = items[items["age"].notna() & items["age"].between(*age_range)].copy()
    D = pd.DataFrame(index=d.index)
    for name, (col, k, reverse) in SF_DEFICITS.items():
        x = d[col].where(d[col].isin(range(1, k + 1)))
        D[f"sf_{name}"] = (k - x) / (k - 1) if reverse else (x - 1) / (k - 1)
    for codes in HEALTH_GROUPS.values():
        for c in codes:
            D[f"area_{c}"] = limitation_count(d, (c,)).to_numpy()
    ever = [c for c in chronic.columns if c.startswith("ever_")]
    ch = d[["pidp", "wave"]].merge(chronic[["pidp", "wave", *ever]], on=["pidp", "wave"], how="left")
    for c in ever:
        D[c] = ch[c].to_numpy()
    fr = D.mean(axis=1).where(D.notna().all(axis=1))
    out = pd.DataFrame({"pidp": d["pidp"].to_numpy(), "wave": d["wave"].to_numpy(),
                        "age": d["age"].to_numpy(), "frailty": fr.to_numpy()})
    out["log_frailty"] = np.log(out["frailty"] + FRAILTY_SHIFT)
    out["log_frailty_pos"] = np.log(out["frailty"].where(out["frailty"] > 0))
    out.attrs["n_deficits"] = D.shape[1]
    return out


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
    "build_frailty_index",
    "FRAILTY_SHIFT",
    "build_health_items",
    "one_factor_score",
    "projection_weights",
]
