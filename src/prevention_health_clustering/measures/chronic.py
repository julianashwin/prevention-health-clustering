"""Chronic-condition histories from the UKHLS diagnosis inventories.

UKHLS asks the full "have you ever been diagnosed" inventory (``hcond1..17``,
plus ``96`` none-of-these) at a person's FIRST interview; afterwards each wave
records conditions newly reported since the last interview, under a name that
changes with the questionnaire redesigns (all verified against the UKDA
dictionaries): ``hcondn{i}`` in waves 2-9, ``hcondcode{i}`` in waves 10-12,
``hcondncode{i}`` from wave 13. The running indicator is

    has ever had condition i by wave w  <=>  reported in ANY of these
    families (or ``hcondns{i}``, still-has-new) at any wave <= w

which is monotone by construction. Only the wave-1 condition list is used, so
the definition is identical in every wave (waves 10/14 added codes; counting
them would make totals jump mid-panel).

This corrects two defects in the EIT script's count: it used only
``hcondns*`` ("STILL HAS newly diagnosed condition") as the new-report family,
which (a) has no codes 6 and 7 at all — every in-panel heart attack and
stroke was missed — and (b) drops new diagnoses reported as already resolved.
Our count is a near-superset of the EIT count by construction.

"Current" status (``hconds##``) is deliberately NOT constructed: it is asked
only at the report wave (at wave 6, ~570 of 5,090 ever-asthma respondents were
routed to it), so a per-wave current variable would be ~90% carry-forward
imputation.

Diagnosis timing, for recency-restricted sensitivity variants:
  * inventory conditions carry ``hconda##`` = age told had the condition;
  * conditions first flagged by ``hcondn*`` in-panel are dated by the wave's
    own age (exact to the interview).

Grouping (used by the physical GRM as ordinal testlet items; per-condition
indicators are all retained on disk):
  CVD 3,4,5,6,7 | METAB 14,16 | RESP 1,8,11 | MSK 2 | CANCER 13
  OTHER 9,10,12,15 | DEPR 17 (mental side)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from prevention_health_clustering.config import UKHLS_PANEL_DIR

WAVES: tuple[str, ...] = tuple("abcdefghijklmno")

CONDITION_LABELS: dict[int, str] = {
    1: "asthma", 2: "arthritis", 3: "congestive heart failure",
    4: "coronary heart disease", 5: "angina", 6: "heart attack / MI",
    7: "stroke", 8: "emphysema", 9: "hyperthyroidism", 10: "hypothyroidism",
    11: "chronic bronchitis", 12: "liver condition", 13: "cancer",
    14: "diabetes", 15: "epilepsy", 16: "high blood pressure",
    17: "clinical depression",
}
CODES: tuple[int, ...] = tuple(CONDITION_LABELS)

CONDITION_GROUPS: dict[str, tuple[int, ...]] = {
    "cvd": (3, 4, 5, 6, 7),
    "metab": (14, 16),
    "resp": (1, 8, 11),
    "msk": (2,),
    "cancer": (13,),
    "other": (9, 10, 12, 15),
    "depr": (17,),
}


def _read_wave(raw_dir: Path, wave: str) -> pd.DataFrame:
    """One wave's inventory, new-diagnosis, age-told and age columns."""
    path = raw_dir / f"{wave}_indresp.tab"
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    present = set(header)
    cols: dict[str, str] = {}
    for i in CODES:
        if f"{wave}_hcond{i}" in present:
            cols[f"{wave}_hcond{i}"] = f"inv{i}"
        # new-report family, renamed at the wave-10 and wave-13 redesigns
        for k, stem in enumerate(
            (f"hcondn{i}", f"hcondcode{i}", f"hcondncode{i}", f"hcondns{i}")
        ):
            if f"{wave}_{stem}" in present:
                cols[f"{wave}_{stem}"] = f"new{i}_{k}"
        # age first told: hconda## at inventory waves, hcondna{i} for new
        # reports after the wave-10 redesign
        for k, stem in enumerate((f"hconda{i:02d}", f"hcondna{i}")):
            if f"{wave}_{stem}" in present:
                cols[f"{wave}_{stem}"] = f"agetold{i}_{k}"
    if f"{wave}_hcond96" in present:
        cols[f"{wave}_hcond96"] = "inv96"
    for age_col in (f"{wave}_dvage", f"{wave}_age_dv"):
        if age_col in present:
            cols[age_col] = "age"
            break
    df = pd.read_csv(path, sep="\t", usecols=["pidp"] + list(cols),
                     low_memory=False).rename(columns=cols)
    for c in df.columns:
        if c != "pidp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan
    df["wave"] = WAVES.index(wave) + 1
    return df


def build_condition_history(
    raw_dir: Path | None = None, *, waves: Sequence[str] = WAVES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Person-wave frame of ever-indicators, diagnosis ages, and group counts.

    Columns per code i: ``ever_{i}`` (0/1; NaN before the person's first
    inventory), ``diagage_{i}`` (age at diagnosis where known). Group counts
    ``n_{group}`` count evers; ``nrec10_{group}`` count only conditions
    diagnosed within the last 10 years (NaN when any needed diagnosis age is
    unknown ONLY IF the condition is present; absent conditions need no age).
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else UKHLS_PANEL_DIR
    frames = [_read_wave(raw_dir, w) for w in waves]
    long = pd.concat(frames, ignore_index=True).sort_values(
        ["pidp", "wave"], ignore_index=True)

    inv_cols = [c for c in long.columns if c.startswith("inv")]
    long["inventory_here"] = long[inv_cols].notna().any(axis=1)
    g = long.groupby("pidp", sort=False)
    long["inventory_seen"] = g["inventory_here"].cummax()

    out = long[["pidp", "wave", "age", "inventory_seen"]].copy()
    for i in CODES:
        inv = long[f"inv{i}"] if f"inv{i}" in long else pd.Series(np.nan, index=long.index)
        flag = inv == 1
        for c in [c for c in long.columns if c.startswith(f"new{i}_")]:
            flag = flag | (long[c] == 1)
        ever = flag.groupby(long["pidp"]).cummax()
        out[f"ever_{i}"] = np.where(long["inventory_seen"], ever.astype(float), np.nan)

        # diagnosis age: age-told where recorded at a report wave, else the
        # in-panel age at the first flag; the min over waves carries the
        # earliest dating to every later wave.
        told = pd.Series(np.nan, index=long.index)
        for c in [c for c in long.columns if c.startswith(f"agetold{i}_")]:
            told = told.fillna(long[c])
        at_flag = np.where(flag, np.where(told.notna(), told, long["age"]), np.nan)
        diag = pd.Series(at_flag, index=long.index).groupby(long["pidp"]).transform("min")
        out[f"diagage_{i}"] = np.where(out[f"ever_{i}"] == 1, diag, np.nan)

    for name, codes in CONDITION_GROUPS.items():
        evers = out[[f"ever_{c}" for c in codes]]
        out[f"n_{name}"] = evers.sum(axis=1).where(evers.notna().any(axis=1))
        recents = []
        for c in codes:
            since = out["age"] - out[f"diagage_{c}"]
            rec = np.where(out[f"ever_{c}"] == 1,
                           (since <= 10).astype(float), 0.0)
            # present but undatable -> unknown recency
            rec = np.where((out[f"ever_{c}"] == 1) & out[f"diagage_{c}"].isna(),
                           np.nan, rec)
            rec = np.where(out[f"ever_{c}"].isna(), np.nan, rec)
            recents.append(rec)
        rec_mat = np.column_stack(recents)
        out[f"nrec10_{name}"] = np.where(
            np.isnan(rec_mat).any(axis=1), np.nan, rec_mat.sum(axis=1))

    out["n_chronic"] = out[[f"ever_{i}" for i in CODES]].sum(axis=1).where(
        out["inventory_seen"])
    if verbose:
        seen = out["inventory_seen"]
        print(f"condition history: {len(out):,} person-waves, "
              f"{out['pidp'].nunique():,} people; "
              f"inventory seen for {seen.mean():.1%} of person-waves")
    return out


__all__ = [
    "CODES",
    "CONDITION_GROUPS",
    "CONDITION_LABELS",
    "build_condition_history",
]
