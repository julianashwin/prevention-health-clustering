"""Shared loading for the PSID replication of the paper's main-text figures.

Everything here reads PSID files only (data/processed/psid/) and writes under
descriptives/psid/ and artifacts/descriptives/psid/. Column names follow the UKHLS
scripts where that lets their helpers be reused unchanged: the person id is renamed
pid -> pidp.

The PSID "contract" mirrors the UKHLS health contract (health_lifecycle_20_89_minobs3_v1):
the five-testlet GRM theta (and h) from data_cleaning/psid/02_build_psid_health.py, ages
20-89, one row per person-age (a person interviewed twice at one integer age has the two
scores averaged), people with at least three observed ages. Waves are biennial, 2003-2023.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "descriptives"))      # _paper_common, _style, the numbered UKHLS scripts
sys.path.insert(0, str(ROOT / "src"))

PSID = ROOT / "data" / "processed" / "psid"
FIG = HERE / "figures" / "main"
DESC = ROOT / "artifacts" / "descriptives" / "psid"
UK_FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DESC.mkdir(parents=True, exist_ok=True)

MIN_AGE, MAX_AGE, K = 20, 90, 3
AGES = np.arange(MIN_AGE, MAX_AGE + 1)
LAST_WAVE = 2023
WAVE_STEP = 2
SRC = "PSID 2003-2023, biennial"

EDUC_ORDER = ["less than high school", "high school", "some college", "bachelor's or more"]
REGION_NAMES = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
RACE_GROUPS = ["black", "hispanic", "asian", "other"]          # white is the reference


def scores() -> pd.DataFrame:
    """Person-wave scores (pidp, wave, age, theta, h), ages 20-90."""
    m = pd.read_parquet(PSID / "measures" / "psid_health_measure.parquet", columns=["pid", "wave", "age", "theta", "h"])
    m = m.rename(columns={"pid": "pidp"})
    m = m[m["age"].between(MIN_AGE, MAX_AGE)].assign(age=lambda x: x["age"].astype(int))
    return m.reset_index(drop=True)


def contract(min_obs: int = 3) -> pd.DataFrame:
    """The PSID health contract: ages 20-89, one row per person-age, people with >= min_obs ages.
    Columns pidp, age, wave (the first wave at that age), theta, h."""
    m = scores()
    m = m[m["age"].between(MIN_AGE, 89)]
    g = m.groupby(["pidp", "age"], as_index=False).agg(wave=("wave", "first"), theta=("theta", "mean"), h=("h", "mean"))
    n = g.groupby("pidp")["age"].transform("size")
    return g[n >= min_obs].sort_values(["pidp", "age"]).reset_index(drop=True)


_PANEL: dict = {}


def panel(cols) -> pd.DataFrame:
    p = pd.read_parquet(PSID / "psid_panel.parquet", columns=["pid", "wave", *cols])
    return p.rename(columns={"pid": "pidp"})


def cost_rows() -> pd.DataFrame:
    """The PSID in-patient cost index per person-wave, at the UKHLS index's stage 1 flat rates
    (prevention_health_clustering.measures.cost: UnitCosts, inpatient_cost, spell rule "one").
    A year with any non-maternity overnight stay counts as one admission (PSID records nights,
    not admissions); 0-1 nights at the non-elective short-stay price, longer stays at the
    long-stay cost per bed day for every night; births already excluded by the panel builder
    (hosp_any_nonmat, hosp_nights_nonmat). Refers to the calendar year before the interview.
    Columns pidp, wave, cost (pounds, 2024/25 prices). Waves 2003-2023."""
    from prevention_health_clustering.measures.cost import UnitCosts, inpatient_cost
    p = panel(["hosp_any_nonmat", "hosp_nights_nonmat"])
    p = p[(p["wave"] >= 2003) & p["hosp_any_nonmat"].notna()]
    p = p[~((p["hosp_any_nonmat"] == 1) & p["hosp_nights_nonmat"].isna())]      # a stay of unknown length: dropped
    frame = pd.DataFrame({"hosp": p["hosp_any_nonmat"].to_numpy(), "hospd": p["hosp_nights_nonmat"].to_numpy(),
                          "hospch": np.zeros(len(p))}, index=p.index)
    p = p.assign(cost=inpatient_cost(frame, UnitCosts(), spell_rule="one").to_numpy())
    return p[["pidp", "wave", "cost"]].reset_index(drop=True)


def mortality_rows() -> pd.DataFrame:
    """Every person-wave at risk with the outcome 'died before the next wave' (pidp, wave, died).
    PSID's mort_event is 1 on a person's last panel row when they died within two years of that
    interview: a two-year window, against one year in the UKHLS. The last wave (2023) has no
    follow-up and is dropped; attrition without a recorded death counts as survival, as in the UKHLS."""
    p = panel(["mort_event"])
    p = p[p["wave"] < LAST_WAVE]
    return p.rename(columns={"mort_event": "died"})[["pidp", "wave", "died"]]


def observables(pids=None) -> pd.DataFrame:
    """One row per person: female, educ_group (four US groups from educ_best), income_rank_mean (the
    mean over waves of the within-wave rank of equivalised family income), region (first valid of the
    four census regions; Alaska and Hawaii folded into the West, abroad missing), race_group (white,
    black, hispanic, asian, other)."""
    p = panel(["sex", "educ_best", "faminc_rank", "region", "race_group"]).sort_values(["pidp", "wave"])
    if pids is not None:
        p = p[p["pidp"].isin(pids)]
    reg = p["region"].replace({5: 4, 6: np.nan})
    first = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan  # noqa: E731
    per = pd.DataFrame({
        "female": p.groupby("pidp")["sex"].first().eq(2).astype(float),
        "educ_best": p.groupby("pidp")["educ_best"].max(),
        "income_rank_mean": p.groupby("pidp")["faminc_rank"].mean(),
        "region": reg.groupby(p["pidp"]).agg(first),
        "race_group": p.groupby("pidp")["race_group"].agg(first),
    })
    per["educ_group"] = per["educ_best"].map({i + 1: g for i, g in enumerate(EDUC_ORDER)})
    return per


def obs_dummies(per: pd.DataFrame) -> pd.DataFrame:
    """Dummies used as predictors: education (less than high school the reference), region (South the
    reference, the largest), ethnic group (white the reference)."""
    out = pd.DataFrame(index=per.index)
    for g in EDUC_ORDER[1:]:
        out[f"educ_{g}"] = (per["educ_group"] == g).astype(float).where(per["educ_group"].notna())
    for r in (1, 2, 4):
        out[f"region_{r}"] = (per["region"] == r).astype(float).where(per["region"].notna())
    for g in RACE_GROUPS:
        out[f"ethnic_{g}"] = (per["race_group"] == g).astype(float).where(per["race_group"].notna())
    return out


EDUC_COLS = [f"educ_{g}" for g in EDUC_ORDER[1:]]
OBS = ["income_rank_mean", "female"] + EDUC_COLS
REGETH = [f"region_{r}" for r in (1, 2, 4)] + [f"ethnic_{g}" for g in RACE_GROUPS]


# ---- K-means outputs (written by 22_psid_kmeans.py) -----------------------------------
KM_LABELS = DESC / "psid_kmeans_labels.parquet"
KM_TRAJ = DESC / "psid_kmeans_trajectories.csv"
KM_COMP = DESC / "psid_kmeans_composition.csv"
KM_RUNS = DESC / "psid_kmeans_runs.csv"


def load_labels(variant: str = "theta", lo: int = MIN_AGE, hi: int = MAX_AGE) -> pd.Series:
    lab = pd.read_parquet(KM_LABELS)
    lab = lab[(lab["variant"] == variant) & (lab["lo"] == lo) & (lab["hi"] == hi)]
    return lab.set_index("pidp")["cluster"].astype(int)


def load_trajectories(variant: str = "theta", lo: int = MIN_AGE, hi: int = MAX_AGE) -> pd.DataFrame:
    t = pd.read_csv(KM_TRAJ)
    return t[(t["variant"] == variant) & (t["lo"] == lo) & (t["hi"] == hi)].copy()


def load_composition(variant: str = "theta", lo: int = MIN_AGE, hi: int = MAX_AGE) -> pd.DataFrame:
    c = pd.read_csv(KM_COMP)
    return c[(c["variant"] == variant) & (c["lo"] == lo) & (c["hi"] == hi)].copy()


def footer(fig, text: str, y: float = -0.01, size: float = 7.4) -> None:
    from _style import INK2
    fig.text(0.01, y, text, fontsize=size, color=INK2, va="top")


def hash_test(pids: pd.Series) -> np.ndarray:
    """The 70/30 person split of the UKHLS scripts (a fixed hash of the id)."""
    return ((pd.util.hash_pandas_object(pids, index=False) % 10) >= 7).to_numpy()


os.environ.setdefault("OMP_NUM_THREADS", "1")
