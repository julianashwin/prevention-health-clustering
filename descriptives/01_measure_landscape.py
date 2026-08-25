"""Assemble the unified measure panel and its comparison tables.

Merges every constructed health measure onto one person-wave frame:
SF-12 family (PCS/MCS, UK variants, phys-only, SF-6D, subscales), the
original 4-testlet GRM, the new physical and mental GRMs (with the ever /
functioning-only / recent-10y sensitivity variants), GHQ-12 Likert (computed
from the items), long-standing illness, and the chronic count.

Outputs:
  data/processed/measures/measure_panel.parquet
  artifacts/descriptives/measure_correlations.csv
  artifacts/descriptives/measure_age_means.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import (
    ARTIFACTS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from prevention_health_clustering.measures.items import GHQ_ITEM_STEMS

MEASURES = [
    "sf12pcs_dv", "sf12mcs_dv", "PCS_phys_only", "PCS_uk_promax",
    "sf6d_utility", "grm_theta", "theta_phys_func", "theta_phys_full",
    "theta_phys_rec10", "theta_ment", "grmh_phys_full", "grmh_ment",
    "ghq_likert", "ill", "n_chronic",
]


def main() -> int:
    ensure_runtime_directories()
    md = PROCESSED_DATA_DIR / "measures"
    sf = pd.read_parquet(md / "sf12_measures.parquet")
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet",
                            columns=["pidp", "wave", "health"]
                            + list(GHQ_ITEM_STEMS))
    old = pd.read_parquet(md / "grm_scores.parquet",
                          columns=["pidp", "wave", "grm_theta", "grmh"])
    new = pd.read_parquet(md / "grm2_scores.parquet")
    chron = pd.read_parquet(md / "chronic_conditions.parquet",
                            columns=["pidp", "wave", "n_chronic"])

    ghq = items[list(GHQ_ITEM_STEMS)]
    complete = ghq.notna().all(axis=1)
    items = items.assign(
        ghq_likert=(ghq.sum(axis=1) - 12).where(complete),
        ill=items["health"].map({1: 1.0, 2: 0.0}),
    )[["pidp", "wave", "ghq_likert", "ill"]]

    panel = (
        sf.merge(items, on=["pidp", "wave"], how="left")
        .merge(old, on=["pidp", "wave"], how="left")
        .merge(new.drop(columns=["age"]), on=["pidp", "wave"], how="left")
        .merge(chron, on=["pidp", "wave"], how="left")
    )
    panel.to_parquet(md / "measure_panel.parquet", index=False)
    print(f"measure panel: {len(panel):,} rows x {len(panel.columns)} cols")

    out_dir = ARTIFACTS_DIR / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr = panel[MEASURES].corr()
    corr.to_csv(out_dir / "measure_correlations.csv")
    print("\ncorrelations with the new physical GRM (theta_phys_full):")
    print(corr["theta_phys_full"].round(3).to_string())
    print("\ncorrelations with the new mental GRM (theta_ment):")
    print(corr["theta_ment"].round(3).to_string())

    ok = panel["age"].notna() & panel["age"].between(20, 90)
    means = (panel[ok].assign(age=panel.loc[ok, "age"].astype(int))
             .groupby("age")[MEASURES].agg(["mean", "count"]))
    means.to_csv(out_dir / "measure_age_means.csv")
    print(f"\nwrote {out_dir / 'measure_correlations.csv'} and measure_age_means.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
