"""Frozen sample contracts for the GRM clustering channels.

Builds lifecycle contracts (ages 20-89, >= 3 observations, same collapse and
standardisation conventions as the PCS contracts) for:

  physgrm_lifecycle_20_89_minobs3_v1   theta_phys_full  (P-FULL)
  combgrm_lifecycle_20_89_minobs3_v1   theta_combined

The panel comes from grm2_scores.parquet with birth year merged from
xwavedat (birthy, doby_dv fallback; needed by the runner's cohort machinery
even when unused).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import PROCESSED_DATA_DIR, UKHLS_PANEL_DIR
from prevention_health_clustering.contracts.sample_contract import (
    ContractSpec,
    build_contract,
    write_contract,
)

SPECS = [
    ContractSpec(contract_id="physgrm_lifecycle_20_89_minobs3_v1",
                 metrics=("theta_phys_full",)),
    ContractSpec(contract_id="combgrm_lifecycle_20_89_minobs3_v1",
                 metrics=("theta_combined",)),
]


def main() -> int:
    ensure_runtime_directories()
    scores = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "grm2_scores.parquet",
        columns=["pidp", "wave", "age", "theta_phys_full", "theta_combined"])
    xw = pd.read_csv(UKHLS_PANEL_DIR / "xwavedat.tab", sep="\t",
                     usecols=["pidp", "birthy", "doby_dv"], low_memory=False)
    for c in ("birthy", "doby_dv"):
        xw[c] = pd.to_numeric(xw[c], errors="coerce")
        xw.loc[xw[c] < 0, c] = np.nan
    # doby_dv fills the 111 people whose birthy is missing; zero remain.
    xw["birthy"] = xw["birthy"].fillna(xw["doby_dv"])
    panel = scores.merge(xw[["pidp", "birthy"]], on="pidp", how="left")

    out_root = PROCESSED_DATA_DIR / "contracts"
    for spec in SPECS:
        result = build_contract(spec, panel)
        path = write_contract(result, out_root / spec.contract_id)
        long = result.long
        m = spec.metrics[0]
        print(f"{spec.contract_id}: {long['pidp'].nunique():,} persons, "
              f"{len(long):,} rows -> {path}")
        print(f"  {m}: mean {long[m].mean():.4f}, sd {long[m].std(ddof=1):.4f}")
        n5 = (long.groupby('pidp').size() >= 5).sum()
        print(f"  persons with >= 5 observations (holdout sample): {n5:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
