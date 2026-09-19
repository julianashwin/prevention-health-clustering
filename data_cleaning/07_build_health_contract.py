"""The frozen clustering contract on the paper's health measure.

One lifecycle contract (ages 20-89, at least three observations, the same
collapse and standardisation conventions as every earlier contract) carrying
the three reported variants of the measure as metrics, so a fit on theta, on
h and on the deficit index runs on exactly the same people and rows:

    health_lifecycle_20_89_minobs3_v1    theta, h, fi10

Birth year comes from xwavedat (birthy, doby_dv fallback), as for the
archived GRM contracts. Reads health_measure.parquet; writes under
data/processed/contracts/ (gitignored).
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

SPEC = ContractSpec(contract_id="health_lifecycle_20_89_minobs3_v1", metrics=("theta", "h", "fi10"))


def main() -> int:
    ensure_runtime_directories()
    scores = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                             columns=["pidp", "wave", "age", "theta", "h", "fi10"])
    xw = pd.read_csv(UKHLS_PANEL_DIR / "xwavedat.tab", sep="\t",
                     usecols=["pidp", "birthy", "doby_dv"], low_memory=False)
    for c in ("birthy", "doby_dv"):
        xw[c] = pd.to_numeric(xw[c], errors="coerce")
        xw.loc[xw[c] < 0, c] = np.nan
    xw["birthy"] = xw["birthy"].fillna(xw["doby_dv"])
    panel = scores.merge(xw[["pidp", "birthy"]], on="pidp", how="left")
    result = build_contract(SPEC, panel)
    path = write_contract(result, PROCESSED_DATA_DIR / "contracts" / SPEC.contract_id)
    long = result.long
    print(f"{SPEC.contract_id}: {long['pidp'].nunique():,} persons, {len(long):,} rows -> {path}")
    for m in SPEC.metrics:
        print(f"  {m}: mean {long[m].mean():.4f}, sd {long[m].std(ddof=1):.4f}")
    print(f"  persons with >= 5 observations (holdout sample): {(long.groupby('pidp').size() >= 5).sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
