"""The frozen clustering contract on the paper's health measure.

One lifecycle contract (ages 20-89, at least three observations) carrying
both reported variants of the measure as metrics, so a fit on theta and on h
runs on exactly the same people and rows:

    health_lifecycle_20_89_minobs3_v1    theta, h

Birth year comes from xwavedat (birthy, doby_dv fallback), as for the
every earlier contract. The Rockwood-style frailty index and its log
(04b_build_frailty.py) ride along as auxiliary columns, ``frailty`` and
``logfrailty``, averaged over a person-age's waves like the metrics (the log,
log(frailty + 1/31), taken of that average, so zeros sit at the floor); they are not metrics, so they change
neither the rows nor the manifest's moments. Reads health_measure.parquet and
frailty_index.parquet; writes under data/processed/contracts/ (gitignored).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import PROCESSED_DATA_DIR, UKHLS_PANEL_DIR
from prevention_health_clustering.measures.health import FRAILTY_SHIFT
from prevention_health_clustering.contracts.sample_contract import (
    ContractSpec,
    build_contract,
    write_contract,
)

SPEC = ContractSpec(contract_id="health_lifecycle_20_89_minobs3_v1", metrics=("theta", "h"))


def main() -> int:
    ensure_runtime_directories()
    scores = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                             columns=["pidp", "wave", "age", "theta", "h"])
    xw = pd.read_csv(UKHLS_PANEL_DIR / "xwavedat.tab", sep="\t",
                     usecols=["pidp", "birthy", "doby_dv"], low_memory=False)
    for c in ("birthy", "doby_dv"):
        xw[c] = pd.to_numeric(xw[c], errors="coerce")
        xw.loc[xw[c] < 0, c] = np.nan
    xw["birthy"] = xw["birthy"].fillna(xw["doby_dv"])
    panel = scores.merge(xw[["pidp", "birthy"]], on="pidp", how="left")
    result = build_contract(SPEC, panel)
    # auxiliary measures: person-age means over the waves the contract collapsed, not metrics
    fr = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "frailty_index.parquet", columns=["pidp", "wave", "frailty"])
    fr = panel[["pidp", "wave", "age"]].assign(age=lambda x: x["age"].astype(int)).merge(fr, on=["pidp", "wave"])
    fr = fr.groupby(["pidp", "age"], as_index=False)["frailty"].mean()
    fr["logfrailty"] = np.log(fr["frailty"] + FRAILTY_SHIFT)
    long = result.long.merge(fr, on=["pidp", "age"], how="left")
    assert len(long) == len(result.long)
    result = result.__class__(spec=result.spec, long=long, roster=result.roster,
                              age_support=result.age_support, manifest=result.manifest)
    path = write_contract(result, PROCESSED_DATA_DIR / "contracts" / SPEC.contract_id)
    long = result.long
    print(f"{SPEC.contract_id}: {long['pidp'].nunique():,} persons, {len(long):,} rows -> {path}")
    for m in SPEC.metrics:
        print(f"  {m}: mean {long[m].mean():.4f}, sd {long[m].std(ddof=1):.4f}")
    print(f"  frailty on {long['frailty'].notna().mean():.1%} of rows, log frailty on {long['logfrailty'].notna().mean():.1%}")
    print(f"  persons with >= 5 observations (holdout sample): {(long.groupby('pidp').size() >= 5).sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
