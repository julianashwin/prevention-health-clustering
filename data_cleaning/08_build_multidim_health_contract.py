"""The multidimensional contract on the paper's health measure.

One lifecycle contract (ages 20-89, at least three observations, the same
collapse and standardisation conventions as the health contract) carrying
the two physical variants, the mental GRM and the mortality event, so a
multidimensional fit on (theta or h) plus mental health plus mortality runs
on the same people whatever the physical ruler:

    multidim_health_20_89_minobs3_v1    theta, h, theta_ment_nodepr, mort_event

Rows are person-waves where the health measure and the mental GRM (the bank
without the depression diagnosis) are both observed. The mental score is
missing on 2% of health-contract rows, almost all in wave 1 where the GHQ
self-completion is thinner, so the people are the health contract's minus
the few hundred whose third observation is one of those rows.

Mortality: dcsedfl_dv == 1 marks a death established at a later wave; the
event is placed on the person's last observed wave, the survival-usable
encoding (the death is recorded at the wave AFTER the last interview, see
the archived 08_build_multidim_contract.py). Rows carry mort_event 1 on a
decedent's last row and 0 elsewhere. Birth year from xwavedat as always.

Reads health_measure.parquet and grm2_scores.parquet; writes under
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

SPEC = ContractSpec(contract_id="multidim_health_20_89_minobs3_v1",
                    metrics=("theta", "h", "theta_ment_nodepr"))


def main() -> int:
    ensure_runtime_directories()
    md = PROCESSED_DATA_DIR / "measures"
    health = pd.read_parquet(md / "health_measure.parquet", columns=["pidp", "wave", "age", "theta", "h"])
    mental = pd.read_parquet(md / "grm2_scores.parquet", columns=["pidp", "wave", "theta_ment_nodepr"])
    scores = health.merge(mental, on=["pidp", "wave"], how="left")
    print(f"health rows {len(health):,}; with a mental score {scores['theta_ment_nodepr'].notna().sum():,}")
    scores = scores.dropna(subset=["theta", "h", "theta_ment_nodepr"])

    xw = pd.read_csv(UKHLS_PANEL_DIR / "xwavedat.tab", sep="\t",
                     usecols=["pidp", "birthy", "doby_dv", "dcsedfl_dv"], low_memory=False)
    for c in ("birthy", "doby_dv", "dcsedfl_dv"):
        xw[c] = pd.to_numeric(xw[c], errors="coerce")
    for c in ("birthy", "doby_dv"):
        xw.loc[xw[c] < 0, c] = np.nan
    xw["birthy"] = xw["birthy"].fillna(xw["doby_dv"])
    xw["died"] = np.where(xw["dcsedfl_dv"].isin([1, 2, 3]), (xw["dcsedfl_dv"] == 1).astype(float), np.nan)
    panel = scores.merge(xw[["pidp", "birthy", "died"]], on="pidp", how="left")

    # the event sits on the last observed wave of the person's full record
    panel = panel.sort_values(["pidp", "wave"]).reset_index(drop=True)
    last = panel.groupby("pidp")["wave"].transform("max") == panel["wave"]
    panel["mort_event"] = np.where(panel["died"].isna(), np.nan, (last & (panel["died"] == 1)).astype(float))

    result = build_contract(SPEC, panel)
    long = result.long.merge(panel[["pidp", "wave", "mort_event"]].drop_duplicates(subset=["pidp", "wave"]),
                             on=["pidp", "wave"], how="left")
    result = result.__class__(spec=result.spec, long=long, roster=result.roster,
                              age_support=result.age_support, manifest=result.manifest)
    path = write_contract(result, PROCESSED_DATA_DIR / "contracts" / SPEC.contract_id)
    print(f"\n{SPEC.contract_id}: {long['pidp'].nunique():,} persons, {len(long):,} rows -> {path}")
    for m in SPEC.metrics:
        print(f"  {m:18s} mean {long[m].mean():+.4f} sd {long[m].std(ddof=1):.4f}")
    ev = long["mort_event"]
    print(f"  mortality observed on {ev.notna().mean():.1%} of rows; events {int(ev.sum()):,} "
          f"({ev.mean():.3%} of rows); a decedent's event on their last contract row: "
          f"{int((long.groupby('pidp')['mort_event'].transform('max') == 1).groupby(long['pidp']).first().sum()):,} people")
    print(f"  persons with >= 5 observations (holdout sample): {(long.groupby('pidp').size() >= 5).sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
