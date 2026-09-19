"""Frozen contract for the four-channel multidimensional model.

Rows are person-waves where BOTH GRM thetas are observed -- the physical
functioning bank (P-FUNC) and the mental bank without the DEPR diagnosis
item, so neither channel carries the condition inventory's BHPS exclusion
(docs section 3.4). Onto those rows are attached, each with its own observed
mask:

  chronic   cumulated chronic-condition count (missing where the inventory
            was never asked -- 24% of rows, a survey-design gate on people
            who are indistinguishable on health)
  mort      discrete-time mortality hazard: 1 on a decedent's last observed
            wave, 0 elsewhere. dcsedfl_dv/dcsedw_dv record the wave at which
            death was ESTABLISHED, which is always after the last interview,
            so the last observed record is the survival-usable encoding.

Output: data/processed/contracts/multidim_lifecycle_20_89_minobs3_v1/

Archived with the banks it scores (06_build_grm2.py); kept runnable because
the multidimensional fits in artifacts/ were estimated on this contract.
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

SPEC = ContractSpec(contract_id="multidim_lifecycle_20_89_minobs3_v1",
                    metrics=("theta_phys_func", "theta_ment_nodepr"))


def main() -> int:
    ensure_runtime_directories()
    md = PROCESSED_DATA_DIR / "measures"
    scores = pd.read_parquet(md / "grm2_scores.parquet",
                             columns=["pidp", "wave", "age",
                                      "theta_phys_func", "theta_ment_nodepr"])
    scores = scores.dropna(subset=["theta_phys_func", "theta_ment_nodepr"])
    print(f"both GRM channels observed: {len(scores):,} person-waves, "
          f"{scores['pidp'].nunique():,} people")

    chron = pd.read_parquet(md / "chronic_conditions.parquet",
                            columns=["pidp", "wave", "n_chronic"])
    xw = pd.read_csv(UKHLS_PANEL_DIR / "xwavedat.tab", sep="\t",
                     usecols=["pidp", "birthy", "doby_dv", "dcsedfl_dv",
                              "dcsedw_dv"], low_memory=False)
    for c in ("birthy", "doby_dv", "dcsedfl_dv", "dcsedw_dv"):
        xw[c] = pd.to_numeric(xw[c], errors="coerce")
        if c in ("birthy", "doby_dv"):
            xw.loc[xw[c] < 0, c] = np.nan
    xw["birthy"] = xw["birthy"].fillna(xw["doby_dv"])
    xw["died"] = np.where(xw["dcsedfl_dv"].isin([1, 2, 3]),
                          (xw["dcsedfl_dv"] == 1).astype(float), np.nan)

    panel = (scores.merge(chron, on=["pidp", "wave"], how="left")
             .merge(xw[["pidp", "birthy", "died"]], on="pidp", how="left"))

    # mortality hazard: the event sits on the last observed wave
    panel = panel.sort_values(["pidp", "age"]).reset_index(drop=True)
    last = panel.groupby("pidp")["wave"].transform("max") == panel["wave"]
    panel["mort_event"] = np.where(panel["died"].isna(), np.nan,
                                   (last & (panel["died"] == 1)).astype(float))
    print(f"chronic observed on {panel['n_chronic'].notna().mean():.1%} of rows; "
          f"mortality observed on {panel['mort_event'].notna().mean():.1%}")
    print(f"mortality events: {int(panel['mort_event'].sum()):,} "
          f"({panel['mort_event'].mean():.3%} of rows)")

    result = build_contract(SPEC, panel)
    # build_contract keeps only the metrics; re-attach the auxiliary channels
    long = result.long.merge(
        panel[["pidp", "wave", "n_chronic", "mort_event"]].drop_duplicates(
            subset=["pidp", "wave"]),
        on=["pidp", "wave"], how="left")
    result = result.__class__(spec=result.spec, long=long,
                              roster=result.roster,
                              age_support=result.age_support,
                              manifest=result.manifest)
    path = write_contract(result, PROCESSED_DATA_DIR / "contracts" / SPEC.contract_id)
    print(f"\n{SPEC.contract_id}: {long['pidp'].nunique():,} persons, "
          f"{len(long):,} rows -> {path}")
    for m in SPEC.metrics:
        print(f"  {m:20s} mean {long[m].mean():+.4f} sd {long[m].std(ddof=1):.4f}")
    print(f"  chronic observed {long['n_chronic'].notna().mean():.1%}, "
          f"mean {long['n_chronic'].mean():.3f}")
    print(f"  mortality events {int(long['mort_event'].sum()):,} on "
          f"{long['mort_event'].notna().sum():,} rows")
    n5 = (long.groupby("pidp").size() >= 5).sum()
    print(f"  persons with >= 5 observations (holdout sample): {n5:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
