"""Build the frozen sample contracts from the processed panel.

    PYTHONPATH=src .venv/bin/python data_cleaning/02_build_contracts.py

Validation target: PCS contract must reproduce 50,194 persons / 446,966 rows
with PCS mean 49.1684050688419, matching the published six-fit-v1 bundle.
"""

import pandas as pd

from prevention_health_clustering.contracts.sample_contract import (
    DEFAULT_CONTRACT_ROOT,
    DEFAULT_PANEL_PATH,
    PCS_LIFECYCLE,
    PCS_MCS_LIFECYCLE,
    build_contract,
    write_contract,
)


def main() -> int:
    panel = pd.read_csv(DEFAULT_PANEL_PATH, low_memory=False)
    for spec in (PCS_LIFECYCLE, PCS_MCS_LIFECYCLE):
        result = build_contract(spec, panel)
        out = write_contract(result, DEFAULT_CONTRACT_ROOT / spec.contract_id)
        counts = result.manifest["counts"]
        print(f"{spec.contract_id}: {counts['retained_persons']:,} persons, "
              f"{counts['retained_rows']:,} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
