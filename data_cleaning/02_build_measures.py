"""Extract the SF-12 and GHQ items from the raw UKHLS waves into the item cache.

Everything downstream reads this cache: the health GRM (04_build_health.py),
the frailty index (04b_build_frailty.py), the mental GRM (04c_build_mental_grm.py),
the cost index (05_build_cost_proxy.py) and the observables (06_build_observables.py).

Output (licensed-data derivative, gitignored):
  data/interim/sf12_items_long.parquet     one row per person-wave with the
                                           SF-12 items, the GHQ-12 items, age
                                           and the cross-wave identifiers

    PYTHONPATH=src .venv/bin/python data_cleaning/02_build_measures.py [--refresh-items]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR
from prevention_health_clustering.measures.items import extract_sf12_items


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-items", action="store_true",
                        help="re-extract items from the raw tab files")
    args = parser.parse_args(argv)
    ensure_runtime_directories()

    cache = INTERIM_DATA_DIR / "sf12_items_long.parquet"
    if cache.exists() and not args.refresh_items:
        items = pd.read_parquet(cache)
        print(f"items: cached {cache} ({len(items):,} person-years)")
    else:
        print("items: extracting from raw indresp waves")
        items = extract_sf12_items()
        items.to_parquet(cache, index=False)
        print(f"items: wrote {cache} ({len(items):,} person-years)")

    item_cols = ["sf1", "sf2a", "sf2b", "sf3a", "sf3b", "sf4a",
                 "sf4b", "sf5", "sf6a", "sf6b", "sf6c", "sf7"]
    complete = items[item_cols].notna().all(axis=1)
    print(f"  {complete.sum():,} person-years with all twelve SF-12 items, "
          f"{items['pidp'].nunique():,} people, waves {items['wave'].min()}-{items['wave'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
