"""Build chronic-condition histories; validate against the EIT count.

Constructs per-condition ever-indicators, diagnosis ages, clinical-group
counts and 10-year-recency counts (measures/chronic.py), then:

  * validates the total count against the EIT note's build_chronic_count.rds
    (same recipe up to the documented hcondn/hcondns correction);
  * reports the diagnosis-timing coverage that the recency sensitivity rests
    on — what share of ever-flags carry a usable diagnosis age.

Output: data/processed/measures/chronic_conditions.parquet (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import PROCESSED_DATA_DIR
from prevention_health_clustering.measures.chronic import (
    CODES,
    CONDITION_GROUPS,
    CONDITION_LABELS,
    build_condition_history,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eit-reference", type=Path, default=None,
                        help="CSV export of the EIT chronic_count.rds")
    args = parser.parse_args(argv)
    ensure_runtime_directories()
    checks: list[tuple[str, bool, str]] = []

    hist = build_condition_history()

    seen = hist[hist["inventory_seen"]]
    print(f"\nprevalence of ever-diagnosis at person-wave level "
          f"(n={len(seen):,}):")
    for i in CODES:
        p = seen[f"ever_{i}"].mean()
        print(f"  {CONDITION_LABELS[i]:28s} {p:6.1%}")
    print("\ngroup count distributions:")
    for g in CONDITION_GROUPS:
        vc = seen[f"n_{g}"].value_counts(normalize=True).sort_index()
        print(f"  n_{g:7s} " + "  ".join(
            f"{int(k)}:{v:5.1%}" for k, v in vc.items() if k <= 3))

    # -- diagnosis-timing coverage (the recency sensitivity's foundation) ----
    print("\ndiagnosis-timing coverage among ever-flags:")
    total_flags, dated_flags = 0, 0
    for i in CODES:
        ever = seen[f"ever_{i}"] == 1
        dated = ever & seen[f"diagage_{i}"].notna()
        total_flags += int(ever.sum())
        dated_flags += int(dated.sum())
    cov = dated_flags / total_flags
    checks.append(("diagnosis ages cover most ever-flags", cov >= 0.90,
                   f"{cov:.1%} of {total_flags:,} condition-person-waves"))

    since_cols = []
    for i in CODES:
        s = (seen["age"] - seen[f"diagage_{i}"]).where(seen[f"ever_{i}"] == 1)
        since_cols.append(s)
    since = pd.concat(since_cols).dropna()
    print(f"  years since diagnosis (all dated condition-person-waves, "
          f"n={len(since):,}):")
    q = since.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("   " + "  ".join(f"p{int(k*100)}={v:.0f}" for k, v in q.items()))
    older = seen[seen["age"].between(65, 80)]
    s_old = pd.concat(
        [(older["age"] - older[f"diagage_{i}"]).where(older[f"ever_{i}"] == 1)
         for i in CODES]).dropna()
    print(f"  at ages 65-80: median {s_old.median():.0f} years, "
          f"share >10y {(s_old > 10).mean():.1%}, "
          f"share >20y {(s_old > 20).mean():.1%}")

    # -- EIT comparison ------------------------------------------------------
    if args.eit_reference is not None and args.eit_reference.exists():
        eit = pd.read_csv(args.eit_reference)
        m = hist.merge(eit, on=["pidp", "wave"], how="inner",
                       suffixes=("", "_eit"))
        m = m[m["n_chronic"].notna() & m["n_chronic_eit"].notna()]
        agree = (m["n_chronic"] == m["n_chronic_eit"]).mean()
        corr = m["n_chronic"].corr(m["n_chronic_eit"])
        diff = (m["n_chronic"] - m["n_chronic_eit"])
        # Ours is a documented near-superset (the EIT family missed MI/stroke
        # codes entirely and all wave-10+ renamed families): the gate is high
        # agreement AND that essentially every disagreement is ours >= EIT.
        n_disagree = int((diff != 0).sum())
        superset = float((diff >= 0).sum() / len(m))
        checks.append((
            "EIT count reproduced up to documented fixes",
            bool(agree >= 0.90 and corr >= 0.97 and superset >= 0.999),
            f"exact {agree:.2%}, corr {corr:.4f}, ours>=EIT on {superset:.3%}",
        ))
        print(f"\nEIT comparison: ours higher on {(diff > 0).mean():.2%} of "
              f"rows, lower on {(diff < 0).mean():.3%} "
              f"({n_disagree:,} disagreements)")

    out_path = PROCESSED_DATA_DIR / "measures" / "chronic_conditions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(hist):,} rows x {len(hist.columns)} cols)")

    print("\n=== checks ===")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:44s} {detail}")
    print("CHRONIC BUILD OK" if all_ok else "CHRONIC BUILD FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
