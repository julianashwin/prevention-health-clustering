"""Frozen sample contracts: the roster every model must share.

A contract fixes the analysis population once — age window, minimum observations
per person, and the deterministic rule for collapsing duplicate person-age cells
— so that every model is fitted on exactly the same people and rows.

This is a lean reimplementation of the predecessor project's
``typology/sample_contract.py``. The collapse, eligibility and centering rules
are equivalent, and reproduce its frozen roster exactly: 50,194 persons and
446,966 person-age rows for the single-metric PCS contract. Two deliberate
changes: it accepts **multiple metrics**, so a joint PCS+MCS contract is a
first-class object; and it does not generate shell job scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from prevention_health_clustering.config import (
    DEFAULT_AGE_CENTER,
    DEFAULT_HEALTH_METRIC,
    DEFAULT_LIFECYCLE_MAX_AGE,
    DEFAULT_LIFECYCLE_MIN_AGE,
    DEFAULT_MIN_OBS_PER_PERSON,
    PROCESSED_DATA_DIR,
)

DEFAULT_PANEL_PATH = PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv"
DEFAULT_CONTRACT_ROOT = PROCESSED_DATA_DIR / "contracts"

BASE_COLUMNS = ("pidp", "age", "wave", "birthy")


@dataclass(frozen=True)
class ContractSpec:
    """The complete, hashable definition of a sample contract."""

    contract_id: str
    metrics: tuple[str, ...]
    min_age: int = DEFAULT_LIFECYCLE_MIN_AGE
    max_age: int = DEFAULT_LIFECYCLE_MAX_AGE
    min_obs_per_person: int = DEFAULT_MIN_OBS_PER_PERSON
    age_center: int = DEFAULT_AGE_CENTER
    require_all_metrics: bool = True

    @property
    def stop_age(self) -> int:
        """Exclusive upper bound used for internal slicing."""
        return self.max_age + 1

    def as_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "metrics": list(self.metrics),
            "age_min_inclusive": self.min_age,
            "age_max_inclusive": self.max_age,
            "min_obs_per_person": self.min_obs_per_person,
            "age_center": self.age_center,
            "require_all_metrics": self.require_all_metrics,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:20]


@dataclass
class ContractResult:
    spec: ContractSpec
    long: pd.DataFrame
    roster: pd.DataFrame
    age_support: pd.DataFrame
    manifest: dict = field(default_factory=dict)


PCS_LIFECYCLE = ContractSpec(
    contract_id="pcs_lifecycle_20_89_minobs3_v1",
    metrics=(DEFAULT_HEALTH_METRIC,),
)

PCS_MCS_LIFECYCLE = ContractSpec(
    contract_id="pcs_mcs_lifecycle_20_89_minobs3_v1",
    metrics=("sf12pcs_dv", "sf12mcs_dv"),
)


def _collapse_person_age_duplicates(
    frame: pd.DataFrame, *, metrics: Sequence[str]
) -> pd.DataFrame:
    """Collapse repeated person-age cells deterministically.

    Metric values are averaged, the earliest wave is retained, and birth year
    takes the first non-missing value. Sorting by ``(pidp, age, wave)`` before
    grouping makes the birth-year choice order-independent.
    """
    aggregation: dict[str, object] = {metric: "mean" for metric in metrics}
    aggregation["wave"] = "min"
    aggregation["birthy"] = lambda values: (
        values.dropna().iloc[0] if values.notna().any() else pd.NA
    )
    grouped = frame.sort_values(["pidp", "age", "wave"]).groupby(
        ["pidp", "age"], as_index=False, sort=True
    )
    return grouped.agg(aggregation)


def _build_person_roster(long: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    grouped = long.groupby("pidp", as_index=False)
    roster = grouped.agg(
        n_observations=(metrics[0], "count"),
        min_age=("age", "min"),
        max_age=("age", "max"),
        min_wave=("wave", "min"),
        max_wave=("wave", "max"),
        birthy=("birthy", "first"),
    )
    return roster.sort_values("pidp").reset_index(drop=True)


def _build_age_support(long: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {"n_persons": ("pidp", "nunique")}
    for metric in metrics:
        aggregations[f"n_{metric}"] = (metric, "count")
        aggregations[f"mean_{metric}"] = (metric, "mean")
        aggregations[f"sd_{metric}"] = (metric, "std")
    support = long.groupby("age", as_index=False).agg(**aggregations)
    return support.sort_values("age").reset_index(drop=True)


def build_contract(spec: ContractSpec, panel: pd.DataFrame) -> ContractResult:
    """Apply a contract to a processed long panel."""
    required = set(BASE_COLUMNS) | set(spec.metrics)
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {', '.join(missing)}")

    columns = list(BASE_COLUMNS) + list(spec.metrics)
    working = panel.loc[:, columns].copy()
    for column in columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    source_rows = len(working)
    source_persons = int(working["pidp"].nunique())

    if spec.require_all_metrics:
        working = working.dropna(subset=["pidp", "age", *spec.metrics])
    else:
        working = working.dropna(subset=["pidp", "age"])
        working = working.dropna(subset=list(spec.metrics), how="all")

    working["age"] = working["age"].round().astype(int)
    working = working[
        (working["age"] >= spec.min_age) & (working["age"] < spec.stop_age)
    ]
    age_filtered_rows = len(working)

    duplicate_rows = int(working.duplicated(subset=["pidp", "age"], keep=False).sum())
    working = _collapse_person_age_duplicates(working, metrics=spec.metrics)

    obs_per_person = working.groupby("pidp")["age"].nunique()
    eligible = obs_per_person[obs_per_person >= spec.min_obs_per_person].index
    long = working[working["pidp"].isin(eligible)].copy()
    long = long.sort_values(["pidp", "age"]).reset_index(drop=True)
    if long.empty:
        raise ValueError("Contract filters produced no eligible rows.")

    long["pidp"] = long["pidp"].round().astype("int64")
    long["wave"] = long["wave"].round().astype("Int64")
    long["birthy"] = long["birthy"].round().astype("Int64")
    long["age_c"] = long["age"] - spec.age_center
    long = long.loc[:, ["pidp", "age", "age_c", "wave", "birthy", *spec.metrics]]

    roster = _build_person_roster(long, metrics=spec.metrics)
    age_support = _build_age_support(long, metrics=spec.metrics)

    manifest = {
        "schema_version": "phc-sample-contract/v1",
        "spec": spec.as_dict(),
        "spec_fingerprint": spec.fingerprint(),
        "counts": {
            "source_rows": source_rows,
            "source_persons": source_persons,
            "age_filtered_rows": age_filtered_rows,
            "duplicate_person_age_rows": duplicate_rows,
            "retained_rows": int(len(long)),
            "retained_persons": int(long["pidp"].nunique()),
        },
        "metric_moments": {
            metric: {
                "mean": float(long[metric].mean()),
                # ddof=1 matches the published fits' standardisation.
                "sd": float(long[metric].std(ddof=1)),
            }
            for metric in spec.metrics
        },
    }
    return ContractResult(
        spec=spec, long=long, roster=roster, age_support=age_support, manifest=manifest
    )


def write_contract(result: ContractResult, output_dir: Path) -> Path:
    """Write the contract products and return the directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.long.to_csv(output_dir / "long.csv", index=False)
    result.roster.to_csv(output_dir / "person_roster.csv", index=False)
    result.age_support.to_csv(output_dir / "age_support.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n"
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a frozen sample contract from the processed panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CONTRACT_ROOT)
    parser.add_argument("--contract-id", default=PCS_LIFECYCLE.contract_id)
    parser.add_argument("--metrics", nargs="+", default=list(PCS_LIFECYCLE.metrics))
    parser.add_argument("--min-age", type=int, default=DEFAULT_LIFECYCLE_MIN_AGE)
    parser.add_argument("--max-age", type=int, default=DEFAULT_LIFECYCLE_MAX_AGE)
    parser.add_argument(
        "--min-obs-per-person", type=int, default=DEFAULT_MIN_OBS_PER_PERSON
    )
    parser.add_argument("--age-center", type=int, default=DEFAULT_AGE_CENTER)
    parser.add_argument(
        "--any-metric",
        action="store_true",
        help="Keep a row when any metric is observed, instead of requiring all.",
    )
    args = parser.parse_args(argv)

    spec = ContractSpec(
        contract_id=args.contract_id,
        metrics=tuple(args.metrics),
        min_age=args.min_age,
        max_age=args.max_age,
        min_obs_per_person=args.min_obs_per_person,
        age_center=args.age_center,
        require_all_metrics=not args.any_metric,
    )
    panel = pd.read_csv(args.panel, low_memory=False)
    result = build_contract(spec, panel)
    output_dir = write_contract(result, args.output_root / spec.contract_id)

    counts = result.manifest["counts"]
    print(
        f"{spec.contract_id}: {counts['retained_persons']:,} persons, "
        f"{counts['retained_rows']:,} person-age rows -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
