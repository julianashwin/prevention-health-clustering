"""CLI construction, logging, and runtime-directory helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prevention_health_clustering._utils import (
    build_cli_parser,
    log_script_end,
    log_script_start,
)
from prevention_health_clustering.config import (
    ARTIFACTS_DIR,
    DEFAULT_HEALTH_METRIC,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def ensure_runtime_directories() -> None:
    """Create the writable output roots. Called explicitly, never on import."""
    for directory in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def default_prepared_trajectory_filepath(
    processed_data_dir: Path | None = None,
    health_metrics: Sequence[str] | None = None,
) -> Path:
    """Canonical unlabeled wide-trajectory path for a metric set."""
    root = PROCESSED_DATA_DIR if processed_data_dir is None else processed_data_dir
    metrics = (DEFAULT_HEALTH_METRIC,) if health_metrics is None else tuple(health_metrics)
    return root / f"ukhls_indresp_trajectories_{'-'.join(metrics)}.csv"


def add_common_age_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-age", type=int, default=None)
    parser.add_argument("--max-age", type=int, default=None)


__all__ = [
    "add_common_age_args",
    "build_cli_parser",
    "default_prepared_trajectory_filepath",
    "ensure_runtime_directories",
    "log_script_end",
    "log_script_start",
]
