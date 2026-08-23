
from __future__ import annotations

import pandas as pd

from prevention_health_clustering.config import MISSING_DATA_THRESHOLD

"""Minimal logging and CLI helpers shared by the data layer."""


import argparse
import sys
import time
from typing import Any

_START: dict[str, float] = {}


def log_script_start(name: str, about: str = "", **outputs: Any) -> None:
    _START[name] = time.time()
    print(f"[start] {name}", file=sys.stderr)
    if about:
        print(f"[about] {about}", file=sys.stderr)
    for key, value in outputs.items():
        print(f"  - {key}: {value}", file=sys.stderr)


def log_script_end(name: str, **outputs: Any) -> None:
    elapsed = time.time() - _START.pop(name, time.time())
    for key, value in outputs.items():
        print(f"  - {key}: {value}", file=sys.stderr)
    print(f"[done] {name} ({elapsed:.1f}s)", file=sys.stderr)


def build_cli_parser(description: str = "") -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

def filter_control_vars(
    df: pd.DataFrame,
    control_vars: Optional[Sequence[str]] = None,
    warn_threshold: float = MISSING_DATA_THRESHOLD,
    verbose: bool = True,
) -> List[str]:
    """
    Filter control variables to those available in DataFrame.

    Args:
        df: DataFrame to check for variables
        control_vars: List of control variables.
            Default: DEFAULT_CONTROL_VARS
        warn_threshold: If fraction missing > threshold, print warning.
            Default: MISSING_DATA_THRESHOLD
        verbose: If True, print warnings about missing variables

    Returns:
        List of available control variables

    Example:
        >>> df = pd.DataFrame({'sex': [1, 2], 'age': [50, 51]})
        >>> available = filter_control_vars(df, ['sex', 'education'])
        >>> print(available)
        ['sex']
    """
    if control_vars is None:
        control_vars = DEFAULT_CONTROL_VARS

    available = [var for var in control_vars if var in df.columns]
    missing = [var for var in control_vars if var not in df.columns]

    if verbose:
        if len(available) == 0:
            print("  Warning: No available control variables found in dataframe.")
            print(f"    Available columns: {list(df.columns)[:20]}...")
            print(f"    Looking for: {control_vars}")
        elif len(missing) / len(control_vars) > warn_threshold:
            print(
                f"  Warning: More than {warn_threshold * 100:.0f}% of control variables are missing. "
                f"Available: {len(available)}/{len(control_vars)}"
            )

    return available

__all__ = ["build_cli_parser", "filter_control_vars", "log_script_end", "log_script_start"]
