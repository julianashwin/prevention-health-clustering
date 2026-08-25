"""Whole-person fold construction, byte-compatible with the frozen manifest.

The predecessor froze one five-fold assignment of the 50,194-person roster
(manifest ID ``1797b5c5a937d7f5671d``, seed 20260723) and reused it for every
validation exercise, so results are paired across models. The rule: sort the
unique integer person ids, permute with ``numpy.random.default_rng(seed)``,
assign ``fold = position % n_folds + 1``, and derive the manifest ID as the
first 20 hex characters of the SHA-256 of a canonical JSON identity.

This module reimplements the rule exactly; ``tests/test_folds.py`` checks that
the frozen manifest ID is reproduced from the rebuilt contract, which validates
both the rule and the roster in one shot.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

FOLD_SCHEMA_VERSION = "ukhls-person-folds/v1"
DEFAULT_FOLD_SEED = 20260723
DEFAULT_N_FOLDS = 5

# The predecessor's frozen manifest, for validation.
FROZEN_MANIFEST_ID = "1797b5c5a937d7f5671d"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_person_folds(
    pids: Iterable[int],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_FOLD_SEED,
) -> pd.DataFrame:
    """One row per person: pidp, fold (1-based), and the manifest identity."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    series = pd.Series(list(pids), dtype="object")
    if series.empty or series.isna().any():
        raise ValueError("Person identifiers must be non-empty and non-missing.")
    numeric = pd.to_numeric(series, errors="raise")
    if not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError("Person identifiers must be integer-valued.")
    unique = np.sort(numeric.astype(np.int64).unique())
    if len(unique) < n_folds:
        raise ValueError(f"Need at least {n_folds} unique people.")

    permuted = np.random.default_rng(seed).permutation(unique)
    assignments = pd.DataFrame(
        {"pidp": permuted, "fold": np.arange(len(permuted), dtype=int) % n_folds + 1}
    ).sort_values("pidp", kind="stable", ignore_index=True)

    identity = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "n_folds": int(n_folds),
        "seed": int(seed),
        "assignments": assignments.to_dict(orient="records"),
    }
    manifest_id = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()[:20]
    assignments["fold_manifest_id"] = manifest_id
    assignments["fold_seed"] = int(seed)
    assignments["n_folds"] = int(n_folds)
    return assignments


def fold_split(
    folds: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """(train_pids, heldout_pids) for one fold; whole people, disjoint."""
    held = folds.loc[folds["fold"] == fold, "pidp"].to_numpy()
    train = folds.loc[folds["fold"] != fold, "pidp"].to_numpy()
    if len(held) == 0 or len(train) == 0:
        raise ValueError(f"Fold {fold} yields an empty split.")
    return train, held


__all__ = [
    "DEFAULT_FOLD_SEED",
    "DEFAULT_N_FOLDS",
    "FROZEN_MANIFEST_ID",
    "build_person_folds",
    "fold_split",
]
