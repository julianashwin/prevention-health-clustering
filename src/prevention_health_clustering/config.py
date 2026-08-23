"""Project paths and frozen analysis constants.

The repository root is resolved from an explicit environment variable when set,
otherwise by walking up from this file until a directory containing
``pyproject.toml`` is found. This avoids the ``__file__.parents[N]`` idiom used
by the predecessor project, which silently resolved into ``site-packages`` under
a non-editable install and created data directories inside the interpreter tree.
"""

from __future__ import annotations

import os
from pathlib import Path


def _discover_root() -> Path:
    override = os.environ.get("PHC_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"PHC_PROJECT_ROOT is not a directory: {root}")
        return root
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate the project root. Set PHC_PROJECT_ROOT explicitly."
    )


ROOT_DIR = _discover_root()

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Licensed UKHLS tab extract. Symlink the UKDS study directory into data/raw/.
UKHLS_PANEL_DIR = RAW_DATA_DIR / "UKDA-6614-tab" / "tab" / "ukhls"

# ---------------------------------------------------------------------------
# Frozen analysis constants
# ---------------------------------------------------------------------------

# UKHLS release waves a..o.
MAX_PERSON_OBSERVATIONS = 15

DEFAULT_HEALTH_METRIC = "sf12pcs_dv"
DEFAULT_SECONDARY_METRIC = "sf12mcs_dv"
DEFAULT_HEALTH_METRICS: tuple[str, ...] = (DEFAULT_HEALTH_METRIC,)

# The frozen lifecycle window. Ages are inclusive at both ends in the public
# interface; internal slicing uses a half-open [start, stop) convention.
DEFAULT_LIFECYCLE_MIN_AGE = 20
DEFAULT_LIFECYCLE_MAX_AGE = 89
DEFAULT_AGE_CENTER = 55
DEFAULT_AGE_SCALE = 10.0
DEFAULT_MIN_OBS_PER_PERSON = 3
DEFAULT_K = 3

MIN_AGE = 14
MAX_AGE = 104

# Share of missing values in a control column above which a warning is emitted.
MISSING_DATA_THRESHOLD = 0.5

RANDOM_STATE = 42
FOLD_COUNT = 5

# ---------------------------------------------------------------------------
# UKHLS reserved missing codes
#
# Corrected against the Understanding Society release dictionary. The
# predecessor project carried a scrambled label set in which -9 was labelled
# "Don't know", -8 "Refused", -7 "Not applicable", -2 "Item not applicable" and
# -1 "Item not answered". Those labels fed provenance manifests. The numeric
# set was and remains correct; only the labels were wrong.
# ---------------------------------------------------------------------------

UKHLS_MISSING_CODES: dict[int, str] = {
    -1: "Don't know",
    -2: "Refusal",
    -7: "Proxy respondent",
    -8: "Inapplicable",
    -9: "Missing",
    -10: "Not applicable (derived)",
}

UKHLS_MISSING_VALUES = tuple(sorted(UKHLS_MISSING_CODES))

__all__ = [
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "DEFAULT_AGE_CENTER",
    "DEFAULT_AGE_SCALE",
    "DEFAULT_HEALTH_METRIC",
    "DEFAULT_HEALTH_METRICS",
    "DEFAULT_K",
    "DEFAULT_LIFECYCLE_MAX_AGE",
    "DEFAULT_LIFECYCLE_MIN_AGE",
    "DEFAULT_MIN_OBS_PER_PERSON",
    "DEFAULT_SECONDARY_METRIC",
    "FOLD_COUNT",
    "INTERIM_DATA_DIR",
    "MAX_AGE",
    "MAX_PERSON_OBSERVATIONS",
    "MISSING_DATA_THRESHOLD",
    "MIN_AGE",
    "PROCESSED_DATA_DIR",
    "RANDOM_STATE",
    "RAW_DATA_DIR",
    "ROOT_DIR",
    "UKHLS_MISSING_CODES",
    "UKHLS_MISSING_VALUES",
    "UKHLS_PANEL_DIR",
]
