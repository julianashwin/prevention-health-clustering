"""Build the processed UKHLS panel from the raw tab extract.

Thin driver over the package's data layer, chaining the two legacy stages the
predecessor exposed as `load-waves` and `preprocess`. Requires the licensed
UKDS extract symlinked at data/raw/UKDA-6614-tab.

Output: data/processed/ukhls_indresp_processed.csv
(~1 GB; 601,304 rows, 99,224 people — the frozen-roster validation targets).

    PYTHONPATH=src .venv/bin/python data_cleaning/01_build_panel.py
"""

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.data.io import load_ukhls_panel_waves
from prevention_health_clustering.data.preprocess import (
    get_default_columns,
    preprocess_ukhls_panel_data_for_clustering,
)


def main() -> int:
    ensure_runtime_directories()
    load_ukhls_panel_waves(panel_type="indresp")
    preprocess_ukhls_panel_data_for_clustering(columns=get_default_columns())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
