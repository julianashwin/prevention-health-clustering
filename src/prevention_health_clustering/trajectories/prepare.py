"""Prepare canonical wide trajectory analysis files from a processed panel.

The preparation layer is the single place where long person-wave UKHLS records
are converted into one-row-per-person trajectory data. The output is unlabeled:
it contains age-specific outcome columns, wave-entry metadata, and available
baseline controls, but no cluster assignments. Clustering, internal diagnostics,
and outcome validation should all consume this prepared file rather than relying
on side effects from a prior clustering run.

Usage:
    uv run ukhls-health-clusters prepare-trajectories --outcomes sf12pcs_dv
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd

from prevention_health_clustering._utils import filter_control_vars
from prevention_health_clustering.cli_utils import (
    build_cli_parser,
    log_script_end,
    log_script_start,
)
from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.cli_utils import default_prepared_trajectory_filepath
from prevention_health_clustering.config import (
    DEFAULT_HEALTH_METRICS,
    MAX_PERSON_OBSERVATIONS,
    PROCESSED_DATA_DIR,)
from prevention_health_clustering.data.preprocess import load_dataframe

RESERVED_PREPARED_TRAJECTORY_COLUMNS = frozenset(
    {
        "cluster_label",
        "runner_up_cluster_label",
        "membership_assigned_distance",
        "membership_runner_up_distance",
        "membership_distance_margin",
        "membership_distance_ratio",
        "membership_relative_margin",
        "membership_n_features_used",
    }
)


def _is_reserved_prepared_trajectory_column(column: str) -> bool:
    return column in RESERVED_PREPARED_TRAJECTORY_COLUMNS or column.endswith(
        "_cluster_label"
    )


def long_to_wide_from_df(
    panel_df: pd.DataFrame,
    outcomes: list[str],
    pidp_col: str = "pidp",
    age_col: str = "age",
    ages: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Pivot long-format panel data into one row per person and age-specific columns."""
    print(f"\nConverting long-format panel data {panel_df.shape} to trajectories...")
    print("  (One row per person, columns for outcomes at each age)")

    required_columns = {pidp_col, age_col, *outcomes}
    missing_columns = sorted(required_columns - set(panel_df.columns))
    if missing_columns:
        raise ValueError(
            "Input panel is missing required trajectory columns: "
            + ", ".join(missing_columns)
        )

    if ages is not None:
        age_mask = cast(pd.Series, panel_df[age_col].isin(ages))
        panel_df = panel_df.loc[age_mask].copy()

    long_df_all: pd.DataFrame = (
        panel_df.loc[:, [pidp_col, age_col, *outcomes]]
        .copy()
        .sort_values(by=[pidp_col, age_col])
    )

    # Pivot all outcomes at once: index=pidp, columns=(outcome, age)
    wide_multi = pd.DataFrame(
        long_df_all.pivot_table(
            index=pidp_col,
            columns=age_col,
            values=outcomes,
        )
    )

    # Flatten MultiIndex columns (metric, age) -> "{metric}_{age}"
    wide_columns = cast(pd.MultiIndex, wide_multi.columns)
    wide_multi.columns = [
        f"{metric}_{int(age)}" for metric, age in wide_columns.to_list()
    ]

    # Manually reset pidp to ensure one row per person with any non-missing outcome data
    valid_pidp_mask = long_df_all[outcomes].notna().any(axis=1)
    original_pidps = np.sort(long_df_all.loc[valid_pidp_mask, pidp_col].unique())
    if wide_multi.shape[0] != len(original_pidps):
        raise ValueError(
            f"Wide-format trajectories have {wide_multi.shape[0]} rows but "
            f"there are {len(original_pidps)} unique pidps in the input data. "
            "This indicates an unexpected duplication or loss during pivot."
        )

    # Drop the existing index and reattach pidp as a clean column
    wide_df = pd.DataFrame(wide_multi.reset_index(drop=True))
    wide_df.insert(0, "pidp", original_pidps)
    assert (
        wide_df["pidp"].nunique() == wide_df.shape[0]
    ), "pidp column is not unique in wide_df"
    print(
        f"  Successfully created wide-format trajectories: {wide_df.shape[0]:,} persons × {wide_df.shape[1] - 1} health-age columns"
    )

    return wide_df


def _validate_unique_person_age(
    panel_df: pd.DataFrame,
    *,
    pidp_col: str,
    age_col: str,
) -> None:
    """Ensure the processed panel has at most one record per person-age."""
    duplicates = panel_df.duplicated(subset=[pidp_col, age_col])
    if duplicates.any():
        raise ValueError(
            f"Found {int(duplicates.sum()):,} duplicate {pidp_col}-{age_col} "
            "combinations. Prepare trajectories from a panel with one row per "
            "person-age."
        )


def _person_wave_metadata(
    panel_df: pd.DataFrame,
    *,
    pidp_col: str,
    age_col: str,
    wave_col: str,
) -> pd.DataFrame:
    """Return person-level wave-entry metadata."""
    required_columns = {pidp_col, age_col, wave_col}
    missing_columns = sorted(required_columns - set(panel_df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot prepare wave metadata because the panel is missing: "
            + ", ".join(missing_columns)
        )

    use = panel_df.loc[:, [pidp_col, wave_col, age_col]].copy()
    use[wave_col] = pd.to_numeric(use[wave_col], errors="coerce")
    use[age_col] = pd.to_numeric(use[age_col], errors="coerce")
    use = use.dropna(subset=[pidp_col, wave_col, age_col])
    if use.empty:
        raise ValueError("No non-missing person-wave-age rows are available.")

    use = use.sort_values([pidp_col, wave_col, age_col])
    grouped = use.groupby(pidp_col, sort=False)
    metadata = grouped.agg(
        wave_min=(wave_col, "min"),
        wave_max=(wave_col, "max"),
    ).reset_index()
    first_wave_age = grouped[age_col].first().rename("wave_min_age")
    last_wave_age = grouped[age_col].last().rename("wave_max_age")
    metadata = metadata.merge(
        first_wave_age.reset_index(),
        on=pidp_col,
        how="left",
        validate="one_to_one",
    )
    metadata = metadata.merge(
        last_wave_age.reset_index(),
        on=pidp_col,
        how="left",
        validate="one_to_one",
    )
    metadata = metadata.rename(columns={pidp_col: "pidp"})

    for column in ("wave_min", "wave_max", "wave_min_age", "wave_max_age"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

    metadata["age_at_wave1"] = metadata["wave_min_age"] - metadata["wave_min"] + 1
    metadata["age_at_wave2"] = metadata["age_at_wave1"] + 1
    metadata["min_imp_age"] = metadata["age_at_wave1"]
    metadata["max_imp_age"] = (
        MAX_PERSON_OBSERVATIONS - metadata["wave_max"] + metadata["wave_max_age"]
    )

    integer_columns = [
        "wave_min",
        "wave_max",
        "wave_min_age",
        "wave_max_age",
        "age_at_wave1",
        "age_at_wave2",
        "min_imp_age",
        "max_imp_age",
    ]
    for column in integer_columns:
        metadata[column] = metadata[column].round().astype("Int64")
    return metadata


def _person_control_metadata(
    panel_df: pd.DataFrame,
    *,
    pidp_col: str,
    control_vars: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Return first observed person-level controls for available control columns."""
    if control_vars is None:
        available_control_vars = filter_control_vars(panel_df, verbose=False)
    else:
        available_control_vars = [
            var for var in control_vars if var in panel_df.columns
        ]
    reserved_controls = [
        var
        for var in available_control_vars
        if _is_reserved_prepared_trajectory_column(var)
    ]
    if reserved_controls:
        print(
            "  Skipping label/diagnostic columns in prepared trajectories: "
            + ", ".join(reserved_controls)
        )
        available_control_vars = [
            var
            for var in available_control_vars
            if not _is_reserved_prepared_trajectory_column(var)
        ]

    if not available_control_vars:
        return pd.DataFrame({"pidp": panel_df[pidp_col].drop_duplicates().to_numpy()})

    control_df = (
        panel_df.loc[:, [pidp_col, *available_control_vars]]
        .groupby(pidp_col, sort=False)
        .first()
        .reset_index()
        .rename(columns={pidp_col: "pidp"})
    )
    print(
        "  Added control variables: "
        + ", ".join(str(var) for var in available_control_vars)
    )
    return control_df


def prepare_trajectories_from_df(
    panel_df: pd.DataFrame,
    outcomes: list[str],
    pidp_col: str = "pidp",
    age_col: str = "age",
    wave_col: str = "wave",
    ages: Optional[Sequence[int]] = None,
    include_controls: bool = True,
    control_vars: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the canonical unlabeled wide trajectory analysis dataframe.

    The prepared dataframe contains:
    - `pidp`
    - wave-entry metadata (`wave_min`, `wave_max`, `age_at_wave1`, ...)
    - one column per requested outcome-age pair
    - available person-level controls when `include_controls=True`
    """
    _validate_unique_person_age(panel_df, pidp_col=pidp_col, age_col=age_col)
    wide_df = long_to_wide_from_df(
        panel_df=panel_df,
        outcomes=outcomes,
        pidp_col=pidp_col,
        age_col=age_col,
        ages=ages,
    )
    metadata_df = _person_wave_metadata(
        panel_df,
        pidp_col=pidp_col,
        age_col=age_col,
        wave_col=wave_col,
    )
    prepared_df = wide_df.merge(
        metadata_df, on="pidp", how="left", validate="one_to_one"
    )

    control_columns: list[str] = []
    if include_controls:
        controls_df = _person_control_metadata(
            panel_df,
            pidp_col=pidp_col,
            control_vars=control_vars,
        )
        control_columns = [col for col in controls_df.columns if col != "pidp"]
        prepared_df = prepared_df.merge(
            controls_df,
            on="pidp",
            how="left",
            validate="one_to_one",
        )

    metadata_columns = [
        "wave_min",
        "wave_max",
        "wave_min_age",
        "wave_max_age",
        "age_at_wave1",
        "age_at_wave2",
        "min_imp_age",
        "max_imp_age",
    ]
    trajectory_columns = [
        col
        for col in prepared_df.columns
        if col not in {"pidp", *metadata_columns, *control_columns}
    ]
    ordered_columns = [
        "pidp",
        *[col for col in metadata_columns if col in prepared_df.columns],
        *trajectory_columns,
        *control_columns,
    ]
    prepared_df = prepared_df.loc[:, ordered_columns]
    leaked_columns = sorted(
        col
        for col in prepared_df.columns
        if _is_reserved_prepared_trajectory_column(col)
    )
    if leaked_columns:
        raise ValueError(
            "Prepared trajectory output must be unlabeled, but label/diagnostic "
            "columns were retained: " + ", ".join(leaked_columns)
        )
    print(
        f"  Prepared canonical trajectory frame: {prepared_df.shape[0]:,} persons × "
        f"{prepared_df.shape[1]:,} columns"
    )
    return prepared_df


def long_to_wide_from_file(
    input_filepath: Path,
    outcomes: list[str],
    pidp_col: str = "pidp",
    age_col: str = "age",
    ages: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Load a processed panel file and convert it to wide-format trajectories."""
    if not input_filepath.exists():
        raise FileNotFoundError(f"Input file not found: {input_filepath}")
    panel_df = load_dataframe(input_filepath)

    target_ages = (
        list(ages) if ages is not None else sorted(panel_df[age_col].dropna().unique())
    )
    return long_to_wide_from_df(
        panel_df=panel_df,
        outcomes=outcomes,
        pidp_col=pidp_col,
        age_col=age_col,
        ages=target_ages,
    )


def prepare_trajectories_from_file(
    input_filepath: Path,
    outcomes: list[str],
    pidp_col: str = "pidp",
    age_col: str = "age",
    wave_col: str = "wave",
    ages: Optional[Sequence[int]] = None,
    include_controls: bool = True,
    control_vars: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a processed panel and prepare the canonical wide trajectory frame."""
    if not input_filepath.exists():
        raise FileNotFoundError(f"Input file not found: {input_filepath}")
    panel_df = load_dataframe(input_filepath)
    target_ages = (
        list(ages) if ages is not None else sorted(panel_df[age_col].dropna().unique())
    )
    return prepare_trajectories_from_df(
        panel_df=panel_df,
        outcomes=outcomes,
        pidp_col=pidp_col,
        age_col=age_col,
        wave_col=wave_col,
        ages=target_ages,
        include_controls=include_controls,
        control_vars=control_vars,
    )


def main() -> None:
    """Run the standalone trajectory-preparation CLI."""
    parser = build_cli_parser(
        "Prepare canonical unlabeled wide trajectories from a processed panel."
    )
    parser.add_argument(
        "--input-filepath",
        type=Path,
        default=Path(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv"),
        help="Input long-format panel data filepath.",
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        nargs="+",
        default=list(DEFAULT_HEALTH_METRICS),
        help=f"List of outcomes to pivot to wide format. Default: {list(DEFAULT_HEALTH_METRICS)}.",
    )
    parser.add_argument("--pidp-col", default="pidp", help="Person identifier column.")
    parser.add_argument("--age-col", default="age", help="Age column.")
    parser.add_argument("--wave-col", default="wave", help="Survey wave column.")
    parser.add_argument(
        "--ages",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit ages to include in the wide trajectory output.",
    )
    parser.add_argument(
        "--control-vars",
        nargs="*",
        default=None,
        help="Optional person-level controls to carry through to the wide file.",
    )
    parser.add_argument(
        "--no-control-vars",
        action="store_true",
        help="Do not append available person-level control columns.",
    )
    parser.add_argument(
        "--output-filepath",
        type=Path,
        default=None,
        help="Optional output filepath for wide CSV.",
    )
    args = parser.parse_args()
    ensure_runtime_directories()

    wide_df = prepare_trajectories_from_file(
        input_filepath=args.input_filepath,
        outcomes=args.outcomes,
        pidp_col=args.pidp_col,
        age_col=args.age_col,
        wave_col=args.wave_col,
        ages=args.ages,
        include_controls=not args.no_control_vars,
        control_vars=args.control_vars,
    )
    output_path = args.output_filepath
    if output_path is None:
        output_path = default_prepared_trajectory_filepath(
            PROCESSED_DATA_DIR,
            args.outcomes,
        )

    log_script_start(
        Path(__file__).name,
        "Prepare canonical unlabeled wide trajectories from the processed panel.",
        inputs={"processed panel": args.input_filepath},
        outputs={"prepared trajectory CSV": output_path},
        config={
            "outcomes": args.outcomes,
            "ages": "all" if args.ages is None else args.ages,
            "control variables": "disabled" if args.no_control_vars else "enabled",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_path, index=False)
    print(f"Wrote {wide_df.shape[0]:,} rows to {output_path}")
    log_script_end(
        Path(__file__).name, outputs={"prepared trajectory CSV": output_path}
    )


if __name__ == "__main__":
    main()
