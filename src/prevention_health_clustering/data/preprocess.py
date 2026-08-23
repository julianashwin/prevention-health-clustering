"""
Preprocess UKHLS panel data for health trajectory clustering.

This module provides comprehensive data cleaning and preprocessing functionality
for UKHLS (Understanding Society) panel data, preparing it for health trajectory
clustering analysis. It handles:

- Missing value imputation and outlier detection
- Birth year inconsistency resolution
- Survey date (istrtdaty) consistency checking and smoothing
- Duplicate observation handling
- Demographic variable aggregation

The preprocessing pipeline ensures data quality and consistency required for
longitudinal health trajectory analysis.

Minimal required columns:
    ['pidp', 'wave', 'birthy', 'sf12pcs_dv']

Example:
    Preprocess UKHLS data for clustering:
    ```python
    from pathlib import Path
    from prevention_health_clustering.data.preprocess import (
        preprocess_ukhls_panel_data_for_clustering,
    )

    preprocess_ukhls_panel_data_for_clustering(
        input_filepath=Path("data/interim/ukhls_indresp_combined.parquet"),
        output_filepath=Path("data/processed/ukhls_indresp_processed.csv"),
        columns=["pidp", "wave", "birthy", "sf12pcs_dv", "sf12mcs_dv"],
    )
    ```
"""

# Standard library imports
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Sequence, cast

# Third-party imports
import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid

from prevention_health_clustering.cli_utils import (
    build_cli_parser,
    log_script_end,
    log_script_start,
)

# Local imports
from prevention_health_clustering.data.io import (
    UKHLS_WAVE_DATES,
    XWAVEDAT_PER_WAVE_COLUMNS,
    XWAVEID_PER_WAVE_COLUMNS,
    XWAVE_PERSON_MORTALITY_COLUMNS,
)
from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    UKHLS_MISSING_CODES,)
from prevention_health_clustering.data.io import (
    UKHLS_WAVE_DATES,
    XWAVE_PERSON_MORTALITY_COLUMNS,
    XWAVEDAT_PER_WAVE_COLUMNS,
    XWAVEID_PER_WAVE_COLUMNS,
)

UKHLS_MISSING_VALUES = sorted(UKHLS_MISSING_CODES.keys())
MORTALITY_RETAINED_COLUMNS = (
    XWAVE_PERSON_MORTALITY_COLUMNS
    + tuple(column for column in XWAVEID_PER_WAVE_COLUMNS if column != "month")
    + XWAVEDAT_PER_WAVE_COLUMNS
)

# Survey design variables required for design-based inference. Retaining them
# alongside the discovered weight columns means design-weighted analyses can
# use the processed panel directly instead of re-joining the interim panel.
# The weight/design contract itself is frozen in
# `prevention_health_clustering/validation/assets/survey-estimand-v1.json`.
SURVEY_DESIGN_SOURCE_COLUMNS = (
    "psu",
    "strata",
    "sampst",
)

# `dcsedw_dv` uses a combined wave coding where:
# - 2..18 correspond to BHPS Waves 2..18
# - 19..(18 + UKHLS wave number) correspond to UKHLS Waves 1..N
#   For example, 20 => UKHLS Wave 2.
# We map the code to a representative calendar year using the BHPS survey year
# and the official UKHLS wave midpoint year metadata.
DCSDEDW_CODE_TO_YEAR = {
    **{wave_code: 1990 + wave_code for wave_code in range(2, 19)},
    **{
        18 + cast(int, metadata["wave_number"]): cast(int, metadata["mid_year"])
        for metadata in UKHLS_WAVE_DATES.values()
    },
}
SERIOUS_CONDITION_SOURCE_COLUMNS = (
    "hcond4",  # Coronary heart disease
    "hcond6",  # Heart attack / myocardial infarction
    "hcond7",  # Stroke
    "hcond13",  # Cancer or malignancy
    "hcond14",  # Diabetes
)
CARE_HELP_CODE_SOURCE_COLUMNS = tuple(f"helpcode{i}" for i in range(1, 12))
DISABILITY_DIFFICULTY_COLUMNS = tuple(f"disdif{i}" for i in range(1, 13)) + (
    "disdif96",
)
DISABILITY_SEVERITY_COLUMNS = tuple(f"dissev{i}" for i in range(1, 13))
ADL_ABILITY_COLUMNS = tuple(f"adl{suffix}" for suffix in "abcdefghijklmn")
ADL_EASE_COLUMNS = tuple(f"adl{suffix}d" for suffix in "abcdefghijklmn")
ADL_SOURCE_COLUMNS = ADL_ABILITY_COLUMNS + ADL_EASE_COLUMNS
GENERAL_HEALTH_SOURCE_COLUMNS = (
    "sf1",
    "scsf1",
)
SF12_ITEM_SOURCE_COLUMNS = (
    "sf2a",
    "sf2b",
    "sf3a",
    "sf3b",
    "sf4a",
    "sf4b",
    "sf5",
    "sf6a",
    "sf6b",
    "sf6c",
    "sf7",
    "scsf2a",
    "scsf2b",
    "scsf3a",
    "scsf3b",
    "scsf4a",
    "scsf4b",
    "scsf5",
    "scsf6a",
    "scsf6b",
    "scsf6c",
    "scsf7",
)
SF12_ROUTING_SOURCE_COLUMNS = (
    "ivfio",
    "scflag_dv",
)
GHQ_ITEM_SOURCE_COLUMNS = tuple(f"scghq{suffix}" for suffix in "abcdefghijkl")
WELLBEING_SOURCE_COLUMNS = (
    "swemwbs_dv",
    "sclfsat1",
    "sclfsat2",
    "sclfsat3",
    "sclfsat7",
    "sclfsato",
    *tuple(f"scwemwb{suffix}" for suffix in "abcdefg"),
)
SLEEP_SOURCE_COLUMNS = (
    "schrs_slph",
    "schrs_slpm",
    "sctslp_30m",
    "sctslp_wak",
    "sctslp_cgh",
    "scmed_slp",
    "med_slp",
    "sctsta_awk",
    "scslp_qual",
)
WALKING_ACTIVITY_SOURCE_COLUMNS = (
    "wkphys",
    "wlk10m",
    "daywlk",
    "wlk30min",
    "walkpace",
)
SMOKING_SOURCE_COLUMNS = (
    "smever",
    "smnow",
    "smoker",
    "ncigs",
    "smcigs",
    "smncigs",
    "aglquit",
    "smagbg",
)
ALCOHOL_SOURCE_COLUMNS = (
    "evralc",
    *tuple(f"auditc{i}" for i in range(1, 6)),
    *tuple(f"auditc{i}fh{j}" for j in range(1, 10) for i in range(1, 6)),
)
DIET_SOURCE_COLUMNS = (
    "food1",
    "food2",
    "food3",
    "food4",
    "food5",
    "food6",
    "food7",
    "usdairy",
    *tuple(f"usdairy{i}" for i in range(1, 7)),
    "usbread",
    *tuple(f"usbread{i}" for i in range(1, 8)),
    "wkfruit",
    "wkvege",
    "fruvege",
)
LONG_COVID_CHILD_SLOTS = tuple(range(1, 14))
LONG_COVID_SYMPTOM_SUFFIXES = (
    1,
    *range(3, 11),
    *range(12, 26),
    96,
    97,
)
LONG_COVID_BASE_COLUMNS = (
    "testposcov",
    *tuple(f"testposcovkid_{slot}" for slot in LONG_COVID_CHILD_SLOTS),
    "longcova",
    *tuple(f"longcovakid_{slot}" for slot in LONG_COVID_CHILD_SLOTS),
    "longcovever",
    "longcovnow",
    "lgcvda",
    "lgcvdanow",
    "lgcvwk",
    "lgcvwknow",
)
LONG_COVID_SOURCE_COLUMNS = (
    *LONG_COVID_BASE_COLUMNS,
    *tuple(f"lgcvwky{i}" for i in (*range(1, 7), 96, 97)),
    *tuple(f"lgcvwkynow{i}" for i in (*range(1, 7), 96, 97)),
    *tuple(f"lgcvsymp{i}" for i in LONG_COVID_SYMPTOM_SUFFIXES),
    *tuple(
        f"lgcvsympkid{i}_{slot}"
        for slot in LONG_COVID_CHILD_SLOTS
        for i in LONG_COVID_SYMPTOM_SUFFIXES
    ),
    *tuple(f"lgcvsympnow{i}" for i in LONG_COVID_SYMPTOM_SUFFIXES),
    "lgcvsympoth_code",
    "lgcvsympnowo_code",
)
CONDITION_CODE_SUFFIXES = tuple(dict.fromkeys((*range(1, 98), *range(66, 90))))
CONDITION_SOURCE_COLUMNS = (
    *tuple(f"hcond{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcond{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hconds{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hconds{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hconda{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hconda{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondn{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondns{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondns{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondp{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondp{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondcode{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondncode{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"prevhcondno{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcond_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondcode_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hconda_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondncode_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    "hconde6",
    "hconde7",
    "hconde96",
    "hcondea6",
    "hcondea7",
)
CONDITION_COUNT_SOURCE_COLUMNS = (
    *tuple(f"hcond{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcond{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondn{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondp{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondp{i:02d}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondcode{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondncode{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"prevhcondno{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcond_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondcode_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hcondncode_cov{i}" for i in CONDITION_CODE_SUFFIXES),
    "hconde6",
    "hconde7",
    "hconde96",
)
HOSPITAL_CONDITION_COLUMNS = (
    *tuple(f"hospc{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hospdc{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hospcp{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hospdcp{i}" for i in CONDITION_CODE_SUFFIXES),
    "hosp",
    "hospd",
    "hospch",
    "hospnhs",
)
HOSPITAL_CONDITION_COUNT_COLUMNS = (
    *tuple(f"hospc{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hospcp{i}" for i in CONDITION_CODE_SUFFIXES),
)
HOSPITAL_DAY_SOURCE_COLUMNS = (
    *tuple(f"hospdc{i}" for i in CONDITION_CODE_SUFFIXES),
    *tuple(f"hospdcp{i}" for i in CONDITION_CODE_SUFFIXES),
    "hospd",
)
PREDICTIVE_ANALYSIS_TOP20_EXTRA_COLUMNS = (
    "hcond1",
    "hcond16",
    "disdif2",
    "disdif8",
    "disdif12",
    "benpen1",
    "benunemp1",
)


def derive_deceased_binary_from_dcsedfl(
    df: pd.DataFrame,
    source_col: str = "dcsedfl_dv",
    output_col: str = "deceased_dcsedfl",
) -> None:
    """
    Derive a binary deceased indicator from `dcsedfl_dv`.

    The retained cross-wave death flag uses:
    - 1: deceased in UKHLS
    - 2: deceased in BHPS
    - 3: not deceased

    The derived binary is:
    - 1 for `dcsedfl_dv` in {1, 2}
    - 0 for `dcsedfl_dv == 3`
    - missing otherwise
    """
    if source_col not in df.columns:
        return

    dcsedfl = pd.to_numeric(df[source_col], errors="coerce")
    df[output_col] = pd.to_numeric(
        dcsedfl.map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0}),
        errors="coerce",
    )


def derive_binary_indicator_from_coded_source(
    df: pd.DataFrame,
    *,
    source_col: str,
    output_col: str,
    positive_values: Sequence[float],
    negative_values: Sequence[float],
) -> None:
    """Derive a binary indicator from a coded survey column."""
    if source_col not in df.columns:
        return

    source = pd.to_numeric(df[source_col], errors="coerce")
    mapping = {float(value): 1.0 for value in positive_values}
    mapping.update({float(value): 0.0 for value in negative_values})
    df[output_col] = pd.to_numeric(source.map(mapping), errors="coerce")


def derive_serious_condition_count(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str] = SERIOUS_CONDITION_SOURCE_COLUMNS,
    output_col: str = "serious_condition_count",
    observed_col: str = "serious_condition_count_observed",
) -> None:
    """Derive a count of major diagnosed conditions from `hcond*` indicators."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available_cols
        }
    )
    # The retained `hcond*` indicators use 0 = not mentioned, 1 = yes mentioned.
    condition_flags = numeric_df.where(numeric_df.isin([0.0, 1.0]))
    df[observed_col] = pd.Series(
        condition_flags.notna().any(axis=1).astype(float),
        index=df.index,
        dtype=float,
    )
    df[output_col] = pd.to_numeric(
        condition_flags.fillna(0.0).sum(axis=1),
        errors="coerce",
    )


def derive_binary_source_count(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str],
    output_col: str,
    observed_col: str,
) -> None:
    """Count positive 0/1 source indicators and retain an observed-source flag."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available_cols
        }
    )
    flags = numeric_df.where(numeric_df.isin([0.0, 1.0]))
    observed = flags.notna().any(axis=1)
    df[observed_col] = pd.Series(observed.astype(float), index=df.index, dtype=float)
    df[output_col] = pd.to_numeric(
        flags.fillna(0.0).sum(axis=1).where(observed),
        errors="coerce",
    )


def derive_coded_positive_count(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str],
    output_col: str,
    observed_col: str,
    positive_values: Sequence[float] = (1.0,),
    negative_values: Sequence[float] = (0.0, 2.0),
) -> None:
    """Count positive coded source values and retain an observed-source flag."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available_cols
        }
    )
    valid_values = set(positive_values) | set(negative_values)
    valid = numeric_df.where(numeric_df.isin(valid_values))
    observed = valid.notna().any(axis=1)
    positives = valid.isin(positive_values)
    df[observed_col] = pd.Series(observed.astype(float), index=df.index, dtype=float)
    df[output_col] = pd.to_numeric(
        positives.sum(axis=1).where(observed),
        errors="coerce",
    )


def derive_numeric_source_sum(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str],
    output_col: str,
    observed_col: str,
) -> None:
    """Sum non-negative numeric source values and retain an observed-source flag."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available_cols
        }
    )
    valid = numeric_df.where(numeric_df >= 0)
    observed = valid.notna().any(axis=1)
    df[observed_col] = pd.Series(observed.astype(float), index=df.index, dtype=float)
    df[output_col] = pd.to_numeric(
        valid.fillna(0.0).sum(axis=1).where(observed),
        errors="coerce",
    )


def derive_adl_limitation_count(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str] = ADL_ABILITY_COLUMNS,
    output_col: str = "adl_limitation_count",
    observed_col: str = "adl_limitation_count_observed",
) -> None:
    """Count ADL/IADL activities where the respondent reports limitation."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available_cols
        }
    )
    valid = numeric_df.where(numeric_df.isin([1.0, 2.0, 3.0]))
    observed = valid.notna().any(axis=1)
    # UKHLS social-care ADL ability items use 1 = manages alone and
    # 2/3 = limitation/help needed.
    limitations = valid.isin([2.0, 3.0])
    df[observed_col] = pd.Series(observed.astype(float), index=df.index, dtype=float)
    df[output_col] = pd.to_numeric(
        limitations.sum(axis=1).where(observed),
        errors="coerce",
    )


def derive_coalesced_column(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str],
    output_col: str,
) -> None:
    """Coalesce source columns left-to-right after preprocessing sentinels to NaN."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    result = pd.to_numeric(df[available_cols[0]], errors="coerce")
    for column in available_cols[1:]:
        candidate = pd.to_numeric(df[column], errors="coerce")
        result = result.where(result.notna(), candidate)
    df[output_col] = pd.to_numeric(result, errors="coerce")


def derive_nonmissing_column_count(
    df: pd.DataFrame,
    *,
    source_cols: Sequence[str],
    output_col: str,
) -> None:
    """Count non-missing values across a family of source columns."""
    available_cols = [column for column in source_cols if column in df.columns]
    if not available_cols:
        return

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available_cols
        }
    )
    df[output_col] = pd.to_numeric(numeric_df.notna().sum(axis=1), errors="coerce")


def derive_lives_alone_flag(
    df: pd.DataFrame,
    *,
    source_col: str = "hhsize",
    output_col: str = "lives_alone_flag",
) -> None:
    """Derive a binary living-alone indicator from household size."""
    if source_col not in df.columns:
        return

    household_size = pd.to_numeric(df[source_col], errors="coerce")
    df[output_col] = pd.Series(
        np.where(household_size.isna(), np.nan, (household_size == 1).astype(float)),
        index=df.index,
        dtype=float,
    )


def derive_deceased_year_from_dcsedw(
    df: pd.DataFrame,
    source_col: str = "dcsedw_dv",
    output_col: str = "deceased_year_dcsedw",
) -> None:
    """
    Derive a calendar year from the `dcsedw_dv` death-wave code.

    The official Understanding Society documentation for `dcsedw_dv` defines
    BHPS death-wave codes from 2 to 18 and then continues the numbering into
    UKHLS, so code 20 corresponds to UKHLS Wave 2.

    Values less than or equal to zero are treated as missing in the derived
    year variable. Unmapped positive codes are also left as missing.
    """
    if source_col not in df.columns:
        return

    dcsedw = pd.to_numeric(df[source_col], errors="coerce")
    positive_codes = dcsedw.where(dcsedw > 0)
    df[output_col] = pd.to_numeric(
        positive_codes.map(DCSDEDW_CODE_TO_YEAR),
        errors="coerce",
    )


def load_dataframe(
    input_filepath: Path, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Load a DataFrame from various file formats with column selection.

    Supports multiple file formats commonly used for panel data:
    - Parquet (recommended for large datasets)
    - CSV
    - Feather
    - HDF5
    - Pickle

    Args:
        input_filepath: Path to the input file. Format is inferred from extension.
        columns: Optional list of column names to load. If None, loads all columns.
            This is useful for memory efficiency when only specific columns are needed.

    Returns:
        pd.DataFrame: Loaded DataFrame with specified columns (or all columns if None)

    Raises:
        ValueError: If file format is unsupported or if there's an error loading
            specified columns from a parquet file
        FileNotFoundError: If the input file doesn't exist

    Note:
        For parquet files, column selection is done at read time for efficiency.
        For CSV files, `low_memory=False` is used for better type inference.

    Example:
        >>> df = load_dataframe(Path("data.parquet"), columns=['pidp', 'wave', 'age'])
        >>> df = load_dataframe(Path("data.csv"))  # Load all columns
    """
    if input_filepath.suffix == ".parquet":
        try:
            return pd.read_parquet(input_filepath, columns=columns)
        except ArrowInvalid:
            raise ValueError("Error loading specified columns from parquet file")
    elif input_filepath.suffix == ".csv":
        return pd.read_csv(input_filepath, usecols=columns, low_memory=False)
    elif input_filepath.suffix == ".feather":
        return pd.read_feather(input_filepath, columns=columns)
    elif input_filepath.suffix == ".hdf5":
        result = pd.read_hdf(input_filepath, columns=columns)
        if isinstance(result, pd.DataFrame):
            return result
        raise ValueError("HDF5 file did not return a DataFrame")
    elif input_filepath.suffix == ".pickle":
        # Guessing compression from file extension
        result = pd.read_pickle(input_filepath, compression="infer")
        if isinstance(result, pd.DataFrame):
            return result
        raise ValueError("Pickle file did not return a DataFrame")
    else:
        raise ValueError(f"Unsupported format: {input_filepath.suffix}")


def _available_columns(input_filepath: Path) -> list[str]:
    """Return discoverable column names for supported input formats."""
    if input_filepath.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise ValueError("pyarrow is required to inspect parquet columns") from e
        pf = pq.ParquetFile(input_filepath)
        return list(pf.schema.names)
    if input_filepath.suffix == ".csv":
        return list(pd.read_csv(input_filepath, nrows=0).columns)
    if input_filepath.suffix in {".feather"}:
        try:
            import pyarrow.feather as feather
        except Exception as e:
            raise ValueError("pyarrow is required to inspect feather columns") from e
        return list(feather.read_table(input_filepath).column_names)
    if input_filepath.suffix in {".hdf5"}:
        # Avoid heavy discovery for HDF5 in preprocessing; treat as unknown.
        return []
    if input_filepath.suffix in {".pickle"}:
        return []
    return []


def _discover_weight_columns(columns: Iterable[str]) -> list[str]:
    """Identify candidate weight columns from a collection of column names."""
    patterns = [
        re.compile(r".*_xw$"),
        re.compile(r".*_lw$"),
        re.compile(r".*weight.*", re.IGNORECASE),
        re.compile(r"^w_"),
    ]
    weight_cols: list[str] = []
    for c in columns:
        if any(p.match(str(c)) for p in patterns):
            weight_cols.append(str(c))
    return sorted(set(weight_cols))


def drop_individuals_with_all_na_column_values(
    df: pd.DataFrame, column: str, verbose: bool = False
) -> int:
    """
    Drop individuals from the DataFrame if all their values for a column are NA.

    This function identifies individuals (by pidp) who have no valid values
    for a specified column across all their observations, and removes all
    observations for those individuals from the DataFrame.

    Args:
        df: DataFrame to modify. Must contain 'pidp' column and the specified column.
            Modified in place.
        column: Column name to check for all-NA values
        verbose: If True, prints detailed information about dropped individuals

    Returns:
        int: Number of unique individuals (pidp values) that were dropped

    Note:
        The function modifies the DataFrame in place. All observations for
        individuals with all-NA values in the specified column are removed.

    Example:
        >>> n_dropped = drop_individuals_with_all_na_column_values(df, 'birthy')
        >>> print(f"Dropped {n_dropped} individuals with no birth year data")
    """
    all_na = df.groupby("pidp")[column].apply(lambda x: x.isna().all())
    pids_to_drop = all_na[all_na].index
    if verbose:
        print(
            f"Dropping {len(pids_to_drop):,} individuals with all NA values in column '{column}'"
        )
        print(f"  Pids to drop: {pids_to_drop}")
    df.drop(df[df["pidp"].isin(pids_to_drop)].index, inplace=True)
    return len(pids_to_drop)


def preprocess_column(
    df: pd.DataFrame,
    column: str,
    known_nans: Optional[Sequence[float]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    outliers_z_threshold: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Preprocess a single column by handling missing values, outliers, and invalid ranges.

    This function performs comprehensive cleaning of a column:
    1. Replaces known sentinel values (e.g., -9, -8, -7) with NaN
    2. Replaces values outside valid range (min_value, max_value) with NaN
    3. Identifies and replaces outliers using z-score thresholding

    Args:
        df: DataFrame to preprocess. Modified in place.
        column: Name of the column to preprocess
        known_nans: List of sentinel values that should be treated as missing.
            Common UKHLS sentinel values: -9 (refused), -8 (don't know), -7 (not applicable),
            -2 (not available), -1 (missing). If None, no sentinel value replacement.
        min_value: Minimum valid value for the column. Values below this are set to NaN.
            If None, no minimum check is performed.
        max_value: Maximum valid value for the column. Values above this are set to NaN.
            If None, no maximum check is performed.
        outliers_z_threshold: Z-score threshold for outlier detection. Values with
            |z-score| >= threshold are replaced with NaN. If None, no outlier detection.
        verbose: If True, prints summary statistics and counts of replaced values

    Returns:
        None: Function modifies DataFrame in place

    Note:
        - All replacements are done in place
        - Statistics are printed for each cleaning step
        - If column doesn't exist, a warning is printed and function returns early

    Example:
        >>> # Clean health score column
        >>> preprocess_column(
        ...     df,
        ...     'sf12pcs_dv',
        ...     known_nans=[-9, -8, -7],
        ...     min_value=0,
        ...     max_value=100,
        ...     outliers_z_threshold=3
        ... )
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in dataframe")
        return

    print(f"\nPreprocessing column '{column}'...")

    # Count and replace known NaN values with NaN
    if known_nans is not None:
        n_known_nans = (df[column].isin(known_nans)).sum()
        df.loc[df[column].isin(known_nans), column] = np.nan
        print(f"Column '{column}': {n_known_nans} known NaN values replaced with NaN")

    # Count and replace values outside known range with NaN
    if min_value is not None:
        n_values_outside_range = (df[column] < min_value).sum()
        df.loc[df[column] < min_value, column] = np.nan
        print(
            f"Column '{column}': {n_values_outside_range} values under {min_value} replaced with NaN"
        )
    if max_value is not None:
        n_values_outside_range = (df[column] > max_value).sum()
        df.loc[df[column] > max_value, column] = np.nan
        print(
            f"Column '{column}': {n_values_outside_range} values over {max_value} replaced with NaN"
        )

    # Count and replace outliers with NaN
    if outliers_z_threshold is not None and outliers_z_threshold > 0:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        z_scores: pd.Series = (
            numeric_series - numeric_series.mean()
        ) / numeric_series.std()
        outlier_mask = cast(
            pd.Series,
            z_scores.abs().ge(float(outliers_z_threshold)).fillna(False),
        )
        n_outliers = int(outlier_mask.to_numpy().sum())
        df.loc[outlier_mask, column] = np.nan
        print(
            f"Column '{column}': {n_outliers} outliers (|z| >= {outliers_z_threshold}) replaced with NaN"
        )

    if verbose:
        print(f"Column '{column}':\n{df[column].describe()}")


def preprocess_birth_years(
    df: pd.DataFrame,
    min_mode_percentage: float = 0.66,
    allowed_birth_year_range: int = 2,
    strategy: str = "mode",
) -> pd.DataFrame:
    """
    Resolve birth year inconsistencies across multiple survey waves.

    This function handles cases where an individual's reported birth year varies
    across waves, which can occur due to data entry errors or reporting inconsistencies.
    It uses a hierarchical approach:

    1. **Initial cleaning**: Replaces known sentinel values with NaN and broadcasts
       consistent birth years to missing values when only one unique value exists.

    2. **Easy cases**: If >66% of observations share the same birth year, use that mode.

    3. **Medium cases**: If birth year range ≤ 2 years (likely data entry error),
       use mode or minimum birth year.

    4. **Fallback**: For remaining cases, apply the specified strategy.

    Args:
        df: DataFrame to process. Must contain 'pidp' and 'birthy' columns.
            Modified in place, but also returned for convenience.
        min_mode_percentage: Minimum percentage of observations that must share
            the same birth year for it to be considered the "mode" (default: 0.66).
            Used to identify easy cases with clear majority.
        allowed_birth_year_range: Maximum allowed range (in years) for birth year
            values to be considered a data entry error (default: 2). If the range
            is within this threshold, the mode or minimum is used.
        strategy: Fallback strategy for handling remaining inconsistencies. Options:
            - 'first_wave': Use birth year from first wave with valid data
            - 'last_wave': Use birth year from last wave with valid data
            - 'median': Use median birth year across all waves
            - 'mode': Use most frequent birth year (default)
            - 'min': Use minimum (earliest) birth year
            - 'max': Use maximum (latest) birth year
            - 'drop': Remove the individual from the dataset

    Returns:
        pd.DataFrame: DataFrame with consistent birth years. Each individual
            now has the same birth year across all waves.

    Raises:
        ValueError: If strategy is not one of the supported options
        AssertionError: If birth years are still inconsistent after processing
            (should never occur if function works correctly)

    Note:
        - Individuals with all-NA birth years are dropped before processing
        - The function ensures that after processing, each individual has exactly
          one birth year value across all waves
        - Processing statistics are printed to stdout

    Example:
        >>> # Use mode as fallback strategy
        >>> df = preprocess_birth_years(df, strategy='mode')

        >>> # Use first wave and allow larger ranges
        >>> df = preprocess_birth_years(
        ...     df,
        ...     min_mode_percentage=0.75,
        ...     allowed_birth_year_range=3,
        ...     strategy='first_wave'
        ... )
    """
    print(
        f"\nHandling birth year inconsistencies using fallback strategy: '{strategy}'"
    )

    # First, handle known NA, min, max, and outliers
    preprocess_column(
        df, "birthy", known_nans=[-9, -8, -7, -2, -1], min_value=1900, max_value=2025
    )
    print(
        f"Have {sum(df['birthy'].isna()):,} (out of {df.shape[0]:,}) NA birth year values to start..."
    )

    # Drop individuals with all NA birth years across all waves (i.e. we have no information about their age)
    n_dropped = drop_individuals_with_all_na_column_values(df, "birthy")
    print(f"Dropped {n_dropped:,} individuals with NA birth year across all waves...")

    birth_year_stats = (
        df.groupby("pidp")["birthy"]
        .agg(
            nunique=lambda x: (
                x.dropna().nunique()
            ),  # Use explicit lambda to avoid potential pandas groupby issues
            first="first",
        )
        .reset_index()
    )

    # Broadcast non-NA value to null birth years if there is a single non-NA value
    # e.g. [1960, 1960, NaN, 1960] -> [1960, 1960, 1960, 1960]
    single_year_values = birth_year_stats[birth_year_stats["nunique"] == 1][
        ["pidp", "first"]
    ]
    print(
        f"{len(single_year_values):,} (out of {len(birth_year_stats):,}) individuals with *single* (i.e. consistent) birth year..."
    )
    pre_null_count = sum(df["birthy"].isna())
    if len(single_year_values) > 0:
        df = df.merge(single_year_values, on="pidp", how="left")
        df["birthy"] = df["first"].fillna(df["birthy"])
        df.drop(columns=["first"], inplace=True)
        print(
            f"After broadcasting consistent birth year, have {sum(df['birthy'].isna()):,} (from {pre_null_count:,}) NA birth year values..."
        )

    # Handle inconsistencies of multiple birth years (including or not including NAs) according to logic
    multiple_birth_year_individuals = birth_year_stats[birth_year_stats["nunique"] > 1]
    if len(multiple_birth_year_individuals) > 0:
        print(
            f"Dealing with remaining {len(multiple_birth_year_individuals)} individuals with multiple birth years..."
        )
    easy_cases = 0
    for pidp in multiple_birth_year_individuals["pidp"]:
        person_data = df.loc[df["pidp"] == pidp]
        birth_years = person_data["birthy"]
        birth_year_counts = birth_years.value_counts()  # Ignores NAs
        birth_year_range = max(birth_years) - min(birth_years)

        # Calculate mode
        mode_birth_year = birth_year_counts.index[0]
        mode_birth_year_count = birth_year_counts.iloc[0]
        mode_birth_year_percentage = mode_birth_year_count / len(birth_years)

        # Easy case: have significant mode to broadcast
        if mode_birth_year_percentage > min_mode_percentage:
            # Use the most frequent birth year
            cleaned_birth_year = mode_birth_year
            easy_cases += 1

        # Medium case: likely a data entry error
        elif birth_year_range <= allowed_birth_year_range:
            if mode_birth_year_count >= 2:
                # e.g. we have 1960, 1960, 1960, 1961, 1961 -> use 1960
                cleaned_birth_year = mode_birth_year
            else:
                # e.g. we have 1960, 1961 -> use 1960
                cleaned_birth_year = birth_years.min()
            easy_cases += 1

        # Fallback strategy
        elif strategy == "first_wave":
            # Use birth year from the first wave in which the birth year is not null
            first_valid_wave = (
                person_data.sort_values("wave")
                .loc[person_data["birthy"].notna()]
                .iloc[0]
            )
            cleaned_birth_year = first_valid_wave["birthy"]
        elif strategy == "last_wave":
            # Use birth year from last wave
            last_valid_wave = (
                person_data.sort_values("wave")
                .loc[person_data["birthy"].notna()]
                .iloc[-1]
            )
            cleaned_birth_year = last_valid_wave["birthy"]
        elif strategy == "mode":
            # Use the most frequent birth year
            cleaned_birth_year = (
                mode_birth_year if mode_birth_year_count >= 2 else birth_years.min()
            )
        elif strategy == "median":
            # Use the median birth year
            cleaned_birth_year = int(birth_years.median())
        elif strategy == "min":
            # Use the min birth year
            cleaned_birth_year = birth_years.min()
        elif strategy == "max":
            # Use the max birth year
            cleaned_birth_year = birth_years.max()
        elif strategy == "drop":
            # Drop the individual
            print(
                f"Warning: Will drop individual '{pidp}' with birth year inconsistencies..."
            )
            cleaned_birth_year = -9999
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Replace person's birth year for all waves with the cleaned birth year
        assert pd.notna(cleaned_birth_year), "Cleaned birth year is NA"
        df.loc[df["pidp"] == pidp, "birthy"] = cleaned_birth_year

    # Since only remaining NAs occured in inconsistent individuals, none should remain
    assert (
        df["birthy"].notna().all()
    ), "Still found individuals with missing birth years"

    if strategy == "drop":
        drop_pids = df[df["birthy"] == -9999]["pidp"].unique()
        df.drop(df[df["pidp"].isin(drop_pids)].index, inplace=True)
        assert len(birth_year_stats) - len(df["pidp"].unique()) == len(
            drop_pids
        ), f"Number of dropped individuals does not match: {len(drop_pids)}"
        print(f"Dropped {len(drop_pids):,} individuals with birth year inconsistencies")

    print(
        f"Dealt with {easy_cases:,} easy cases (broadcasting mode) and {len(multiple_birth_year_individuals) - easy_cases:,} fallback cases (using {strategy} strategy)...\n"
    )

    # Verify that the birth year is now consistent
    assert (
        df["birthy"].notna().all()
    ), "Still found individuals with missing birth years"
    new_birth_stats = (
        df.groupby("pidp")["birthy"]
        .agg(
            nunique=lambda x: (
                x.dropna().nunique()
            )  # Use explicit lambda to avoid potential pandas groupby issues
        )
        .reset_index()
    )
    assert (
        new_birth_stats["nunique"] == 1
    ).all(), "Still found individuals with birth year inconsistencies"

    return df


def investigate_istrtdaty_consistency(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Investigate consistency between wave numbers and interview dates (istrtdaty).

    For each individual, this function checks whether the difference between wave
    numbers matches the difference between interview dates for all pairs of
    observations. In a consistent dataset, if wave_i - wave_j = k, then
    istrtdaty_i - istrtdaty_j should also equal k (or approximately k, accounting
    for survey timing).

    An individual is considered inconsistent if there exists at least one pair
    (i, j) where wave_i - wave_j ≠ istrtdaty_i - istrtdaty_j.

    Args:
        df: DataFrame with columns 'pidp', 'wave', and 'istrtdaty'.
            'wave' should be numeric wave numbers (1, 2, 3, ...).
            'istrtdaty' should be interview start date years (e.g., 2009, 2010, ...).
        verbose: If True, prints detailed statistics and sample inconsistencies.
            If False, only returns summary counts.

    Returns:
        dict: Dictionary containing consistency analysis results:
            - 'n_consistent' (int): Number of individuals with consistent wave-date relationships
            - 'n_inconsistent' (int): Number of individuals with inconsistent relationships
            - 'n_total' (int): Total number of individuals analyzed
            - 'inconsistent_pidps' (list): List of pidp values with inconsistencies
            - 'inconsistency_details' (pd.DataFrame): DataFrame with detailed inconsistency
              information, including wave pairs, date differences, and discrepancy amounts.
              Only populated if verbose=True or if inconsistencies are found.

    Raises:
        ValueError: If required columns ('pidp', 'wave', 'istrtdaty') are missing

    Note:
        - Only individuals with at least 2 observations are analyzed
        - Uses np.isclose() for floating-point comparison (rtol=1e-9)
        - Missing values in 'wave' or 'istrtdaty' are excluded from analysis

    Example:
        >>> results = investigate_istrtdaty_consistency(df, verbose=True)
        >>> print(f"Found {results['n_inconsistent']} inconsistent individuals")
        >>> if len(results['inconsistency_details']) > 0:
        ...     print(results['inconsistency_details'].head())
    """
    if verbose:
        print("=" * 80)
        print("INVESTIGATING istrtdaty CONSISTENCY WITH WAVE NUMBERS")
        print("=" * 80)

    # Check required columns
    required_cols = ["pidp", "wave", "istrtdaty"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Filter to rows with valid wave and istrtdaty values
    df_check: pd.DataFrame = df.loc[:, ["pidp", "wave", "istrtdaty"]].copy()
    df_check = df_check.dropna(subset=["wave", "istrtdaty"])

    if len(df_check) == 0:
        if verbose:
            print("No valid observations found (all have missing wave or istrtdaty)")
        return {
            "n_consistent": 0,
            "n_inconsistent": 0,
            "n_total": 0,
            "inconsistent_pidps": [],
            "inconsistency_details": pd.DataFrame(),
        }

    # Group by pidp and check consistency
    inconsistent_pidps = []
    inconsistency_details = []

    for pidp, group in df_check.groupby("pidp"):
        # Sort by wave to ensure proper ordering
        group = group.sort_values("wave").reset_index(drop=True)

        # Need at least 2 observations to check consistency
        if len(group) < 2:
            continue

        # Check all pairs (i, j) where i != j
        is_inconsistent = False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                wave_i = group.iloc[i]["wave"]
                wave_j = group.iloc[j]["wave"]
                istrtdaty_i = group.iloc[i]["istrtdaty"]
                istrtdaty_j = group.iloc[j]["istrtdaty"]

                wave_diff = wave_i - wave_j
                istrtdaty_diff = istrtdaty_i - istrtdaty_j

                # Check if differences match
                if not np.isclose(wave_diff, istrtdaty_diff, rtol=1e-9):
                    is_inconsistent = True
                    inconsistency_details.append(
                        {
                            "pidp": pidp,
                            "wave_i": wave_i,
                            "wave_j": wave_j,
                            "istrtdaty_i": istrtdaty_i,
                            "istrtdaty_j": istrtdaty_j,
                            "wave_diff": wave_diff,
                            "istrtdaty_diff": istrtdaty_diff,
                            "difference": abs(wave_diff - istrtdaty_diff),
                        }
                    )

        if is_inconsistent:
            inconsistent_pidps.append(pidp)

    # Calculate statistics
    n_total = df_check["pidp"].nunique()
    n_inconsistent = len(inconsistent_pidps)
    n_consistent = n_total - n_inconsistent

    # Create details dataframe
    inconsistency_df = (
        pd.DataFrame(inconsistency_details) if inconsistency_details else pd.DataFrame()
    )

    # Print results
    if verbose:
        print(f"\nTotal individuals analyzed: {n_total:,}")
        print(
            f"Consistent individuals: {n_consistent:,} ({n_consistent / n_total * 100:.2f}%)"
        )
        print(
            f"Inconsistent individuals: {n_inconsistent:,} ({n_inconsistent / n_total * 100:.2f}%)"
        )

        if len(inconsistency_df) > 0:
            print(f"\nTotal inconsistent pairs found: {len(inconsistency_df):,}")
            print("\nSample inconsistencies (first 10):")
            print(inconsistency_df.head(10).to_string(index=False))

            if len(inconsistency_df) > 0:
                print("\nInconsistency statistics:")
                print(
                    f"  Mean absolute difference: {inconsistency_df['difference'].mean():.2f} years"
                )
                print(
                    f"  Median absolute difference: {inconsistency_df['difference'].median():.2f} years"
                )
                print(
                    f"  Max absolute difference: {inconsistency_df['difference'].max():.2f} years"
                )
                print(
                    f"  Min absolute difference: {inconsistency_df['difference'].min():.2f} years"
                )

        print("=" * 80)

    return {
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "n_total": n_total,
        "inconsistent_pidps": inconsistent_pidps,
        "inconsistency_details": inconsistency_df,
    }


def create_wavey_variable(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Create a smoothed 'wavey' variable by resolving wave-date inconsistencies.

    This function attempts to create a consistent "wave year" variable (wavey) that
    satisfies the relationship: wavey = wave + k for a constant k across all
    observations for each individual. This smooths out inconsistencies in interview
    dates that may arise from survey timing variations.

    The function:
    1. Identifies individuals with inconsistent wave-date relationships
    2. Tries to find a constant k such that wavey = wave + k for all observations
    3. Validates that all resulting wavey values fall within wave boundaries
    4. Sets wavey to NaN for individuals where no valid k can be found
    5. Drops individuals with missing wavey values

    Args:
        df: DataFrame with required columns:
            - 'pidp': Individual identifier
            - 'wave': Wave number (1, 2, 3, ...)
            - 'istrtdaty': Interview start date year
            - 'wave_start': Wave start date (YYYY-MM-DD format)
            - 'wave_end': Wave end date (YYYY-MM-DD format)
            - 'wave_mid_year': Wave midpoint year (for imputing missing values)
        verbose: If True, prints detailed information for each individual processed.
            Useful for debugging but can be verbose for large datasets.

    Returns:
        pd.DataFrame: DataFrame with 'wavey' column added. Individuals for whom
            no consistent wavey could be determined are dropped (rows with NaN wavey).

    Raises:
        ValueError: If required columns are missing
        AssertionError: If validation checks fail (should not occur in normal operation)

    Note:
        - For individuals with a single observation and missing istrtdaty, wavey
          is set to the wave's mid_year
        - The function modifies the input DataFrame in place but also returns it
        - Processing statistics (number shifted, skipped, etc.) are printed to stdout

    Example:
        >>> df = create_wavey_variable(df, verbose=False)
        >>> # Now use wavey instead of istrtdaty for age calculations
        >>> df['age'] = df['wavey'] - df['birthy']
    """
    print("\nCreating smooth 'wavey' variable over survey waves...")

    # Check required columns
    required_cols = [
        "pidp",
        "wave",
        "istrtdaty",
        "wave_start",
        "wave_end",
        "wave_mid_year",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Helper function to extract year from date string (e.g., '2010-01' -> 2010)
    def extract_year(date_str):
        if pd.isna(date_str):
            return np.nan
        if isinstance(date_str, str):
            return int(date_str.split("-")[0])
        return float(date_str)

    # Helper function to check if a date (year) falls within wave_start and wave_end
    def is_within_wave_bounds(year, wave_start, wave_end):
        start_year = extract_year(wave_start)
        end_year = extract_year(wave_end)
        assert pd.notna(start_year) and pd.notna(end_year), "Wave start or end is NA"
        return start_year <= year <= end_year

    # Set istrtdaty = np.nan if not between wave_start and wave_end
    df.loc[
        (df["istrtdaty"] < df["wave_start"].apply(extract_year))
        | (df["istrtdaty"] > df["wave_end"].apply(extract_year)),
        "istrtdaty",
    ] = np.nan

    # Initialize wavey with original istrtdaty (will be updated below)
    df["wavey"] = df["istrtdaty"].copy()

    # Process each individual
    n_shifted = 0
    n_skipped = 0
    n_single_missing = 0

    for pidp, group in df.groupby("pidp"):
        # Need at least 2 observations to check consistency
        if len(group) == 1 and pd.isna(group["istrtdaty"].iloc[0]):
            # If missing istrtdaty, set wavey to mid year of wave
            n_single_missing += 1
            df.loc[df["pidp"] == pidp, "wavey"] = group["wave_mid_year"].iloc[0]
            if verbose:
                print(
                    f"Set missing wavey to mid year of wave for individual {pidp} with only one observation"
                )
            continue

        # Check if already consistent (i.e. one survey year per wave)
        if len(group) == group["wavey"].nunique():
            continue

        # Check if there is a consistent wavey ordering such that
        # the difference between wavey and wave is the same for all observations
        # in the group (e.g. wavey = wave + k for all observations)
        # *AND* that the wavey values are within the wave_start and wave_end boundaries
        # If so, set wavey to the wavey values, otherwise set wavey to np.nan for all observations

        # Sort by wave for consistent processing, but preserve original index
        group = group.sort_values("wave")
        group_indices = group.index.tolist()
        group = group.reset_index(drop=True)

        # Try to find a constant k such that wavey = wave + k for all observations
        # We'll try k values based on reasonable year ranges (e.g., 2008-2010 for early waves)
        # Calculate a reasonable range for k based on the first observation
        best_k = None
        best_wavey_series = None

        # Infer reasonable k range from the data
        # If we have istrtdaty values, use them to estimate k
        valid_mask = group["istrtdaty"].notna()
        if valid_mask.sum() > 0:
            # Estimate k from existing data: k ≈ istrtdaty - wave
            estimated_k_values = (
                group.loc[valid_mask, "istrtdaty"] - group.loc[valid_mask, "wave"]
            )
            k_min = int(estimated_k_values.min()) - 2  # Try a few values below
            k_max = int(estimated_k_values.max()) + 2  # Try a few values above
        else:
            # Default range if no istrtdaty data
            k_min = 2008
            k_max = 2010

        # Try different k values
        for k in range(k_min, k_max + 1):
            # Calculate wavey = wave + k for all observations
            potential_wavey = group["wave"] + k

            # Check if all wavey values are within their wave boundaries
            all_within_bounds = True
            for i in range(len(group)):
                wavey_val = potential_wavey.iloc[i]
                row = group.iloc[i]
                if not is_within_wave_bounds(
                    wavey_val, row["wave_start"], row["wave_end"]
                ):
                    all_within_bounds = False
                    break

            if all_within_bounds:
                # Found a valid k!
                best_k = k
                # Create a Series with original indices for proper alignment
                best_wavey_series = pd.Series(
                    potential_wavey.values, index=group_indices
                )
                break

        # Apply the result
        if best_k is not None and best_wavey_series is not None:
            # Set wavey to the consistent values using original indices
            df.loc[best_wavey_series.index, "wavey"] = best_wavey_series.values
            n_shifted += 1
            if verbose:
                print(
                    f"Consistent ordering found for individual {pidp}, wavey = wave + {best_k}"
                )
        else:
            # No consistent ordering found, set all to np.nan
            if verbose:
                print(
                    f"No consistent ordering found for individual {pidp}, setting wavey to np.nan"
                )
            df.loc[group_indices, "wavey"] = np.nan
            n_skipped += 1

    # Print results
    n_total_individuals = df["pidp"].nunique()
    print(f"\nTotal individuals analyzed: {n_total_individuals:,}")
    print(f"Individuals with single missing istrtdaty imputed: {n_single_missing:,}")
    print(f"Individuals with dates shifted to consistent wavey ordering: {n_shifted:,}")

    missing_wavey = sum(df["wavey"].isna())
    missing_wavey_individuals = df[df["wavey"].isna()]["pidp"].nunique()
    assert (
        missing_wavey_individuals == n_skipped
    ), "Number of missing wavey individuals does not match"
    print(
        f"Dropping {missing_wavey:,} missing wavey values from {missing_wavey_individuals:,} individuals (no valid shift found)"
    )
    return df.dropna(subset=["wavey"])


def preprocess_ukhls_panel_data_for_clustering(
    input_filepath: Path,
    output_filepath: Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for UKHLS panel data for clustering analysis.

    This is the main preprocessing function that orchestrates the entire data
    cleaning and preparation pipeline. It performs the following steps:

    1. **Data Loading**: Loads UKHLS panel data from various file formats
    2. **Data Validation**: Verifies required columns and data integrity
    3. **Birth Year Processing**: Resolves birth year inconsistencies across waves
    4. **Age Calculation**: Computes age from birth year and wave information
    5. **Column Cleaning**: Handles missing values, outliers, and invalid ranges
       for health, demographic, and other variables
    6. **Duplicate Handling**: Identifies and handles duplicate pidp-age combinations
    7. **Data Export**: Saves processed panel data to CSV

    Args:
        input_filepath: Path to the input UKHLS panel data file. Supported formats:
            parquet, CSV, feather, HDF5, pickle. Format is inferred from extension.
        output_filepath: Path where processed data will be saved. Output format
            is CSV regardless of input format.
        columns: Optional list of column names to load and process. If None,
            all columns are loaded.

    Returns:
        pd.DataFrame: Fully processed DataFrame ready for clustering analysis.
            Contains consistent birth years, cleaned health variables, and
            calculated age variable.

    Raises:
        FileNotFoundError: If input_filepath doesn't exist
        ValueError: If required columns are missing or file format is unsupported
        AssertionError: If data integrity checks fail (duplicates, missing values)

    Note:
        - Required columns: ['pidp', 'wave', 'birthy', 'sf12pcs_dv']
        - The function prints detailed progress information throughout processing
        - One output file is created: the main processed panel data CSV

    Example:
        >>> from pathlib import Path
        >>> df = preprocess_ukhls_panel_data_for_clustering(
        ...     input_filepath=Path("data/ukhls_indresp_combined.parquet"),
        ...     output_filepath=Path("data/ukhls_indresp_processed.csv"),
        ...     columns=['pidp', 'wave', 'birthy', 'sf12pcs_dv', 'sf12mcs_dv']
        ... )
        >>> print(f"Processed {len(df):,} observations from {df['pidp'].nunique():,} individuals")
    """
    print("=" * 80)
    print("PREPROCESSING UKHLS PANEL DATA FOR CLUSTERING")
    print("=" * 80)
    ensure_runtime_directories()
    print(f"Loading data from:\n'{input_filepath}'")
    print(f"Target columns: {', '.join(columns) if columns else 'all'}")
    print(f"Saving processed data to:\n'{output_filepath}'")
    print("=" * 80 + "\n")

    cols_use = columns
    if cols_use is not None:
        available = _available_columns(input_filepath)
        if available:
            cols_use = [c for c in cols_use if c in available]
            # Always include any UKHLS survey weights present in the raw file
            weight_cols = _discover_weight_columns(available)
            cols_use = list(dict.fromkeys(cols_use + weight_cols))
            # Always include survey design variables present in the raw file,
            # mirroring the v2 lane's foundation-column injection, so an
            # explicit column override cannot drop PSU/strata linkage.
            design_cols = [
                column for column in SURVEY_DESIGN_SOURCE_COLUMNS if column in available
            ]
            cols_use = list(dict.fromkeys(cols_use + design_cols))
    df = load_dataframe(input_filepath, cols_use)

    # Verify that we have necessary minimal columns for clustering
    for column in ["pidp", "wave", "birthy", "sf12pcs_dv"]:
        assert column in df.columns, f"Required '{column}' column is missing"

    # Verify that we have no null values in index columns
    assert not df["pidp"].isna().any(), "pidp column contains null values"
    assert not df["wave"].isna().any(), "wave column contains null values"

    # Verify that we have no duplicate pidp-wave combinations
    duplicates = df.duplicated(subset=["pidp", "wave"])
    assert not duplicates.any(), "Duplicate pidp-wave combinations found"

    starting_person_years = df.shape[0]
    starting_persons = len(df["pidp"].unique())
    print(
        f"Beginning preprocessing with {starting_person_years:,} person-years "
        f"from {starting_persons:,} persons...\n"
    )

    # Drop individuals with less than min_health_points waves
    tmp = df.groupby("pidp")["wave"].agg(num_waves="nunique").reset_index()
    print(tmp.describe())
    max_wave_count = int(tmp["num_waves"].max()) if not tmp.empty else 0
    wave_count_thresholds = list(range(6, max_wave_count + 1, 2))
    if max_wave_count >= 6 and max_wave_count not in wave_count_thresholds:
        wave_count_thresholds.append(max_wave_count)
    for n in wave_count_thresholds:
        print(
            f"We have {tmp[tmp['num_waves'] >= n].shape[0]:,} individuals "
            f"({tmp[tmp['num_waves'] >= n].shape[0] / tmp.shape[0] * 100:.2f}%) with at least {n} waves"
        )

    # Determine birth years
    df = preprocess_birth_years(df)
    df["age"] = df["wave"] + 2008 - df["birthy"]  # Wave year - birth year
    df["age"] = df["age"].astype(int)
    assert df["age"].notna().all(), "Still found individuals with missing age"

    print(
        "Preprocessing remaining variables handling missing values and outliers..."
    )  # Note: no error raised if columns not present
    # Core health variables
    preprocess_column(
        df, "sf12pcs_dv", known_nans=[-9, -8, -7], min_value=0, max_value=100
    )
    preprocess_column(
        df, "sf12mcs_dv", known_nans=[-9, -8, -7], min_value=0, max_value=100
    )
    # UKHLS raw SF-12 scores (recommended for longitudinal work with survey weights, if present)
    preprocess_column(
        df, "w_sf12pcs", known_nans=[-9, -8, -7], min_value=0, max_value=100
    )
    preprocess_column(
        df, "w_sf12mcs", known_nans=[-9, -8, -7], min_value=0, max_value=100
    )
    # UKHLS survey weights (keep if present)
    for weight_col in _discover_weight_columns(list(df.columns)):
        if weight_col in {"w_sf12pcs", "w_sf12mcs"}:
            continue
        preprocess_column(df, weight_col, known_nans=UKHLS_MISSING_VALUES, min_value=0)

    # Mortality and death-linkage fields merged in io.py (keep if present)
    for mortality_col in MORTALITY_RETAINED_COLUMNS:
        preprocess_column(
            df,
            mortality_col,
            known_nans=UKHLS_MISSING_VALUES,
            min_value=0,
        )
    derive_deceased_binary_from_dcsedfl(df)
    derive_deceased_year_from_dcsedw(df)

    # Household composition and care context
    preprocess_column(df, "hhsize", known_nans=[-9, -8, -7, -2, -1], min_value=1)
    preprocess_column(df, "hhtype_dv", known_nans=[-9, -8, -7, -2, -1], min_value=1)
    preprocess_column(df, "nchild_dv", known_nans=[-9, -8, -7, -2, -1], min_value=0)
    derive_lives_alone_flag(df)

    preprocess_column(
        df, "ccare", known_nans=[-9, -8, -7, -2, -1], min_value=1, max_value=2
    )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="ccare",
        output_col="provides_care_flag",
        positive_values=[1.0],
        negative_values=[2.0],
    )
    preprocess_column(
        df, "careass", known_nans=[-9, -8, -7, -2, -1], min_value=1, max_value=2
    )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="careass",
        output_col="care_needs_assessment_flag",
        positive_values=[1.0],
        negative_values=[2.0],
    )
    preprocess_column(
        df, "lacare", known_nans=[-9, -8, -7, -2, -1], min_value=1, max_value=2
    )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="lacare",
        output_col="local_authority_care_flag",
        positive_values=[1.0],
        negative_values=[2.0],
    )
    preprocess_column(
        df, "paypriv", known_nans=[-9, -8, -7, -2, -1], min_value=1, max_value=2
    )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="paypriv",
        output_col="private_care_payment_flag",
        positive_values=[1.0],
        negative_values=[2.0],
    )
    preprocess_column(
        df,
        "servuse3",
        known_nans=[-10, -9, -8, -7, -2, -1],
        min_value=0,
        max_value=1,
    )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="servuse3",
        output_col="social_care_service_use_flag",
        positive_values=[1.0],
        negative_values=[0.0],
    )
    for helpcode_col in CARE_HELP_CODE_SOURCE_COLUMNS:
        preprocess_column(
            df,
            helpcode_col,
            known_nans=[-9, -8, -7, -2, -1],
            min_value=1,
        )
    derive_nonmissing_column_count(
        df,
        source_cols=CARE_HELP_CODE_SOURCE_COLUMNS,
        output_col="care_helper_count",
    )

    # Self-reported health and SF-12 questionnaire items
    for sf_col in GENERAL_HEALTH_SOURCE_COLUMNS:
        if sf_col in df.columns:
            preprocess_column(
                df,
                sf_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=1,
                max_value=5,
            )
    derive_coalesced_column(
        df,
        source_cols=GENERAL_HEALTH_SOURCE_COLUMNS,
        output_col="general_health_combined",
    )
    for sf_item_col in SF12_ITEM_SOURCE_COLUMNS:
        if sf_item_col in df.columns:
            preprocess_column(
                df,
                sf_item_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=1,
                max_value=5,
                verbose=False,
            )
    preprocess_column(
        df, "health", known_nans=[-9, -8, -7, -2, -1], min_value=1, max_value=2
    )  # 1=Yes long-standing illness/disability, 2=No
    derive_binary_indicator_from_coded_source(
        df,
        source_col="health",
        output_col="longstanding_illness_flag",
        positive_values=[1.0],
        negative_values=[2.0],
    )
    for disdif_col in DISABILITY_DIFFICULTY_COLUMNS:
        if disdif_col in df.columns:
            preprocess_column(
                df,
                disdif_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=0,
                max_value=1,
                verbose=False,
            )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="disdif1",
        output_col="mobility_impairment_flag",
        positive_values=[1.0],
        negative_values=[0.0],
    )
    derive_binary_source_count(
        df,
        source_cols=tuple(f"disdif{i}" for i in range(1, 13)),
        output_col="disability_difficulty_count",
        observed_col="disability_difficulty_count_observed",
    )
    if {"disdif96", "disability_difficulty_count"}.issubset(df.columns):
        no_difficulty = pd.to_numeric(df["disdif96"], errors="coerce").eq(1.0)
        count_missing = df["disability_difficulty_count"].isna()
        df.loc[no_difficulty & count_missing, "disability_difficulty_count"] = 0.0
        df.loc[
            no_difficulty & count_missing, "disability_difficulty_count_observed"
        ] = 1.0
    for dissev_col in DISABILITY_SEVERITY_COLUMNS:
        if dissev_col in df.columns:
            preprocess_column(
                df,
                dissev_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=1,
                max_value=4,
                verbose=False,
            )

    for adl_col in ADL_ABILITY_COLUMNS:
        if adl_col in df.columns:
            preprocess_column(
                df,
                adl_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=1,
                max_value=3,
                verbose=False,
            )
    for adl_ease_col in ADL_EASE_COLUMNS:
        if adl_ease_col in df.columns:
            preprocess_column(
                df,
                adl_ease_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=1,
                max_value=4,
                verbose=False,
            )
    derive_adl_limitation_count(df)

    # Diagnosed conditions and hospitalisation
    for condition_col in CONDITION_SOURCE_COLUMNS:
        if condition_col in df.columns:
            preprocess_column(
                df,
                condition_col,
                known_nans=UKHLS_MISSING_VALUES,
                verbose=False,
            )
    derive_serious_condition_count(df)
    derive_binary_source_count(
        df,
        source_cols=CONDITION_COUNT_SOURCE_COLUMNS,
        output_col="chronic_condition_count",
        observed_col="chronic_condition_count_observed",
    )
    for hospital_col in HOSPITAL_CONDITION_COLUMNS:
        if hospital_col in df.columns:
            preprocess_column(
                df,
                hospital_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=0,
                verbose=False,
            )
    derive_binary_indicator_from_coded_source(
        df,
        source_col="hosp",
        output_col="hospital_stay_flag",
        positive_values=[1.0],
        negative_values=[2.0],
    )
    derive_coded_positive_count(
        df,
        source_cols=HOSPITAL_CONDITION_COUNT_COLUMNS,
        output_col="hospital_condition_count",
        observed_col="hospital_condition_count_observed",
        positive_values=[1.0],
        negative_values=[0.0, 2.0],
    )
    derive_numeric_source_sum(
        df,
        source_cols=HOSPITAL_DAY_SOURCE_COLUMNS,
        output_col="hospital_days_total",
        observed_col="hospital_days_total_observed",
    )

    # Mental health and wellbeing
    preprocess_column(
        df, "scghq1_dv", known_nans=[-9, -8, -7, -2, -1], min_value=0, max_value=36
    )  # GHQ-12 score (0-36, higher = worse)
    preprocess_column(
        df, "scghq2_dv", known_nans=[-9, -8, -7, -2, -1], min_value=0, max_value=12
    )  # GHQ-12 caseness count (0-12); apply a 4+ threshold downstream if needed
    for ghq_item_col in GHQ_ITEM_SOURCE_COLUMNS:
        if ghq_item_col in df.columns:
            preprocess_column(
                df,
                ghq_item_col,
                known_nans=[-9, -8, -7, -2, -1],
                min_value=1,
                max_value=4,
                verbose=False,
            )
    if "swemwbs_dv" in df.columns:
        preprocess_column(
            df,
            "swemwbs_dv",
            known_nans=[-9, -8, -7, -2, -1],
            min_value=7,
            max_value=35,
            verbose=False,
        )
    for wellbeing_col in WELLBEING_SOURCE_COLUMNS:
        if wellbeing_col == "swemwbs_dv" or wellbeing_col not in df.columns:
            continue
        preprocess_column(
            df,
            wellbeing_col,
            known_nans=[-9, -8, -7, -2, -1],
            min_value=1,
            max_value=7,
            verbose=False,
        )

    # Demographics
    preprocess_column(df, "sex", known_nans=[-9, -2, -1], min_value=1, max_value=2)
    preprocess_column(df, "racel_dv", known_nans=[-9], min_value=1)  # Race
    preprocess_column(df, "hiqual_dv", known_nans=[-9, -8], min_value=1)  # Education
    preprocess_column(
        df, "country", known_nans=[-9, -8, -2, -1], min_value=1, max_value=4
    )  # Country of residence
    preprocess_column(df, "gor_dv", known_nans=[-9], min_value=1)  # Region
    preprocess_column(
        df, "urban_dv", known_nans=[-9, -8, -2, -1], min_value=1, max_value=2
    )  # 1=Urban, 2=Rural
    preprocess_column(df, "fimngrs_dv", known_nans=[-9])  # Income
    preprocess_column(
        df, "benpen1", known_nans=[-9, -8, -7, -2, -1], min_value=0, max_value=1
    )
    preprocess_column(
        df, "benunemp1", known_nans=[-9, -8, -7, -2, -1], min_value=0, max_value=1
    )

    # Marital status (additional socioeconomic)
    preprocess_column(
        df, "currmstat", known_nans=[-8, -7, -2, -1], min_value=1
    )  # Marital status

    # Employment status (additional socioeconomic)
    preprocess_column(
        df, "jbstat", known_nans=[-9, -8, -7, -2, -1], min_value=1
    )  # Job status
    preprocess_column(
        df, "employ", known_nans=[-9, -8, -7, -2, -1], min_value=1
    )  # Employment status

    # Smoking variables
    for smoking_col in ("smever", "smnow", "smoker"):
        if smoking_col in df.columns:
            preprocess_column(
                df,
                smoking_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=1,
                max_value=2,
                verbose=False,
            )
    for smoking_intensity_col in ("ncigs", "smcigs", "smncigs"):
        if smoking_intensity_col in df.columns:
            preprocess_column(
                df,
                smoking_intensity_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=0,
                verbose=False,
            )
    for smoking_age_col in ("aglquit", "smagbg"):
        if smoking_age_col in df.columns:
            preprocess_column(
                df,
                smoking_age_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=0,
                max_value=120,
                verbose=False,
            )

    # Physical activity and walking
    for activity_col in WALKING_ACTIVITY_SOURCE_COLUMNS:
        if activity_col in df.columns:
            preprocess_column(
                df,
                activity_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=0,
                verbose=False,
            )

    # Alcohol consumption
    if "evralc" in df.columns:
        preprocess_column(
            df,
            "evralc",
            known_nans=[-10, -9, -8, -7, -2, -1],
            min_value=1,
            max_value=2,
            verbose=False,
        )
    for alcohol_col in ALCOHOL_SOURCE_COLUMNS:
        if alcohol_col == "evralc" or alcohol_col not in df.columns:
            continue
        preprocess_column(
            df,
            alcohol_col,
            known_nans=[-10, -9, -8, -7, -2, -1],
            min_value=0,
            verbose=False,
        )

    # Diet and sleep
    for diet_col in DIET_SOURCE_COLUMNS:
        if diet_col in df.columns:
            preprocess_column(
                df,
                diet_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=0,
                verbose=False,
            )
    for sleep_col in SLEEP_SOURCE_COLUMNS:
        if sleep_col in df.columns:
            preprocess_column(
                df,
                sleep_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                min_value=0,
                verbose=False,
            )

    # Long Covid
    for long_covid_col in LONG_COVID_SOURCE_COLUMNS:
        if long_covid_col in df.columns:
            preprocess_column(
                df,
                long_covid_col,
                known_nans=[-10, -9, -8, -7, -2, -1],
                verbose=False,
            )

    # BMI - limited coverage (nurse assessment waves)
    preprocess_column(
        df, "bmi_dv", known_nans=[-9, -7], min_value=10, max_value=60
    )  # BMI (derived variable, continuous)

    # Additional study-only outcomes can be handled by callers explicitly.
    top20_vars: list[str] = []
    for var in top20_vars:
        if var in {
            "sf12pcs_dv",
            "sf12mcs_dv",
            "w_sf12pcs",
            "w_sf12mcs",
            "scsf1",
            "scghq1_dv",
            "scghq2_dv",
            "bmi_dv",
        }:
            continue
        preprocess_column(df, var, known_nans=UKHLS_MISSING_VALUES)

    # Handle duplicate pidp-age combinations (same person, same age, different waves)
    # Unfortuate reality of overlapping waves: same person can be interviewed multiple times in the same year
    duplicates = df.duplicated(subset=["pidp", "age"])
    assert not duplicates.any(), "Duplicate pidp-age combinations found"

    # Sort by pidp and age
    df = df.sort_values(by=["pidp", "age"]).reset_index(drop=True)

    # Save the dataframe to output filepath
    df.to_csv(output_filepath.with_suffix(".csv"), index=False)

    print("\n" + "=" * 80)
    print("Processing done!")
    print(
        f"Have {df.shape[0]:,} final observations from {len(df['pidp'].unique()):,} individuals."
    )
    print(
        f"Lost {starting_persons - len(df['pidp'].unique()):,} individuals due to missing or inconsistent data."
    )
    print(f"Processed data saved to:\n- '{output_filepath}'")
    print("=" * 80 + "\n")

    return df


def load_top20_outcome_variables(filepath: Path) -> list[str]:
    """Load the optional Top-20 outcome variable list from a CSV file."""
    if not filepath.exists():
        return []
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception:
        return []
    if "base_variable" not in df.columns:
        return []
    return [str(value) for value in df["base_variable"].dropna().tolist()]


def get_default_columns() -> list[str]:
    """Return the default analysis columns retained by preprocessing."""
    base_columns = [
        "pidp",
        "wave",
        "birthy",
        "sf12pcs_dv",
        "sf12mcs_dv",
        "w_sf12pcs",
        "w_sf12mcs",
        "sex",
        "racel_dv",
        "hiqual_dv",
        "country",
        "gor_dv",
        "urban_dv",
        "hhsize",
        "hhtype_dv",
        "nchild_dv",
        "ccare",
        "careass",
        "lacare",
        "paypriv",
        "servuse3",
        *CARE_HELP_CODE_SOURCE_COLUMNS,
        "fimngrs_dv",
        "health",
        *GENERAL_HEALTH_SOURCE_COLUMNS,
        *SF12_ITEM_SOURCE_COLUMNS,
        *SF12_ROUTING_SOURCE_COLUMNS,
        "scghq1_dv",
        "scghq2_dv",
        *GHQ_ITEM_SOURCE_COLUMNS,
        *WELLBEING_SOURCE_COLUMNS,
        *SLEEP_SOURCE_COLUMNS,
        *DISABILITY_DIFFICULTY_COLUMNS,
        *DISABILITY_SEVERITY_COLUMNS,
        *ADL_SOURCE_COLUMNS,
        *CONDITION_SOURCE_COLUMNS,
        *HOSPITAL_CONDITION_COLUMNS,
        "bmi_dv",
        *WALKING_ACTIVITY_SOURCE_COLUMNS,
        *SMOKING_SOURCE_COLUMNS,
        *ALCOHOL_SOURCE_COLUMNS,
        *DIET_SOURCE_COLUMNS,
        *LONG_COVID_SOURCE_COLUMNS,
        "currmstat",
        "jbstat",
        "employ",
        *PREDICTIVE_ANALYSIS_TOP20_EXTRA_COLUMNS,
        *MORTALITY_RETAINED_COLUMNS,
        *SURVEY_DESIGN_SOURCE_COLUMNS,
    ]
    return list(dict.fromkeys(base_columns))


if __name__ == "__main__":
    parser = build_cli_parser(
        (
            "Preprocess UKHLS panel data by cleaning values, resolving "
            "birth-year inconsistencies, and deriving analysis fields."
        ),
    )
    parser.add_argument(
        "--input_filepath",
        type=str,
        default=str(INTERIM_DATA_DIR / "ukhls_indresp_combined.parquet"),
        help=(
            "Path to the input UKHLS panel data file. "
            f"Default: {INTERIM_DATA_DIR / 'ukhls_indresp_combined.parquet'}"
        ),
    )

    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        default=get_default_columns(),
        help=(
            "Optional list of columns to load and process. "
            "If not specified, uses default set of columns. "
            "Example: --columns pidp wave birthy sf12pcs_dv "
            "Recommended full set: pidp wave birthy sf12pcs_dv sf12mcs_dv w_sf12pcs w_sf12mcs "
            "Note: if the input file contains UKHLS survey weight columns (e.g. *_xw, *_lw), "
            "they are automatically included when columns are provided. "
            "sex racel_dv hiqual_dv gor_dv fimngrs_dv "
            "general health, GHQ, wellbeing, functional-limitations, ADLs, chronic conditions, "
            "hospitalisation, BMI, walking/activity, smoking, alcohol, diet, sleep, Long Covid, "
            "currmstat jbstat employ"
        ),
    )

    parser.add_argument(
        "--output_filepath",
        type=str,
        default=str(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv"),
        help=(
            "Path to the output processed data file (CSV format). "
            f"Default: {PROCESSED_DATA_DIR / 'ukhls_indresp_processed.csv'}"
        ),
    )
    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_filepath)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please ensure the file exists or specify a different path with --input_filepath"
        )
    output_path = Path(args.output_filepath)
    log_script_start(
        Path(__file__).name,
        "Clean the combined long panel into the canonical processed analysis dataset.",
        inputs={"combined panel": input_path},
        outputs={"processed panel CSV": output_path},
        config={"column count": len(args.columns)},
    )

    # Run preprocessing pipeline
    preprocess_ukhls_panel_data_for_clustering(input_path, output_path, args.columns)
    log_script_end(Path(__file__).name, outputs={"processed panel CSV": output_path})
