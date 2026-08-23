"""
Legacy loader for UKHLS (Understanding Society) panel tab files.

This module is retained unchanged at the behavioral level so existing frozen
analyses can be reconstructed. New survey ingestion should use
``prevention_health_clustering.data.ingest`` / ``ingest-waves``, which enforces an
explicit wave and column contract, fingerprints inputs, preserves source
provenance, and writes a verified manifest.

This module provides functionality to:
- Load UKHLS survey wave data from tab-separated files
- Process and clean wave data with proper type conversion
- Combine multiple waves into a single panel dataset
- Optimize memory usage for large datasets
- Handle data collection dates and wave metadata

The UKHLS is a longitudinal household survey that collects data annually.
Each wave spans approximately two years of data collection with overlapping periods.

Example:
    Load all individual response waves:
    ```python
    from prevention_health_clustering.data.io import load_ukhls_panel_waves
    load_ukhls_panel_waves(panel_type="indresp", chunksize=10000)
    ```

References:
    - Understanding Society: https://www.understandingsociety.ac.uk/
    - Survey Timeline: https://www.understandingsociety.ac.uk/documentation/mainstage/survey-timeline/
"""

# Standard library imports
import time
from gc import collect as garbage_collect
from pathlib import Path
from typing import Optional, Union, cast

# Third-party imports
import numpy as np
import pandas as pd
from pandas.errors import DtypeWarning

from prevention_health_clustering.cli_utils import (
    build_cli_parser,
    log_script_end,
    log_script_start,
)

# Local imports
from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import (
    DATA_DIR,
    INTERIM_DATA_DIR,
    UKHLS_MISSING_CODES,
    UKHLS_PANEL_DIR,)

# UKHLS Survey Wave Dates (Externally Verified)
# Source: Understanding Society: UK Household Longitudinal Study (UKHLS)
# Reference: https://www.understandingsociety.ac.uk/documentation/mainstage/survey-timeline/
#
# Note: Each wave spans approximately two years of data collection with overlapping periods.
# Dates are in 'YYYY-MM-DD' format from the official survey timeline documentation.
#
# Structure:
#   - 'a' through 'o': Wave letters (a = wave 1, b = wave 2, etc.)
#   - 'start_date': First day of data collection (YYYY-MM-DD)
#   - 'end_date': Last day of data collection (YYYY-MM-DD)
#   - 'mid_year': Approximate midpoint year for the wave
#   - 'wave_number': Sequential wave number (1-15)
UKHLS_WAVE_DATES = {
    "a": {
        "start_date": "2009-01-08",
        "end_date": "2011-03-10",
        "mid_year": 2010,
        "wave_number": 1,
    },
    "b": {
        "start_date": "2010-01-10",
        "end_date": "2012-03-27",
        "mid_year": 2011,
        "wave_number": 2,
    },
    "c": {
        "start_date": "2011-01-02",
        "end_date": "2013-04-22",
        "mid_year": 2012,
        "wave_number": 3,
    },
    "d": {
        "start_date": "2012-01-03",
        "end_date": "2014-05-19",
        "mid_year": 2013,
        "wave_number": 4,
    },
    "e": {
        "start_date": "2013-01-03",
        "end_date": "2015-04-13",
        "mid_year": 2014,
        "wave_number": 5,
    },
    "f": {
        "start_date": "2014-01-08",
        "end_date": "2016-05-09",
        "mid_year": 2015,
        "wave_number": 6,
    },
    "g": {
        "start_date": "2015-01-08",
        "end_date": "2017-05-16",
        "mid_year": 2016,
        "wave_number": 7,
    },
    "h": {
        "start_date": "2016-01-04",
        "end_date": "2018-05-11",
        "mid_year": 2017,
        "wave_number": 8,
    },
    "i": {
        "start_date": "2017-01-05",
        "end_date": "2019-05-21",
        "mid_year": 2018,
        "wave_number": 9,
    },
    "j": {
        "start_date": "2018-01-09",
        "end_date": "2020-05-17",
        "mid_year": 2019,
        "wave_number": 10,
    },
    "k": {
        "start_date": "2019-01-04",
        "end_date": "2021-05-19",
        "mid_year": 2020,
        "wave_number": 11,
    },
    "l": {
        "start_date": "2020-01-08",
        "end_date": "2022-05-18",
        "mid_year": 2021,
        "wave_number": 12,
    },
    "m": {
        "start_date": "2021-01-06",
        "end_date": "2023-05-18",
        "mid_year": 2022,
        "wave_number": 13,
    },
    "n": {
        "start_date": "2022-01-13",
        "end_date": "2024-05-15",
        "mid_year": 2023,
        "wave_number": 14,
    },
    "o": {
        # Wave 15 was issued from January 2023 through December 2024, with
        # completed interviews extending into 2025 in the redownloaded release.
        # Day-level bounds are only used for display and year-based consistency
        # checks downstream, so calendar-window bounds are sufficient here.
        "start_date": "2023-01-01",
        "end_date": "2025-05-31",
        "mid_year": 2024,
        "wave_number": 15,
    },
}

UKHLS_MISSING_VALUES = set(UKHLS_MISSING_CODES.keys())
UKHLS_WAVE_LETTERS = tuple(UKHLS_WAVE_DATES.keys())
XWAVE_PERSON_MORTALITY_COLUMNS = (
    "dcsedfl_dv",
    "dcsedw_dv",
    "fwenum_dv",
    "lwenum_dv",
    "fwintvd_dv",
    "lwintvd_dv",
)
XWAVEID_PER_WAVE_COLUMNS = ("ivfio", "ivfho", "month")
XWAVEDAT_PER_WAVE_COLUMNS = ("mortbh_tw", "mortus_tw")


def get_wave_dates(wave: str) -> dict:
    """
    Get the data collection dates and metadata for a specific UKHLS wave.

    Args:
        wave: Wave letter identifier (e.g., 'a', 'b', 'c', etc.)
            - 'a' corresponds to wave 1, 'b' to wave 2, and so on

    Returns:
        dict: Dictionary containing wave metadata with keys:
            - 'start_date' (str): First day of data collection (YYYY-MM-DD)
            - 'end_date' (str): Last day of data collection (YYYY-MM-DD)
            - 'mid_year' (int): Approximate midpoint year for the wave
            - 'wave_number' (int): Sequential wave number (1-15)

    Raises:
        KeyError: If the specified wave letter is not found in UKHLS_WAVE_DATES

    Example:
        >>> wave_info = get_wave_dates('a')
        >>> print(wave_info['wave_number'])
        1
        >>> print(wave_info['start_date'])
        '2009-01-08'
    """
    if wave not in UKHLS_WAVE_DATES:
        available_waves = ", ".join(UKHLS_WAVE_DATES.keys())
        raise KeyError(f"Wave '{wave}' not found. Available waves: {available_waves}")

    return UKHLS_WAVE_DATES[wave]


def print_wave_information() -> None:
    """
    Print formatted information about all UKHLS survey waves.

    Displays a summary table showing:
    - Wave letter and corresponding wave number
    - Data collection period (start and end dates) for each wave

    This function is useful for understanding the survey timeline and
    identifying which waves are available for analysis.

    Example:
        >>> print_wave_information()
        ================================================================================
        UKHLS SURVEY WAVE INFORMATION
        ================================================================================
        Understanding Society: UK Household Longitudinal Study (UKHLS)
        ...
    """
    print("=" * 80)
    print("UKHLS SURVEY WAVE INFORMATION")
    print("=" * 80)
    print("Understanding Society: UK Household Longitudinal Study (UKHLS)")
    print(
        "Source: https://www.understandingsociety.ac.uk/documentation/mainstage/survey-timeline/"
    )
    print(
        "Each wave spans roughly two years of data collection with overlapping periods"
    )
    print("=" * 80)

    for wave, info in UKHLS_WAVE_DATES.items():
        print(f"Wave '{wave.upper()}' (i.e. survey wave {info['wave_number']})")
        print(f"Data Collection Period: {info['start_date']} to {info['end_date']}")
        print()
    print("=" * 80)


def load_ukhls_dataset(
    file_path: Path,
    sep: str = "\t",
    chunksize: Optional[int] = None,
    str_fallback: bool = False,
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Load UKHLS dataset from a tab-separated file with dtype warning handling.

    This function reads UKHLS tab files, which are tab-separated text files
    containing survey data. It handles dtype warnings that may occur when
    pandas cannot automatically infer column types.

    Args:
        file_path: Path to the tab-separated file to load
        sep: Column separator (default: "\t" for tab-separated files)
        chunksize: Number of rows to read per chunk. If None, reads entire file.
            When specified, returns a TextFileReader iterator for memory-efficient
            processing of large files.
        str_fallback: If True, reads all columns as strings when DtypeWarning occurs.
            If False, raises an error on dtype warnings.

    Returns:
        pd.DataFrame or pd.io.parsers.TextFileReader:
            - If chunksize is None: Returns a DataFrame with the full dataset
            - If chunksize is specified: Returns a TextFileReader iterator

    Raises:
        DtypeWarning: If dtype inference fails and str_fallback is False
        FileNotFoundError: If the specified file_path does not exist

    Note:
        Using `low_memory=False` may be very slow for large files but provides
        better type inference. For very large files, consider using chunksize
        for memory-efficient processing.

    Example:
        >>> # Load entire file
        >>> df = load_ukhls_dataset(Path("wave_a_indresp.tab"))

        >>> # Load in chunks for memory efficiency
        >>> for chunk in load_ukhls_dataset(Path("wave_a_indresp.tab"), chunksize=10000):
        ...     process_chunk(chunk)
    """
    try:
        df = pd.read_csv(file_path, sep=sep, low_memory=False, chunksize=chunksize)
    except DtypeWarning:
        if str_fallback:
            print(f"Warning: DtypeWarning for {file_path}, reading as string...")
            df = pd.read_csv(file_path, sep=sep, dtype=str, chunksize=chunksize)
        else:
            raise

    return df


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    This function reduces memory footprint by converting numeric columns to
    the smallest appropriate dtype that can represent all values:
    - int64 -> int8/int16/int32/uint8/uint16/uint32 (based on value range)
    - float64 -> float32 (when possible)

    Args:
        df: DataFrame to optimize. Must not contain empty strings in numeric columns.

    Returns:
        pd.DataFrame: Optimized DataFrame with reduced memory usage.
            The original DataFrame is modified in place, but the reference
            is also returned for convenience.

    Raises:
        AssertionError: If any numeric column contains empty strings

    Note:
        Memory savings are printed to stdout. Typical savings range from
        30-70% depending on the data distribution.

    Example:
        >>> df = pd.DataFrame({'col1': range(100), 'col2': [1.5] * 100})
        >>> df_optimized = optimize_dataframe_memory(df)
        >>> print(f"Memory reduced from X GB to Y GB")
    """
    print("Optimizing DataFrame memory usage...")
    original_memory = df.memory_usage(deep=True).sum() / 1024**3

    # Downcast numeric types
    numeric_columns = [
        str(column)
        for column, dtype in df.dtypes.items()
        if str(dtype) in {"int64", "float64"}
    ]
    for col in numeric_columns:
        if df[col].dtype == "int64":
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() < 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype("int8")
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")

        elif df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**3
    print(
        f"  Memory reduced from {original_memory:.1f} GB to {optimized_memory:.1f} GB"
    )
    return df


def _tab_header_columns(file_path: Path) -> list[str]:
    """Return the header columns for a tab-delimited UKHLS file."""
    if not file_path.exists():
        return []
    return list(pd.read_csv(file_path, sep="\t", nrows=0).columns)


def _read_tab_subset(file_path: Path, usecols: list[str]) -> pd.DataFrame:
    """
    Read a subset of columns from a UKHLS tab file.

    Missing files or empty column selections return an empty DataFrame with the
    requested schema so callers can compose fallbacks cleanly.
    """
    if not file_path.exists():
        return pd.DataFrame(columns=usecols)

    available_cols = _tab_header_columns(file_path)
    selected_cols = [col for col in usecols if col in available_cols]
    if not selected_cols:
        return pd.DataFrame(columns=usecols)

    return pd.read_csv(file_path, sep="\t", usecols=selected_cols, low_memory=False)


def _ukhls_missing_mask(series: pd.Series) -> pd.Series:
    """Treat both NaN and standard UKHLS negative sentinel codes as missing."""
    return series.isna() | series.isin(UKHLS_MISSING_VALUES)


def _coalesce_ukhls_series(
    index: pd.Index, *series_list: Optional[pd.Series]
) -> pd.Series:
    """
    Coalesce multiple UKHLS series while treating sentinel codes as missing.

    This prefers the first non-missing/non-sentinel value in source priority
    order, but preserves the original sentinel value if every source is missing.
    """
    aligned_series = [
        series.reindex(index) for series in series_list if series is not None
    ]
    if not aligned_series:
        return pd.Series(index=index, dtype="float64")

    result = aligned_series[0].copy()
    result_missing = _ukhls_missing_mask(result)
    for candidate in aligned_series[1:]:
        candidate_missing = _ukhls_missing_mask(candidate)
        take_candidate = result_missing & ~candidate_missing
        result = result.where(~take_candidate, candidate)
        result_missing = _ukhls_missing_mask(result)

    return result


def _harmonize_dcsedfl_source(
    index: pd.Index, series: Optional[pd.Series], source_name: str
) -> pd.Series:
    """
    Harmonize `dcsedfl_dv` to the shared 1/2/3 deceased coding.

    `xwaveid` uses `1 = Yes` and `2 = No`, while `xwavedat` and `xhhrel`
    use `1 = Yes, in UKHLS`, `2 = Yes, in BHPS`, and `3 = No`.
    """
    if series is None:
        return pd.Series(index=index, dtype="float64")

    aligned = pd.to_numeric(series.reindex(index), errors="coerce")
    aligned = aligned.where(~_ukhls_missing_mask(aligned))

    if source_name == "xwaveid":
        return pd.to_numeric(aligned.map({1.0: 1.0, 2.0: 3.0}), errors="coerce")

    return pd.to_numeric(aligned.where(aligned.isin([1.0, 2.0, 3.0])), errors="coerce")


def _coalesce_dcsedfl_series(
    index: pd.Index,
    xwaveid_series: Optional[pd.Series],
    xwavedat_series: Optional[pd.Series],
    xhhrel_series: Optional[pd.Series],
) -> pd.Series:
    """
    Coalesce `dcsedfl_dv` after harmonizing source-specific code schemes.

    Death signals take precedence over alive signals. If a UKHLS death flag is
    present anywhere, return `1`; otherwise return `2` for BHPS death, `3` for
    alive, and missing when no source provides a valid status.
    """
    harmonized_sources = (
        _harmonize_dcsedfl_source(index, xwaveid_series, "xwaveid"),
        _harmonize_dcsedfl_source(index, xwavedat_series, "xwavedat"),
        _harmonize_dcsedfl_source(index, xhhrel_series, "xhhrel"),
    )

    dead_ukhls = pd.Series(False, index=index, dtype="bool")
    dead_bhps = pd.Series(False, index=index, dtype="bool")
    alive = pd.Series(False, index=index, dtype="bool")
    for source in harmonized_sources:
        dead_ukhls = cast(pd.Series, dead_ukhls | source.eq(1.0))
        dead_bhps = cast(pd.Series, dead_bhps | source.eq(2.0))
        alive = cast(pd.Series, alive | source.eq(3.0))

    result = pd.Series(np.nan, index=index, dtype="float64")
    result = result.mask(dead_ukhls, 1.0)
    result = result.mask(~dead_ukhls & dead_bhps, 2.0)
    result = result.mask(~dead_ukhls & ~dead_bhps & alive, 3.0)
    return result


def _harmonize_dcsedw_source(
    index: pd.Index, series: Optional[pd.Series], source_name: str
) -> pd.Series:
    """
    Harmonize `dcsedw_dv` to the combined BHPS/UKHLS wave scale.

    `xwaveid` stores UKHLS-only wave codes `1..14`, while `xwavedat` and
    `xhhrel` continue the BHPS numbering so UKHLS Wave 1 starts at code `19`.
    """
    if series is None:
        return pd.Series(index=index, dtype="float64")

    aligned = pd.to_numeric(series.reindex(index), errors="coerce")
    aligned = aligned.where(~_ukhls_missing_mask(aligned))
    positive_codes = aligned.where(aligned > 0)

    if source_name == "xwaveid":
        ukhls_only_mask = positive_codes.between(1, len(UKHLS_WAVE_LETTERS))
        positive_codes = positive_codes.where(~ukhls_only_mask, positive_codes + 18.0)

    return positive_codes


def _coalesce_dcsedw_series(
    index: pd.Index,
    xwaveid_series: Optional[pd.Series],
    xwavedat_series: Optional[pd.Series],
    xhhrel_series: Optional[pd.Series],
) -> pd.Series:
    """
    Coalesce `dcsedw_dv` after harmonizing source-specific wave codings.

    When multiple non-missing death-wave codes disagree, keep the latest code
    so downstream analyses do not infer an earlier death report than any source
    supports.
    """
    harmonized = [
        _harmonize_dcsedw_source(index, xwaveid_series, "xwaveid"),
        _harmonize_dcsedw_source(index, xwavedat_series, "xwavedat"),
        _harmonize_dcsedw_source(index, xhhrel_series, "xhhrel"),
    ]
    return pd.concat(harmonized, axis=1).max(axis=1, skipna=True)


def load_xwave_mortality_sources(
    panel_dir: Path = UKHLS_PANEL_DIR,
) -> Optional[dict[str, pd.DataFrame]]:
    """
    Load mortality-related fields from the cross-wave `x*.tab` files.

    Sources:
    - `xwaveid.tab`: cross-wave death flags plus per-wave individual/household outcomes
    - `xwavedat.tab`: cross-wave death flags plus mortality weight adjustments
    - `xhhrel.tab`: fallback cross-wave death flags

    Returns:
        A dictionary of source DataFrames keyed by source name, or None when no
        `x*.tab` files are available.
    """
    xwaveid_path = panel_dir / "xwaveid.tab"
    xwavedat_path = panel_dir / "xwavedat.tab"
    xhhrel_path = panel_dir / "xhhrel.tab"

    if not any(path.exists() for path in (xwaveid_path, xwavedat_path, xhhrel_path)):
        return None

    xwaveid_cols = ["pidp", *XWAVE_PERSON_MORTALITY_COLUMNS]
    xwaveid_cols.extend(
        f"{wave}_{col}"
        for wave in UKHLS_WAVE_LETTERS
        for col in XWAVEID_PER_WAVE_COLUMNS
    )
    xwaveid = _read_tab_subset(xwaveid_path, xwaveid_cols)

    xwavedat_cols = ["pidp", *XWAVE_PERSON_MORTALITY_COLUMNS]
    xwavedat_cols.extend(
        f"{wave}_{col}"
        for wave in UKHLS_WAVE_LETTERS
        for col in XWAVEDAT_PER_WAVE_COLUMNS
    )
    xwavedat = _read_tab_subset(xwavedat_path, xwavedat_cols)

    xhhrel_cols = [
        "pidp",
        "dcsedfl_dv",
        "dcsedw_dv",
        "lwenum_dv",
        "lwintvd_dv",
    ]
    xhhrel = _read_tab_subset(xhhrel_path, xhhrel_cols)

    indexed_sources: list[pd.DataFrame] = []
    for source in (xwaveid, xwavedat, xhhrel):
        if source.empty:
            continue
        source = source.drop_duplicates(subset=["pidp"]).set_index("pidp")
        indexed_sources.append(source)

    if not indexed_sources:
        return None

    union_index = indexed_sources[0].index
    for source in indexed_sources[1:]:
        union_index = union_index.union(source.index)

    cross_wave = pd.DataFrame({"pidp": union_index.to_numpy()})
    xwaveid_indexed = xwaveid.drop_duplicates(subset=["pidp"]).set_index("pidp")
    xwavedat_indexed = xwavedat.drop_duplicates(subset=["pidp"]).set_index("pidp")
    xhhrel_indexed = xhhrel.drop_duplicates(subset=["pidp"]).set_index("pidp")

    for col in XWAVE_PERSON_MORTALITY_COLUMNS:
        xwaveid_series = (
            xwaveid_indexed[col] if col in xwaveid_indexed.columns else None
        )
        xwavedat_series = (
            xwavedat_indexed[col] if col in xwavedat_indexed.columns else None
        )
        xhhrel_series = xhhrel_indexed[col] if col in xhhrel_indexed.columns else None

        if col == "dcsedfl_dv":
            cross_wave[col] = _coalesce_dcsedfl_series(
                union_index,
                xwaveid_series,
                xwavedat_series,
                xhhrel_series,
            ).to_numpy()
            continue

        if col == "dcsedw_dv":
            cross_wave[col] = _coalesce_dcsedw_series(
                union_index,
                xwaveid_series,
                xwavedat_series,
                xhhrel_series,
            ).to_numpy()
            continue

        cross_wave[col] = _coalesce_ukhls_series(
            union_index,
            xwaveid_series,
            xwavedat_series,
            xhhrel_series,
        ).to_numpy()

    sources: dict[str, pd.DataFrame] = {"cross_wave": cross_wave}
    if not xwaveid.empty:
        sources["xwaveid"] = xwaveid
    if not xwavedat.empty:
        sources["xwavedat"] = xwavedat
    if not xhhrel.empty:
        sources["xhhrel"] = xhhrel

    return sources


def build_xwave_mortality_wave_extract(
    wave: str, x_sources: Optional[dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Build the per-wave mortality extract keyed on `pidp`.

    The returned DataFrame includes cross-wave death flags plus wave-specific
    fields like `ivfho` and mortality weights when available for that wave.
    """
    if x_sources is None or "cross_wave" not in x_sources:
        return pd.DataFrame(columns=["pidp"])

    wave_extract = x_sources["cross_wave"].copy()

    xwaveid = x_sources.get("xwaveid")
    if xwaveid is not None and not xwaveid.empty:
        rename_map = {
            f"{wave}_{col}": col
            for col in XWAVEID_PER_WAVE_COLUMNS
            if f"{wave}_{col}" in xwaveid.columns
        }
        if rename_map:
            selected_columns = ["pidp", *list(rename_map)]
            xwaveid_wave = (
                xwaveid.loc[:, selected_columns].rename(columns=rename_map).copy()
            )
            wave_extract = wave_extract.merge(
                xwaveid_wave, on="pidp", how="left", validate="one_to_one"
            )

    xwavedat = x_sources.get("xwavedat")
    if xwavedat is not None and not xwavedat.empty:
        rename_map = {
            f"{wave}_{col}": col
            for col in XWAVEDAT_PER_WAVE_COLUMNS
            if f"{wave}_{col}" in xwavedat.columns
        }
        if rename_map:
            selected_columns = ["pidp", *list(rename_map)]
            xwavedat_wave = (
                xwavedat.loc[:, selected_columns].rename(columns=rename_map).copy()
            )
            wave_extract = wave_extract.merge(
                xwavedat_wave, on="pidp", how="left", validate="one_to_one"
            )

    return wave_extract


def merge_xwave_mortality_variables(
    wave_df: pd.DataFrame,
    wave: str,
    x_sources: Optional[dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Merge mortality-related variables from the `x*.tab` sources into a wave DataFrame.

    Existing wave columns are preserved, so `indresp` keeps its native `ivfio`
    and `month` columns while still gaining death flags, `ivfho`, and mortality
    weights from the cross-wave files.
    """
    wave_extract = build_xwave_mortality_wave_extract(wave, x_sources)
    if wave_extract.empty:
        return wave_df

    columns_to_add = [
        col
        for col in wave_extract.columns
        if col == "pidp" or col not in wave_df.columns
    ]
    if columns_to_add == ["pidp"]:
        return wave_df

    return wave_df.merge(
        wave_extract[columns_to_add], on="pidp", how="left", validate="many_to_one"
    )


def save_dataframe(
    df: pd.DataFrame, filepath: Path, preferred_format: str = "parquet"
) -> bool:
    """
    Save DataFrame to disk with multiple format options and fallback strategies.

    This function attempts to save a DataFrame using the preferred format,
    with automatic fallbacks if the preferred method fails. This is particularly
    useful for large datasets where memory constraints may cause save failures.

    Supported formats (in order of preference):
    1. Parquet: Columnar format with excellent compression (default)
    2. Feather: Fast binary format for pandas DataFrames
    3. HDF5: Hierarchical data format with compression
    4. CSV: Plain text format (with gzip compression)

    Fallback strategy:
    - If preferred format fails, tries alternative compression methods
    - Falls back to CSV with compression
    - Finally tries uncompressed CSV

    Args:
        df: DataFrame to save to disk
        filepath: Output file path. Extension may be modified based on format.
        preferred_format: Preferred file format. One of:
            - 'parquet': Apache Parquet format (default, recommended)
            - 'feather': Feather format (fast, pandas-optimized)
            - 'hdf5': HDF5 format (good compression)
            - 'csv': CSV format (universal compatibility)

    Returns:
        bool: True if the DataFrame was saved successfully, False otherwise.
            Note: If all fallback methods fail, False is returned but no
            exception is raised (errors are printed to stdout).

    Raises:
        ValueError: If preferred_format is not one of the supported formats

    Note:
        - Memory usage is printed before saving
        - All errors during fallback attempts are caught and printed
        - The function does not raise exceptions for save failures (returns False)
        - File extension may be changed during fallback (e.g., .parquet -> .csv)

    Example:
        >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> success = save_dataframe(df, Path("output.parquet"))
        >>> if success:
        ...     print("Saved successfully!")
    """
    print(f"Saving {len(df):,} rows to {filepath}...")
    print(
        f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.1f} GB"
    )

    try:  # Try preferred format first
        if preferred_format == "parquet":
            df.to_parquet(filepath, compression="snappy", engine="pyarrow")
            print("✓ Saved with Parquet (snappy compression)")
        elif preferred_format == "feather":
            df.to_feather(filepath)
            print("✓ Saved with Feather format")
        elif preferred_format == "hdf5":
            df.to_hdf(filepath, key="data", mode="w", complevel=9, complib="blosc")
            print("✓ Saved with HDF5 format")
        elif preferred_format == "csv":
            df.to_csv(filepath, compression="gzip")
            print("✓ Saved with CSV (gzip compression)")
        else:
            raise ValueError(f"Unsupported format: {preferred_format}")
        return True

    except MemoryError as e:
        print(f"Error with {preferred_format}: {e}")
        print("Trying fallback methods...")

        # Fallback 1: Try different Parquet compression
        if preferred_format == "parquet":
            try:
                print("Trying Parquet with gzip compression...")
                df.to_parquet(filepath, compression="gzip", engine="pyarrow")
                print("✓ Saved with Parquet (gzip compression)")
                return True
            except Exception as e2:
                print(f"Parquet gzip failed: {e2}")

            try:
                print("Trying Parquet with no compression...")
                df.to_parquet(filepath, compression=None, engine="pyarrow")
                print("✓ Saved with Parquet (no compression)")
                return True
            except Exception as e3:
                print(f"Parquet no compression failed: {e3}")

        # Fallback 2: Try CSV
        try:
            print("Trying CSV with gzip compression...")
            df.to_csv(filepath.with_suffix(".csv"), compression="gzip")
            print("✓ Saved with CSV (gzip compression)")
            return True
        except Exception as e4:
            print(f"CSV gzip failed: {e4}")

        # Fallback 3: Try basic CSV
        try:
            print("Trying basic CSV...")
            df.to_csv(filepath.with_suffix(".csv"))
            print("✓ Saved with basic CSV")
            return True
        except Exception as e5:
            print(f"Basic CSV failed: {e5}")
            print("All methods failed - dataset may be too large or corrupted")
            return False


def load_chunk(chunk: pd.DataFrame, wave: str, panel: str) -> pd.DataFrame:
    """
    Process a chunk of UKHLS wave data: clean, convert types, and optimize.

    This function performs the following operations on a data chunk:
    1. Validates that required columns exist (pidp)
    2. Removes wave prefix from column names (e.g., 'a_sex' -> 'sex')
    3. Converts empty strings to NaN
    4. Converts columns to numeric types where appropriate
    5. Optimizes memory usage by downcasting numeric types

    Args:
        chunk: DataFrame containing raw wave data with wave-prefixed columns
            (e.g., columns like 'a_sex', 'a_age' for wave 'a')
        wave: Wave letter identifier (e.g., 'a', 'b', 'c')
            Used to identify and remove wave prefixes from column names
        panel: Panel type identifier (e.g., 'indresp', 'hh')
            Used for error messages and logging

    Returns:
        pd.DataFrame: Processed DataFrame with:
            - Wave prefixes removed from column names
            - Numeric columns properly typed
            - Memory-optimized data types
            - Empty strings converted to NaN

    Raises:
        AssertionError: If any of the following conditions are violated:
            - chunk is not a pandas DataFrame
            - 'pidp' column is missing or contains null values
            - Number of data columns doesn't match expected pattern
            - Number of missing values changes unexpectedly after type conversion
            - Wave prefix length is not 2 characters (wave letter + underscore)

    Note:
        The function modifies the input DataFrame in place for memory efficiency,
        but also returns it for convenience.

    Example:
        >>> raw_chunk = pd.DataFrame({'pidp': [1, 2], 'a_sex': ['1', '2'], 'a_age': ['25', '30']})
        >>> processed = load_chunk(raw_chunk, wave='a', panel='indresp')
        >>> # Columns are now 'sex' and 'age' (prefix removed)
    """
    assert isinstance(chunk, pd.DataFrame), f"Chunk is not a DataFrame: {type(chunk)}"
    assert chunk["pidp"].notna().all(), "pidp column is missing or has null values"

    # Using `pidp` as the index, drop `pid` column if exists
    chunk = chunk.drop(columns=["pid"], errors="ignore")

    data_cols = [col for col in chunk.columns if col.startswith(f"{wave}_")]
    assert (
        len(data_cols) == len(chunk.columns) - 1
    ), f"Unexpected number of data columns in chunk (wave: {wave}, panel: {panel})"

    # Replace empty strings with NaN
    chunk[data_cols] = chunk[data_cols].replace(r"^\s*$", np.nan, regex=True)
    missing_vals = int(chunk[data_cols].isna().to_numpy().sum())
    numeric_cols = cast(
        pd.DataFrame,
        chunk[data_cols].apply(pd.to_numeric, errors="coerce"),
    )
    assert (
        int(numeric_cols.isna().to_numpy().sum()) == missing_vals
    ), f"Unexpected number of missing values after conversion (wave: {wave}, panel: {panel})"
    chunk[data_cols] = numeric_cols

    # Remove wave prefix from column names
    assert len(wave) + 1 == 2, "Wave and underscore is not 2 characters"
    chunk = chunk.rename(columns={col: col[2:] for col in data_cols})

    # Optimize memory
    chunk = optimize_dataframe_memory(chunk)

    return chunk


def load_wave_and_save(
    wave_file: Path,
    chunksize: Optional[int] = None,
    output_filename: str = "ukhls",
    save_format: str = "parquet",
    x_sources: Optional[dict[str, pd.DataFrame]] = None,
) -> Path:
    """
    Load a single UKHLS wave file, process it, and save to disk.

    This function orchestrates the complete processing pipeline for a single wave:
    1. Extracts wave and panel identifiers from filename
    2. Loads wave metadata (dates, wave number)
    3. Processes the file (in chunks if specified)
    4. Adds wave metadata columns (wave number, dates)
    5. Validates data integrity (no duplicate pidp-wave combinations)
    6. Saves processed data to disk

    Args:
        wave_file: Path to the wave tab file. Filename should follow pattern:
            '{wave}_{panel}.tab' (e.g., 'a_indresp.tab')
        chunksize: Number of rows to process per chunk. If None, processes
            entire file in memory. Use chunksize for very large files to
            avoid memory issues.
        output_filename: Base name for output file. Wave letter will be appended.
            Final filename: '{output_filename}_{wave}.parquet'
        save_format: File format for output. See save_dataframe() for options.
            Default: 'parquet'
        x_sources: Optional cross-wave mortality sources loaded from the
            `x*.tab` files. When provided, mortality-related variables are
            merged onto the processed wave by `pidp`.

    Returns:
        Path: Path to the saved output file

    Raises:
        AssertionError: If:
            - Filename doesn't match expected pattern (wave_panel.tab)
            - Duplicate pidp-wave combinations are found
            - pidp or wave columns contain null values

    Note:
        - Wave information (number, dates) is printed to stdout
        - Memory is garbage collected after processing
        - Intermediate chunks are deleted after concatenation

    Example:
        >>> wave_file = Path("data/UKDA-6614-tab/tab/ukhls/a_indresp.tab")
        >>> output_path = load_wave_and_save(wave_file, chunksize=10000)
        >>> print(f"Saved to: {output_path}")
    """
    wave, panel = wave_file.stem.split("_")

    # Display wave information with dates
    wave_info = get_wave_dates(wave)
    print(
        f"Processing panel {panel}, wave {wave.upper()} (i.e. survey wave {wave_info['wave_number']})..."
    )
    print(
        f"  Data Collection Period: {wave_info['start_date']} to {wave_info['end_date']}"
    )

    # Output file path
    wave_output_file: Path = INTERIM_DATA_DIR / f"{output_filename}_{wave}.parquet"

    if chunksize is None:
        # Process in one large chunk
        dataset = load_ukhls_dataset(wave_file)
        if isinstance(dataset, pd.DataFrame):
            all_chunks = [load_chunk(dataset, wave, panel)]
        else:
            raise TypeError("Expected DataFrame when chunksize is None")
    else:
        # Process in small chunks
        all_chunks = [
            load_chunk(chunk, wave, panel)
            for chunk in load_ukhls_dataset(wave_file, chunksize=chunksize)
            if isinstance(chunk, pd.DataFrame)
        ]

    wave_df = pd.concat(all_chunks, ignore_index=True)
    # Assign new columns in one go to avoid fragmentation
    wave_df = pd.concat(
        [
            wave_df,
            pd.DataFrame(
                {
                    "wave": wave_info["wave_number"],
                    "wave_start_date": wave_info["start_date"],
                    "wave_mid_year": wave_info["mid_year"],
                    "wave_end_date": wave_info["end_date"],
                },
                index=wave_df.index,
            ),
        ],
        axis=1,
    )
    wave_df = merge_xwave_mortality_variables(wave_df, wave, x_sources)
    assert not (
        wave_df["pidp"].isna().any() or wave_df["wave"].isna().any()
    ), "index values contains null values"
    duplicates = wave_df.duplicated(subset=["pidp", "wave"])
    assert not duplicates.any(), "Duplicate pidp-wave combinations found"

    save_dataframe(wave_df, wave_output_file, preferred_format=save_format)
    print(
        f"✓ Processed panel {panel}, wave {wave} ({len(wave_df):,} rows in {len(all_chunks)} chunk/s)\n"
    )

    del all_chunks, wave_df
    garbage_collect()

    return Path(wave_output_file)


def load_ukhls_panel_waves(
    panel_type: str = "indresp",
    chunksize: Optional[int] = None,
    delete_intermediate_files: bool = True,
    output_filepath: Path = INTERIM_DATA_DIR / "ukhls_indresp_combined.parquet",
) -> None:
    """
    Load and combine all waves of UKHLS panel data into a single file.

    This is the main orchestration function that:
    1. Finds all wave files for the specified panel type
    2. Processes each wave file individually (saving intermediate files)
    3. Combines all waves into a single panel dataset
    4. Validates the combined dataset (no duplicate pidp-wave combinations)
    5. Saves the final combined dataset
    6. Optionally cleans up intermediate wave files

    The function is designed to handle large datasets efficiently by:
    - Processing waves sequentially to minimize memory usage
    - Using chunked reading for very large individual waves
    - Garbage collecting between waves
    - Providing progress information throughout

    Args:
        panel_type: Type of panel data to process. Must match the panel identifier
            in wave filenames. Common values:
            - 'indresp': Individual respondent data (default)
            - 'hh': Household data
            - 'indresp_ip': Individual respondent interview proxy
            - 'hh_ip': Household interview proxy
        chunksize: Number of rows to process per chunk when reading wave files.
            If None, each wave is processed entirely in memory. Recommended:
            - None for files < 1GB
            - 10000-50000 for larger files
        delete_intermediate_files: If True, deletes individual wave files after
            successful combination. If False, keeps them for debugging or
            incremental processing. Default: True
        output_filepath: Path where the final combined dataset will be saved.
            Default: data/interim/ukhls_indresp_combined.parquet

    Returns:
        None: Function prints status information to stdout but returns nothing.

    Raises:
        AssertionError: If duplicate pidp-wave combinations are found in the
            combined dataset (should never occur if source data is clean)
        FileNotFoundError: If the panel directory doesn't exist or contains
            no matching wave files

    Note:
        - Requires UKHLS data to be linked to the data directory:
          ``
        - Processing time depends on number of waves and file sizes
        - A 10-second pause is included between wave processing and combination
          to allow memory cleanup
        - The combined file may contain duplicate pidp-age combinations if
          individuals were interviewed multiple times in the same year
          (due to overlapping waves)

    Example:
        >>> # Load all individual respondent waves
        >>> load_ukhls_panel_waves(panel_type="indresp")

        >>> # Load with chunking for memory efficiency
        >>> load_ukhls_panel_waves(
        ...     panel_type="indresp",
        ...     chunksize=10000,
        ...     delete_intermediate_files=False
        ... )
    """
    print("=" * 80)
    print("UKHLS PANEL DATA LOADING\n")
    ensure_runtime_directories()
    print("We assume you have linked UKHLS to local data directory, e.g.:")
    print("```bash\nln -s /path/to/sensitive/UKHLS/UKDA-6614-tab data/\n```\n")
    print(
        f"We will read data from panel '{panel_type}' in local data directory:\n'{UKHLS_PANEL_DIR}'\n"
    )

    # Get list of wave files matching the panel type
    wave_files = sorted(
        UKHLS_PANEL_DIR.glob(f"*_{panel_type}.tab"),
        key=lambda path: get_wave_dates(path.stem.split("_")[0])["wave_number"],
    )
    if not wave_files:
        raise FileNotFoundError(
            f"No wave files found for panel type '{panel_type}' in {UKHLS_PANEL_DIR}. "
            f"Expected files matching pattern '*_{panel_type}.tab'"
        )
    print(f"Found {len(wave_files)} wave files, e.g., {wave_files[0].name}")

    x_sources = load_xwave_mortality_sources(UKHLS_PANEL_DIR)
    if x_sources is None:
        print("Cross-wave mortality files: none found, skipping `x*.tab` merge")
    else:
        available_x_files = sorted(
            name for name in ("xwaveid", "xwavedat", "xhhrel") if name in x_sources
        )
        print(
            "Cross-wave mortality files: "
            + ", ".join(f"{name}.tab" for name in available_x_files)
        )
        print(
            "Will merge death flags, household outcome, and mortality weights where available"
        )

    # Print processing configuration
    processing_mode = "one go" if chunksize is None else f"chunks of {chunksize:,} rows"
    print(f"Processing mode: {processing_mode}")
    print(f"Output file: {output_filepath}")
    print(
        f"Intermediate files: {'will be deleted' if delete_intermediate_files else 'will be kept'}"
    )
    print("=" * 80 + "\n")

    # Process each wave file sequentially
    print(f"Processing {len(wave_files)} wave files...")
    wave_files_processed = []
    for i, wave_file in enumerate(wave_files, 1):
        print(f"\n[{i}/{len(wave_files)}] Processing {wave_file.name}...")
        output_path = load_wave_and_save(
            wave_file,
            chunksize=chunksize,
            output_filename=f"ukhls_{panel_type}",
            x_sources=x_sources,
        )
        wave_files_processed.append(output_path)

    # Wait before combining to allow memory cleanup
    print("\nWaiting 10 seconds for memory cleanup before combining waves...")
    time.sleep(10)

    # Combine all processed waves into a single file
    print(
        f"\nCombining {len(wave_files_processed)} processed waves into a single file..."
    )
    try:
        # Read and concatenate all wave files
        print("Reading processed wave files...")
        wave_dataframes = [
            pd.read_parquet(wave_file) for wave_file in wave_files_processed
        ]
        df = pd.concat(wave_dataframes, ignore_index=True)
        del wave_dataframes  # Free memory
        print(
            f"✓ Concatenated {len(wave_files_processed)} waves into a single DataFrame"
        )
        print(f"  Total rows: {len(df):,}")
        print(f"  Total individuals (unique pidp): {df['pidp'].nunique():,}")

        # Validate data integrity
        duplicates = df.duplicated(subset=["pidp", "wave"])
        if duplicates.any():
            n_duplicates = duplicates.sum()
            raise AssertionError(
                f"Found {n_duplicates} duplicate pidp-wave combinations. "
                f"This should not occur in clean UKHLS data."
            )
        print("✓ No duplicate pidp-wave combinations found")

        # Save combined data
        # Determine format from file extension, default to 'parquet'
        file_ext = output_filepath.suffix.lower()
        if file_ext == ".parquet":
            save_format = "parquet"
        elif file_ext == ".feather":
            save_format = "feather"
        elif file_ext == ".h5" or file_ext == ".hdf5":
            save_format = "hdf5"
        elif file_ext == ".csv":
            save_format = "csv"
        else:
            save_format = "parquet"  # Default to parquet

        print("\nSaving combined panel data...")
        save_dataframe(df, output_filepath, preferred_format=save_format)
        print(f"✓ Saved full panel to '{output_filepath}'")

        # Clean up intermediate files if requested
        if delete_intermediate_files:
            print("\nCleaning up intermediate wave files...")
            for wave_file in wave_files_processed:
                if wave_file.exists():
                    wave_file.unlink()
            print(f"✓ Deleted {len(wave_files_processed)} intermediate wave files")

    except Exception as e:
        print(f"\n✗ Error combining waves: {e}")
        print(f"Note: Individual wave files are still available in: {DATA_DIR}")
        print("  You can manually combine them or re-run this function.")
        raise

    # Print completion message and important notes
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Combined panel data saved to: {output_filepath}")
    print("=" * 80)


if __name__ == "__main__":
    parser = build_cli_parser(
        "Load raw UKHLS wave files, harmonize them, and combine them into one panel."
    )
    parser.add_argument(
        "--panel_type",
        type=str,
        default="indresp",
        help=(
            "Type of panel data to process. Must match the panel identifier in "
            "wave filenames (e.g., 'indresp', 'hh', 'indresp_ip'). "
            "Default: 'indresp'"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help=(
            "Number of rows to process at a time. If None, processes entire "
            "file in memory. Use for very large files to avoid memory issues. "
            "Recommended: 10000-50000. Default: None"
        ),
    )
    parser.add_argument(
        "--save_intermediate_files",
        action="store_true",
        help=(
            "Keep intermediate wave files after processing. By default, "
            "intermediate files are deleted after successful combination."
        ),
    )
    parser.add_argument(
        "--no_wave_info",
        action="store_true",
        help="Skip displaying wave information at startup",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default=str(INTERIM_DATA_DIR / "ukhls_indresp_combined.parquet"),
        help=(
            "Path to the processed output file. Default: "
            f"{INTERIM_DATA_DIR / 'ukhls_indresp_combined.parquet'}"
        ),
    )

    args = parser.parse_args()
    output_filepath = Path(args.output_filepath)
    log_script_start(
        Path(__file__).name,
        "Load wave-level UKHLS extracts and combine them into a single long panel.",
        inputs={"raw wave directory": UKHLS_PANEL_DIR},
        outputs={
            "combined panel": output_filepath,
            "intermediate directory": INTERIM_DATA_DIR,
        },
        config={"panel type": args.panel_type, "chunksize": args.chunksize},
    )

    # Display wave information unless suppressed
    if not args.no_wave_info:
        print_wave_information()
        print()

    # Load and process panel waves
    load_ukhls_panel_waves(
        panel_type=args.panel_type,
        chunksize=args.chunksize,
        delete_intermediate_files=not args.save_intermediate_files,
        output_filepath=output_filepath,
    )
    log_script_end(
        Path(__file__).name,
        outputs={
            "combined panel": output_filepath,
            "intermediate directory": INTERIM_DATA_DIR,
        },
    )
