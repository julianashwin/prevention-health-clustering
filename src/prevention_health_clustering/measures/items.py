"""Extraction of the 12 SF-12 items from the raw UKHLS indresp waves.

Wave 1 carries the items in the main interview (``a_sf2a``); waves 2-15 use
the self-completion prefix (``b_scsf2a``). The self-completion version is
preferred wherever both exist, because that is what feeds the UKHLS-derived
``sf12pcs_dv``/``sf12mcs_dv``. Ported from the sandbox extraction
(julian_sandbox/01_extract_sf12_items.py) that the PCS construction note is
built on.

Anything read here is licensed UKHLS microdata: outputs live under data/ and
are never committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from prevention_health_clustering.config import UKHLS_PANEL_DIR

WAVES: tuple[str, ...] = tuple("abcdefghijklmno")

# Canonical item stems -> the eight subscales they feed.
SF12_ITEM_STEMS: tuple[str, ...] = (
    "sf1",   # GH1  general health                  -> GH
    "sf2a",  # PF02 moderate activities             -> PF
    "sf2b",  # PF04 climbing several flights        -> PF
    "sf3a",  # RP2  accomplished less (physical)    -> RP
    "sf3b",  # RP3  limited in kind of work         -> RP
    "sf4a",  # RE2  accomplished less (emotional)   -> RE
    "sf4b",  # RE3  did work less carefully         -> RE
    "sf5",   # BP2  pain interfered with work       -> BP
    "sf6a",  # MH3  calm and peaceful               -> MH
    "sf6b",  # VT2  had a lot of energy             -> VT
    "sf6c",  # MH4  downhearted and depressed       -> MH
    "sf7",   # SF2  social activities interfered    -> SF
)

# The wider every-wave health module: the 12 GHQ items (1 = best, 4 = worst),
# the Equality Act impairment list (disdif, asked when health == 1: long-
# standing illness; health == 2 respondents are valid structural zeros), and
# the long-standing illness gate itself.
GHQ_ITEM_STEMS: tuple[str, ...] = tuple(f"scghq{c}" for c in "abcdefghijkl")
DISDIF_STEMS: tuple[str, ...] = tuple(f"disdif{i}" for i in range(1, 13)) + ("disdif96",)
HEALTH_MODULE_STEMS: tuple[str, ...] = GHQ_ITEM_STEMS + DISDIF_STEMS + ("health",)

# Derived scores, the self-completion cross-sectional weight (the wave-1
# weight defines the UK reference population for the re-scored variants), and
# the every-wave health module used by the two-dimensional GRM.
DEFAULT_EXTRA_STEMS: tuple[str, ...] = (
    "sf12pcs_dv", "sf12mcs_dv", "indscus_xw",
) + HEALTH_MODULE_STEMS


def wave_item_columns(
    wave: str, header: Sequence[str], extra_stems: Iterable[str] = DEFAULT_EXTRA_STEMS
) -> dict[str, str]:
    """Map canonical stem -> the column name this wave actually carries."""
    present = set(header)
    mapping: dict[str, str] = {}
    for stem in SF12_ITEM_STEMS:
        for candidate in (f"{wave}_sc{stem}", f"{wave}_{stem}"):
            if candidate in present:
                mapping[stem] = candidate
                break
    for stem in extra_stems:
        if f"{wave}_{stem}" in present:
            mapping[stem] = f"{wave}_{stem}"
    for age_col in (f"{wave}_dvage", f"{wave}_age_dv"):
        if age_col in present:
            mapping["age"] = age_col
            break
    return mapping


def extract_sf12_items(
    raw_dir: Path | None = None,
    *,
    waves: Sequence[str] = WAVES,
    extra_stems: Iterable[str] = DEFAULT_EXTRA_STEMS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Long person-wave frame of the 12 items plus derived scores and weight.

    UKHLS negative codes (all forms of missing/inapplicable/proxy) become NaN.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else UKHLS_PANEL_DIR
    frames = []
    for wave_number, wave in enumerate(waves, start=1):
        path = raw_dir / f"{wave}_indresp.tab"
        header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        mapping = wave_item_columns(wave, header, extra_stems)
        df = pd.read_csv(
            path, sep="\t", usecols=["pidp"] + list(mapping.values()),
            low_memory=False,
        ).rename(columns={v: k for k, v in mapping.items()})
        df["wave"] = wave_number
        df["wave_letter"] = wave
        if verbose:
            n_items = sum(s in df.columns for s in SF12_ITEM_STEMS)
            print(f"  wave {wave} (n={len(df):,}): {n_items}/12 items")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)

    for col in panel.columns:
        if col in {"pidp", "wave", "wave_letter"}:
            continue
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        panel.loc[panel[col] < 0, col] = np.nan
    return panel


__all__ = [
    "DEFAULT_EXTRA_STEMS",
    "DISDIF_STEMS",
    "GHQ_ITEM_STEMS",
    "HEALTH_MODULE_STEMS",
    "SF12_ITEM_STEMS",
    "WAVES",
    "extract_sf12_items",
    "wave_item_columns",
]
