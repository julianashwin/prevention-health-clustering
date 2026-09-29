"""Build the Rockwood-style frailty index and its log, as comparison rulers.

The index is defined in ``measures/health.py`` (``build_frailty_index``): the
share of 31 equal-weight deficits (six SF-12 physical items graded 0-1, the
eight physical impairment areas, and the ever-diagnosed conditions), higher =
frailer; ``log_frailty`` is log(frailty + c) with c = 1/31, one whole deficit, so person-waves with no deficit stay in at the floor;
``log_frailty_pos`` is the log on positive frailty only, as in Hosseini, Kopecky
and Zhao (2022), kept for reference. The script prints how the young-age variance
of the log moves if c is 1/248 (half the smallest positive value) instead.

Neither is the paper's measure. They are written beside it so every paper
script can be run on them with ``--frailty`` and the health contract carries
them as auxiliary columns (07_build_health_contract.py), without changing its
rows or metrics. Runs after 04_build_health.py.

Output (gitignored): data/processed/measures/frailty_index.parquet
    pidp, wave, age, frailty, log_frailty, log_frailty_pos
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.health import FRAILTY_SHIFT, build_frailty_index


def main() -> int:
    ensure_runtime_directories()
    md = PROCESSED_DATA_DIR / "measures"
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet")
    chronic = pd.read_parquet(md / "chronic_conditions.parquet")
    fr = build_frailty_index(items, chronic)
    n_def = fr.attrs["n_deficits"]
    fr = fr.dropna(subset=["frailty"]).reset_index(drop=True)
    fr["age"] = fr["age"].astype(int)
    hm = pd.read_parquet(md / "health_measure.parquet", columns=["pidp", "wave", "h", "theta", "fi10"])
    both = hm.merge(fr, on=["pidp", "wave"])
    print(f"frailty index: {n_def} deficits, {len(fr):,} person-waves scored; {len(both):,} also carry the health measure "
          f"({len(both) / len(hm):.1%} of its rows)")
    print(f"  zero frailty {(fr['frailty'] == 0).mean():.1%} of person-waves; mean {fr['frailty'].mean():.3f}; "
          f"shift c = {FRAILTY_SHIFT:.5f} (zeros at log {np.log(FRAILTY_SHIFT):.2f}); positive-only log on {fr['log_frailty_pos'].notna().mean():.1%}")
    # sensitivity of the log's age profile of variance to the shift
    for lab, c in (("c = 1/31", FRAILTY_SHIFT), ("c = 1/248", 1 / 248)):
        g = pd.DataFrame({"age": fr["age"], "y": np.log(fr["frailty"] + c)}).groupby("age")["y"].var()
        print(f"  log(frailty + {lab}): variance at 25 {g.loc[23:27].mean():.3f}, 45 {g.loc[43:47].mean():.3f}, 65 {g.loc[63:67].mean():.3f}, "
              f"85 {g.loc[83:87].mean():.3f}; peak at {int(g.loc[25:88].idxmax())}")
    for v in ("h", "theta", "fi10"):
        print(f"  Spearman frailty vs {v}: {spearmanr(both['frailty'], both[v]).statistic:+.3f}")
    fr[["pidp", "wave", "age", "frailty", "log_frailty", "log_frailty_pos"]].to_parquet(md / "frailty_index.parquet", index=False)
    print(f"wrote {md / 'frailty_index.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
