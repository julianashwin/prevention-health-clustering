"""Simple linear twins of the leading banks, on exactly the same testlet codes.

For each bank two transparent indices, both higher = better health:
  FS   the Pearson one-factor score: each testlet standardised, weighted by
       loading over uniqueness (07_limitation_weights.R), re-standardised
  SUM  the plain sum of the testlet codes, equal weights, no standardisation

They answer the question the graded-response score raises: does a weighted sum
of the same answers reach the same conclusions? Written to
data/processed/baseline_measures/simple_twins.parquet and carried through the
criteria by 05_master_criteria.py and 06_master_ident.R.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.config import PROCESSED_DATA_DIR

MEASD = PROCESSED_DATA_DIR / "measures"
OUT = PROCESSED_DATA_DIR / "baseline_measures"
SF = ["GH", "PF", "RP", "BP"]
# bank -> testlets, for the banks the note shortlists
TWIN_BANKS = {
    "P-FUNC": ["FUNC"],
    "P-LIM": ["LIM_PF", "LIM_SC"],
    "P-LIM3": ["LIM_FL", "LIM_SELF", "LIM_SENS"],
    "P-LIM3+CC": ["LIM_FL", "LIM_SELF", "LIM_SENS", "COND"],
}


def main() -> int:
    codes = pd.read_parquet(MEASD / "limitation_bank_items.parquet")
    fw = pd.read_csv(OUT / "limitation_factor_weights.csv")
    out = codes[["pidp", "wave", "age"]].copy()
    rows = []
    for bank, lims in TWIN_BANKS.items():
        names = SF + lims
        X = codes[names]
        keep = X.notna().all(axis=1).to_numpy()
        Z = ((X[keep] - X[keep].mean()) / X[keep].std()).to_numpy()
        w = fw[(fw["bank"] == bank) & (fw["method"] == "pearson_factor")].set_index("item").loc[names, "weight"]
        fs = Z @ w.to_numpy()
        col = bank.replace("+", "").replace("-", "").lower()
        out[f"fs_{col}"] = np.where(keep, np.nan, np.nan)
        out.loc[keep, f"fs_{col}"] = (fs - fs.mean()) / fs.std()
        out[f"sum_{col}"] = np.where(keep, np.nan, np.nan)
        out.loc[keep, f"sum_{col}"] = X[keep].sum(axis=1).to_numpy()
        rows.append({"bank": bank, "n": int(keep.sum()), "items": len(names),
                     "weights": ", ".join(f"{n} {v:.2f}" for n, v in w.items()),
                     "corr_fs_sum": float(np.corrcoef(out.loc[keep, f"fs_{col}"], out.loc[keep, f"sum_{col}"])[0, 1]),
                     "distinct_sum": int(out.loc[keep, f"sum_{col}"].nunique())})
    out.to_parquet(OUT / "simple_twins.parquet", index=False)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print(f"\nwrote simple_twins.parquet: {len(out):,} rows, {len(TWIN_BANKS) * 2} indices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
