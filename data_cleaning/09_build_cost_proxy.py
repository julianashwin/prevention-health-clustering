"""Build the UKHLS cost index (plan stages 1 and 2).

Stage 1 is the flat-rate index: every admission priced at the average
non-elective spell plus excess bed days, out-patient and GP contacts at
their national average unit costs, maternity separated.

Stage 2 weights each admission by the cost group of the condition it was
attributed to, where UKHLS records one. Attribution covers about a third of
admissions and two-thirds of nights. Unattributed admissions take the
case-mix-weighted average of the attributed ones WITHIN the same decile of
physical health, not globally: a global imputation would push every
unattributed admission toward the population mean and flatten the very
gradient the index exists to measure.

Output: data/processed/measures/cost_index.parquet, one row per person-wave
with the flat and condition-weighted variants and their components.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR, UKHLS_PANEL_DIR)
from prevention_health_clustering.measures.cost import (
    CONDITION_GROUP, UnitCosts, build_cost_index)

WAVES = "ghijklmno"          # waves 7-15 carry the utilisation block


def load_attribution() -> pd.DataFrame:
    """Nights attributed to each condition, per person-wave, long format."""
    rows = []
    for wi, w in enumerate(WAVES, start=7):
        path = UKHLS_PANEL_DIR / f"{w}_indresp.tab"
        header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        cols = {}
        for c in header:
            for pre in (f"{w}_hospdcp", f"{w}_hospdc"):
                if c.startswith(pre) and c[len(pre):].isdigit():
                    cols[c] = int(c[len(pre):])
                    break
        if not cols:
            continue
        d = pd.read_csv(path, sep="\t", usecols=["pidp"] + list(cols),
                        low_memory=False)
        long = d.melt(id_vars="pidp", var_name="col", value_name="nights")
        long["nights"] = pd.to_numeric(long["nights"], errors="coerce")
        long = long[long["nights"] > 0]
        long["code"] = long["col"].map(cols)
        long["wave"] = wi
        rows.append(long[["pidp", "wave", "code", "nights"]])
    out = pd.concat(rows, ignore_index=True)
    out["group"] = out["code"].map(CONDITION_GROUP).fillna("other")
    return out


def main() -> int:
    ensure_runtime_directories()
    costs = UnitCosts()
    util = pd.read_parquet(INTERIM_DATA_DIR / "utilisation_long.parquet")
    panel = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
                            columns=["pidp", "wave", "age", "theta_phys_full"])
    d = panel.merge(util[["pidp", "wave", "hosp", "hospd", "hospch",
                          "hl2gp", "hl2hop"]], on=["pidp", "wave"], how="inner")
    d = d[d["hosp"].notna()].reset_index(drop=True)
    print(f"utilisation sample: {len(d):,} person-waves, "
          f"{d['pidp'].nunique():,} people, waves {d['wave'].min()}-{d['wave'].max()}")

    # ---- stage 1: flat rate ------------------------------------------------
    flat = build_cost_index(d, costs, spell_rule="one")
    for c in flat.columns:
        d[f"flat_{c}"] = flat[c]
    print(f"\nstage 1 (flat): mean total £{d['flat_cost_total'].mean():,.0f}; "
          f"in-patient £{d['flat_cost_inpatient'].mean():,.0f}, "
          f"out-patient £{d['flat_cost_outpatient'].mean():,.0f}, "
          f"GP £{d['flat_cost_gp'].mean():,.0f}")

    # ---- stage 2: condition weighting --------------------------------------
    att = load_attribution()
    print(f"\nattribution: {len(att):,} condition-nights records; "
          f"{att.groupby(['pidp','wave']).ngroups:,} person-waves with any")
    # a person-wave's weight is its nights-weighted average group multiplier
    att["w"] = att["group"].map(costs.condition_weight).fillna(1.0)
    agg = att.groupby(["pidp", "wave"]).apply(
        lambda g: np.average(g["w"], weights=g["nights"]), include_groups=False)
    agg.name = "cond_weight"
    d = d.merge(agg.reset_index(), on=["pidp", "wave"], how="left")
    admitted = d["hosp"] == 1
    print(f"  admissions with an attributed weight: "
          f"{(admitted & d['cond_weight'].notna()).sum():,} of "
          f"{admitted.sum():,} ({(d.loc[admitted,'cond_weight'].notna().mean()):.1%})")

    # impute the missing weights within decile of physical health
    d["hdec"] = pd.qcut(d["theta_phys_full"], 10, labels=False, duplicates="drop")
    within = d[admitted].groupby("hdec")["cond_weight"].mean()
    overall = d.loc[admitted, "cond_weight"].mean()
    d["cond_weight_filled"] = d["cond_weight"].fillna(
        d["hdec"].map(within)).fillna(overall)
    print("  imputed weight by health decile (1 = least healthy): "
          + "  ".join(f"{within.get(k, np.nan):.2f}" for k in range(10)))

    weighted = build_cost_index(d, costs, spell_rule="one",
                                condition_weights=d["cond_weight_filled"])
    for c in weighted.columns:
        d[f"wtd_{c}"] = weighted[c]
    print(f"\nstage 2 (condition-weighted): mean total "
          f"£{d['wtd_cost_total'].mean():,.0f} "
          f"(flat £{d['flat_cost_total'].mean():,.0f})")

    keep = (["pidp", "wave", "age", "theta_phys_full", "hosp", "hospd",
             "hospch", "hl2gp", "hl2hop", "cond_weight", "cond_weight_filled"]
            + [c for c in d.columns if c.startswith(("flat_", "wtd_"))])
    out_path = PROCESSED_DATA_DIR / "measures" / "cost_index.parquet"
    d[keep].to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(d):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
