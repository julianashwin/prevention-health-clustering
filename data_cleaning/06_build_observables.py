"""Socio-economic observables for the paper's observational comparison.

Reads the raw UKHLS tab files directly, because the processed panel carries
personal gross income and highest qualification but not household income.

Person-wave (data/processed/measures/observables.parquet):
  hidp             household identifier that wave
  hiqual_dv        highest qualification, UKHLS coding
                   1 degree, 2 other higher, 3 A-level etc, 4 GCSE etc,
                   5 other qualification, 9 none
  hh_net_income    total household net income, month before interview
                   (fihhmnnet1_dv, no deductions)
  eq_scale         modified OECD equivalence scale (ieqmoecd_dv)
  eq_income        hh_net_income / eq_scale
  income_rank      percentile rank of eq_income among every adult respondent
                   in the same wave, so no deflation is needed

Person level (observables_person.parquet):
  educ             highest qualification ever recorded, taken over waves
                   observed at age 25 or over where any exist (earlier waves
                   otherwise, so the youngest are not dropped); coded as the
                   best of the UKHLS codes with 9 ranked below 5
  educ_group       4 groups: degree or higher (1-2: degree, other higher qualification),
                   A-level (3), GCSE (4), other/none (5, 9)
  income_rank_mean mean of income_rank over the person's waves with a value
  n_income_waves   how many waves that mean rests on

Negative UKHLS codes are missing throughout.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.config import PROCESSED_DATA_DIR, UKHLS_PANEL_DIR
from prevention_health_clustering.data.io import UKHLS_WAVE_DATES, _read_tab_subset

EDUC_GROUP = {1: "degree or higher", 2: "degree or higher", 3: "A-level", 4: "GCSE", 5: "other/none", 9: "other/none"}
EDUC_RANK = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 9: 5}     # lower is better


def clean(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0)


def main() -> int:
    rows = []
    for letter, info in UKHLS_WAVE_DATES.items():
        w = info["wave_number"]
        ind = _read_tab_subset(UKHLS_PANEL_DIR / f"{letter}_indresp.tab",
                               ["pidp", f"{letter}_hidp", f"{letter}_hiqual_dv"])
        hh = _read_tab_subset(UKHLS_PANEL_DIR / f"{letter}_hhresp.tab",
                              [f"{letter}_hidp", f"{letter}_fihhmnnet1_dv", f"{letter}_ieqmoecd_dv"])
        if ind.empty:
            print(f"wave {w} ({letter}): no indresp file", flush=True)
            continue
        strip = lambda c: c[len(letter) + 1:] if c.startswith(f"{letter}_") else c  # noqa: E731
        ind.columns = [strip(c) for c in ind.columns]
        hh.columns = [strip(c) for c in hh.columns]
        d = ind.merge(hh.drop_duplicates("hidp"), on="hidp", how="left").assign(wave=w)
        d["hiqual_dv"] = clean(d["hiqual_dv"])
        d["hh_net_income"] = clean(d["fihhmnnet1_dv"])
        d["eq_scale"] = clean(d["ieqmoecd_dv"])
        d["eq_income"] = d["hh_net_income"] / d["eq_scale"]
        d["income_rank"] = d["eq_income"].rank(pct=True)
        rows.append(d[["pidp", "wave", "hidp", "hiqual_dv", "hh_net_income", "eq_scale",
                       "eq_income", "income_rank"]])
        print(f"wave {w:2d} ({letter}): {len(d):,} adults, income valid "
              f"{d['eq_income'].notna().mean():.1%}, qualification valid "
              f"{d['hiqual_dv'].notna().mean():.1%}, median eq. income "
              f"£{d['eq_income'].median():,.0f}", flush=True)
    pw = pd.concat(rows, ignore_index=True)

    age = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv",
                      usecols=["pidp", "wave", "age"])
    pw = pw.merge(age, on=["pidp", "wave"], how="left")
    q = pw.dropna(subset=["hiqual_dv"]).assign(r=lambda x: x["hiqual_dv"].map(EDUC_RANK))
    adult = q[q["age"] >= 25]
    best = adult.groupby("pidp")["r"].min()
    best_any = q.groupby("pidp")["r"].min()
    educ_rank = best_any.to_frame("r_any").join(best.rename("r_adult"), how="left")
    educ_rank["r"] = educ_rank["r_adult"].fillna(educ_rank["r_any"]).astype(int)
    inv = {v: k for k, v in EDUC_RANK.items()}
    person = pd.DataFrame({"educ": educ_rank["r"].map(inv)})
    person["educ_group"] = person["educ"].map(EDUC_GROUP)
    inc = pw.dropna(subset=["income_rank"]).groupby("pidp")["income_rank"].agg(["mean", "size"])
    person = person.join(inc.rename(columns={"mean": "income_rank_mean", "size": "n_income_waves"}),
                         how="outer").rename_axis("pidp").reset_index()

    out = PROCESSED_DATA_DIR / "measures"
    pw.to_parquet(out / "observables.parquet", index=False)
    person.to_parquet(out / "observables_person.parquet", index=False)
    print(f"\nwrote observables.parquet ({len(pw):,} person-waves) and "
          f"observables_person.parquet ({len(person):,} people)")
    print("education groups:\n", person["educ_group"].value_counts(dropna=False).to_string())
    print("income rank mean:\n", person["income_rank_mean"].describe().round(3).to_string())
    print("share of people whose recorded qualification ever changes across waves: "
          f"{(q.groupby('pidp')['r'].nunique() > 1).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
