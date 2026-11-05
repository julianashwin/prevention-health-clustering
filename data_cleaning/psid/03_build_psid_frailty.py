"""Build the PSID frailty index and its log, the way 04b_build_frailty.py builds the UKHLS one.

The index follows Hosseini, Kopecky and Zhao (2022, Appendix Table 1), who build it on PSID and
whom the paper cites for log frailty: the share of 28 equal-weight 0/1 deficits, higher = frailer.

    13 difficulties with ADLs and IADLs because of health: eating, dressing, getting in or out of
       a bed or chair, using the toilet, bathing, walking, getting outside, using the telephone,
       managing money, shopping, preparing meals, heavy housework, light housework
    12 ever-diagnosed conditions, kept once reported (the *_ever_abs columns, the UKHLS rule):
       high blood pressure, diabetes, cancer, lung disease, heart disease, heart attack, stroke,
       arthritis, asthma, memory loss, a psychological problem, another serious chronic condition
     3 risk factors: obese (BMI 30 or more), has ever smoked, smokes now

A person-wave is scored only when every deficit asked that wave is observed, as in the UKHLS
code. "Another serious chronic condition" is first asked in 2005, so in 2003 the index is the
share of the 27 deficits asked; the script prints how little that moves the index.

``log_frailty`` is log(frailty + 1/28), one whole deficit, as the UKHLS code uses 1/31.
``log_frailty_pos`` is the log on positive frailty only, as Hosseini, Kopecky and Zhao do.

The script compares the index with their Table 1 on their sample (heads and spouses aged 25 and
over, 2003-2017) and with theta and h from 02_build_psid_health.py.

Outputs (gitignored), data/processed/psid/measures/:
    psid_frailty_index.parquet          pid, wave, age, frailty, log_frailty, log_frailty_pos, n_deficits
    psid_frailty_hkz_benchmark.csv      the index beside their Table 1 (the 85+ zero share they give only
                                        as "under 1%", left blank)

    python data_cleaning/psid/03_build_psid_frailty.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FIRST_WAVE = 2003
AGE_RANGE = (20, 90)
DEFICITS: dict[str, str] = {
    "eating": "adl_eat_diff", "dressing": "adl_dress_diff", "bed_or_chair": "adl_bed_diff",
    "toilet": "adl_toilet_diff", "bathing": "adl_bath_diff", "walking": "adl_walk_diff",
    "getting_outside": "adl_outside_diff", "telephone": "iadl_phone_diff", "money": "iadl_money_diff",
    "shopping": "iadl_shop_diff", "meals": "iadl_meals_diff", "heavy_housework": "iadl_heavyhw_diff",
    "light_housework": "iadl_lighthw_diff",
    "high_blood_pressure": "hypertension_ever_abs", "diabetes": "diabetes_ever_abs",
    "cancer": "cancer_ever_abs", "lung_disease": "lung_disease_ever_abs",
    "heart_disease": "heart_disease_ever_abs", "heart_attack": "heart_attack_ever_abs",
    "stroke": "stroke_ever_abs", "arthritis": "arthritis_ever_abs", "asthma": "asthma_ever_abs",
    "memory_loss": "memory_loss_ever_abs", "psychological": "psych_ever_abs",
    "other_chronic": "other_chronic_ever_abs",
    "obese": "bmi", "ever_smoked": "ever_smoked", "smokes_now": "smoke_now",
}
NOT_ASKED = {2003: ("other_chronic",)}          # wave -> deficits PSID did not ask that wave
FRAILTY_SHIFT = 1 / len(DEFICITS)
# Hosseini, Kopecky and Zhao (2022), Appendix Table 1: mean and sd of frailty by age group
HKZ = {"all": (0.11, 0.11), "25-49": (0.08, 0.09), "50-74": (0.15, 0.12), "75+": (0.25, 0.16)}
HKZ_OTHER = "median 0.07, 95th percentile 0.36; no deficit: 15% overall, 24% at 25-34, under 1% at 85+"


def deficit_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    D = pd.DataFrame(index=panel.index)
    for name, col in DEFICITS.items():
        x = panel[col]
        D[name] = (x >= 30).astype(float).where(x.notna()) if name == "obese" else x
    return D


def build_frailty(panel: pd.DataFrame) -> pd.DataFrame:
    D = deficit_matrix(panel)
    asked = pd.DataFrame(True, index=D.index, columns=D.columns)
    for wave, names in NOT_ASKED.items():
        asked.loc[panel["wave"] == wave, list(names)] = False
    complete = (D.notna() | ~asked).all(axis=1)
    n_def = asked.sum(axis=1)
    fr = D.where(asked).sum(axis=1).div(n_def).where(complete)
    out = panel[["pid", "wave", "age"]].copy()
    out["frailty"] = fr
    out["log_frailty"] = np.log(fr + FRAILTY_SHIFT)
    out["log_frailty_pos"] = np.log(fr.where(fr > 0))
    out["n_deficits"] = n_def
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", default="data/processed/psid/psid_panel.parquet")
    ap.add_argument("--measures", default="data/processed/psid/measures")
    args = ap.parse_args()
    md = pathlib.Path(args.measures)

    cols = ["pid", "wave", "age", *sorted(set(DEFICITS.values()))]
    panel = pd.read_parquet(args.panel, columns=cols)
    panel = panel[(panel["wave"] >= FIRST_WAVE) & (panel["age"] >= AGE_RANGE[0])].reset_index(drop=True)
    panel["age"] = panel["age"].astype(int)
    fr_all = build_frailty(panel)

    # ---------------- against Hosseini, Kopecky and Zhao (2022) -------------
    hk = fr_all[fr_all["wave"].between(2003, 2017) & (fr_all["age"] >= 25)]
    hs = hk.dropna(subset=["frailty"])
    print(f"Hosseini, Kopecky and Zhao sample (heads and spouses 25+, 2003-2017): {len(hk):,} person-waves, "
          f"{len(hs):,} scored ({len(hs) / len(hk):.1%}) on {hs['pid'].nunique():,} people; theirs 96,759 on 19,996")
    bands = {"all": hs, "25-49": hs[hs["age"].between(25, 49)], "50-74": hs[hs["age"].between(50, 74)],
             "75+": hs[hs["age"] >= 75]}
    print(f"  {'ages':6s} {'mean':>6s} {'theirs':>7s} {'sd':>6s} {'theirs':>7s}")
    bench = []
    for k, d in bands.items():
        print(f"  {k:6s} {d['frailty'].mean():6.3f} {HKZ[k][0]:7.2f} {d['frailty'].std():6.3f} {HKZ[k][1]:7.2f}")
        bench += [{"statistic": f"mean, ages {k}", "psid": d["frailty"].mean(), "hkz": HKZ[k][0]},
                  {"statistic": f"sd, ages {k}", "psid": d["frailty"].std(), "hkz": HKZ[k][1]}]
    zero = {"all": hs, "25-34": hs[hs["age"].between(25, 34)], "85+": hs[hs["age"] >= 85]}
    bench += [{"statistic": "median", "psid": hs["frailty"].median(), "hkz": 0.07},
              {"statistic": "95th percentile", "psid": hs["frailty"].quantile(0.95), "hkz": 0.36},
              *[{"statistic": f"share with no deficit, ages {k}", "psid": (d["frailty"] == 0).mean(), "hkz": v}
                for (k, d), v in zip(zero.items(), (0.15, 0.24, np.nan))],
              {"statistic": "person-waves", "psid": len(hs), "hkz": 96759},
              {"statistic": "people", "psid": hs["pid"].nunique(), "hkz": 19996}]
    print(f"  median {hs['frailty'].median():.3f}, 95th percentile {hs['frailty'].quantile(0.95):.3f}; no deficit "
          f"{(hs['frailty'] == 0).mean():.0%} overall, {(hs.loc[hs['age'].between(25, 34), 'frailty'] == 0).mean():.0%} "
          f"at 25-34, {(hs.loc[hs['age'] >= 85, 'frailty'] == 0).mean():.1%} at 85+")
    print(f"  theirs: {HKZ_OTHER}")
    pd.DataFrame(bench).to_csv(md / "psid_frailty_hkz_benchmark.csv", index=False)

    # ---------------- the 2003 rule ---------------------------------------
    w05 = panel[panel["wave"] == 2005]
    D05 = deficit_matrix(w05)
    ok = D05.notna().all(axis=1)
    f28, f27 = D05[ok].mean(axis=1), D05[ok].drop(columns="other_chronic").mean(axis=1)
    print(f"\n2003 asks 27 of the 28 deficits. In 2005, scoring on those 27 instead of all 28 moves the mean "
          f"{f27.mean() - f28.mean():+.4f} ({f28.mean():.3f} on 28) and keeps the ranks (Spearman "
          f"{spearmanr(f28, f27).statistic:.3f})")

    # ---------------- the paper's window and the comparison with theta -----
    fr = fr_all[fr_all["age"].between(*AGE_RANGE)].dropna(subset=["frailty"]).reset_index(drop=True)
    n_win = int(fr_all["age"].between(*AGE_RANGE).sum())
    print(f"\nages {AGE_RANGE[0]}-{AGE_RANGE[1]}, 2003-2023: {len(fr):,} person-waves scored ({len(fr) / n_win:.1%}); "
          f"no deficit {(fr['frailty'] == 0).mean():.1%}; mean {fr['frailty'].mean():.3f}; shift 1/28 puts the "
          f"zeros at log {np.log(FRAILTY_SHIFT):.2f}; positive-only log on {fr['log_frailty_pos'].notna().mean():.1%}")
    for lab, c in (("c = 1/28", FRAILTY_SHIFT), ("c = 1/224", 1 / 224)):
        g = pd.DataFrame({"age": fr["age"], "y": np.log(fr["frailty"] + c)}).groupby("age")["y"].var()
        print(f"  log(frailty + {lab}): variance at 25 {g.loc[23:27].mean():.3f}, 45 {g.loc[43:47].mean():.3f}, "
              f"65 {g.loc[63:67].mean():.3f}, 85 {g.loc[83:87].mean():.3f}; peak at {int(g.loc[25:88].idxmax())}")
    hm_path = md / "psid_health_measure.parquet"
    if hm_path.exists():
        hm = pd.read_parquet(hm_path, columns=["pid", "wave", "theta", "h"])
        both = hm.merge(fr, on=["pid", "wave"])
        print(f"  {len(both):,} rows also carry theta ({len(both) / len(hm):.1%} of its rows)")
        for v in ("h", "theta"):
            print(f"  Spearman frailty vs {v}: {spearmanr(both['frailty'], both[v]).statistic:+.3f}")

    out = fr[["pid", "wave", "age", "frailty", "log_frailty", "log_frailty_pos", "n_deficits"]]
    out.to_parquet(md / "psid_frailty_index.parquet", index=False)
    print(f"wrote {md / 'psid_frailty_index.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
