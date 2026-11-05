"""Predicting the type on PSID: a health reading against the observables, K-means only.

descriptives/43_paper_type_from_initial.py restricted to its K-means typology: the partial K-means
types fitted with each person's first two rows held out (22_psid_kmeans.py, theta_first2held; people
with at least five rows on the PSID contract), predicted from what is known at those two rows:
y0 (theta at the first row, t-2) and y1 (at the second, t-1), each with its age interaction, and
the observables. Multinomial logit on a quadratic in age at the first row plus the set, people split
70/30 on the UKHLS hash, scored on the 30 with 43's own score() (accuracy, balanced accuracy, macro
AUC). Also writes the per-person types with the observables for panel (d) of Figure 13 and for the
targeting exercise in 47_psid_prevention.py.

Outputs, artifacts/descriptives/psid/: type_from_initial.csv, type_first2_people.parquet
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, OBS, REGETH, EDUC_COLS, contract, hash_test, load_labels, obs_dummies, observables  # noqa: E402

M43 = importlib.import_module("43_paper_type_from_initial")      # score() only; nothing UKHLS is read at import
KHELD = 2


def first_rows() -> pd.DataFrame:
    """Copy of 43's first_rows on the PSID contract: people with >= 5 rows, the two held rows and observables."""
    long = contract()
    n = long.groupby("pidp")["age"].transform("size")
    sub = long[n >= KHELD + 3].reset_index(drop=True)
    first = sub.groupby("pidp", as_index=False).first()[["pidp", "age", "theta"]].rename(columns={"age": "age0", "theta": "y0"})
    second = sub.groupby("pidp", as_index=False).nth(1)[["pidp", "age", "theta"]].rename(columns={"age": "age1", "theta": "y1"})
    first = first.merge(second, on="pidp")
    per = observables(first["pidp"].unique())
    first = first.join(per[["educ_group", "income_rank_mean", "female", "region", "race_group"]], on="pidp").join(obs_dummies(per), on="pidp")
    first["a0"] = (first["age0"] - 55) / 10
    return first


def typed() -> pd.DataFrame:
    first = first_rows()
    L = load_labels("theta_first2held")
    df = first.merge(pd.DataFrame({"pidp": L.index, "type": L.to_numpy()}), on="pidp")
    df = df.dropna(subset=["y0", "y1", "income_rank_mean"]).reset_index(drop=True)
    df["y0a"] = df["y0"] * df["a0"]; df["y1a"] = df["y1"] * df["a0"]
    return df


def main() -> int:
    df = typed()
    test = hash_test(df["pidp"])
    shares = df["type"].value_counts(normalize=True).sort_index()
    majority = float(df.loc[test, "type"].value_counts(normalize=True).max())
    print(f"K-means, first two rows held out: {len(df):,} people, type shares {np.round(shares.to_numpy(), 3)}, majority {majority:.3f}")
    SETS = {"age(0)": [], "+ education": EDUC_COLS, "+ income rank": ["income_rank_mean"], "+ education, income, sex": OBS,
            "+ education, income, sex, region, ethnicity": OBS + REGETH, "+ theta(1)": ["y1", "y1a"], "+ theta(0)": ["y0", "y0a"],
            "+ theta(0), theta(1)": ["y0", "y0a", "y1", "y1a"], "+ theta(0), education, income, sex": ["y0", "y0a"] + OBS}
    rows = []
    for name, cols in SETS.items():
        r = M43.score(df, cols, test); r.pop("probs")
        rows.append({"typology": "kmeans", "predictors": name, "n": len(df), "n_test": int(test.sum()), "majority": majority, **r})
        print(f"  {name:45s} accuracy {r['accuracy']:.3f}  balanced {r['balanced_accuracy']:.3f}  macro AUC {r['auc']:.3f}")
    pd.DataFrame(rows).to_csv(DESC / "type_from_initial.csv", index=False)
    df.to_parquet(DESC / "type_first2_people.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
