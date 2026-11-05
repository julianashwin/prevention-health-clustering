"""Predicting ahead on PSID: theta and the cost proxy at t+1 and t+2 from what is known at t.

descriptives/33_paper_prediction.py without the Bayesian fits: the partial K-means typology fitted
with each person's last two rows held out (22_psid_kmeans.py, theta_last2held; people with at least
five rows on the PSID contract) is the only typology. t is each person's last fitted row; outcomes are
at the next wave (t+1, two years on) and the one after (t+2, four years on), so a PSID "wave ahead"
is two years against one in the UKHLS. Predictor sets, each with a quadratic in age (as the UKHLS):
  age; + observables (education in four groups, income rank, sex); + observables, region (four census
  regions), ethnic group (five US groups); + K-means type; + theta(t); + theta(t), theta(t-1) (the
  previous wave, two years earlier); + K-means forecast (the type's mean theta at the outcome's age);
  + K-means type, theta(t); + all (K-means type, theta(t), observables).
People split 70/30 on the UKHLS's fixed hash; R-squared on the 30 (M33.score_sets, M33.scores_by_age).
The held-out death comparison (death within two years of the last interview against leaving alive or
surviving, predicted from t) goes to prediction_death_heldout.csv for 41_psid_mortality_prediction.py.

Outputs, artifacts/descriptives/psid/: prediction.csv, prediction_by_age.csv, prediction_death_heldout.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import (DESC, EDUC_COLS, K, LAST_WAVE, OBS, REGETH, WAVE_STEP, contract, cost_rows, load_labels,  # noqa: E402
                          load_trajectories, obs_dummies, observables, panel, scores)

M33 = importlib.import_module("33_paper_prediction")      # score_sets, scores_by_age, logistic_fit, auc (no UKHLS data read)
MIN_OBS = 5
COST_CAP = 0.99
BY_AGE_SETS = ["age", "+ observables", "+ theta(t)", "+ theta(t), theta(t-1)", "+ K-means forecast"]


def death_wave() -> pd.Series:
    """The wave of the last panel row of each person who died within two years of it."""
    p = panel(["mort_event"])
    return p[p["mort_event"] == 1].groupby("pidp")["wave"].max()


def build() -> tuple[pd.DataFrame, dict]:
    long = contract()
    n = long.groupby("pidp")["age"].transform("size")
    sub = long[n >= MIN_OBS].reset_index(drop=True)
    pos = sub.groupby("pidp").cumcount(); nn = sub.groupby("pidp")["age"].transform("size")
    at = sub[pos == nn - 3][["pidp", "wave", "age", "theta"]].reset_index(drop=True)            # t: the last fitted row
    hs = sub[pos == nn - 2].set_index("pidp"); he = sub[pos == nn - 1].set_index("pidp")
    at["age_hs"] = at["pidp"].map(hs["age"]).astype(float); at["age_he"] = at["pidp"].map(he["age"]).astype(float)
    at["wave_he"] = at["pidp"].map(he["wave"])
    L = load_labels("theta_last2held")
    at["km_type"] = at["pidp"].map(L)
    for c in range(K - 1):
        at[f"km_{c + 1}"] = (at["km_type"] == c).astype(float).where(at["km_type"].notna())
    traj = load_trajectories("theta_last2held")
    prof = traj.pivot(index="age", columns="cluster", values="mean").reindex(range(20, 90)).interpolate(limit_direction="both")
    meas = scores()
    cost = cost_rows()
    cost["cost_next"] = cost["cost"].clip(upper=cost["cost"].quantile(COST_CAP))
    per = observables(at["pidp"].unique())
    per = per[["income_rank_mean", "female", "educ_group"]].join(obs_dummies(per))
    dw = death_wave()
    frames = {}
    for k, target in ((1, at["age_hs"].to_numpy()), (2, at["age_he"].to_numpy()), (3, at["age_he"].to_numpy() + 2.0)):
        f = at.copy()
        f["km_fc"] = M33.kmeans_forecast(f, prof, target)
        nxt = meas[["pidp", "wave", "theta"]].rename(columns={"theta": "y_next"}); nxt["wave"] -= WAVE_STEP * k
        lag = meas[["pidp", "wave", "theta"]].rename(columns={"theta": "y_lag"}); lag["wave"] += WAVE_STEP
        c = cost[["pidp", "wave", "cost_next"]].copy(); c["wave"] -= WAVE_STEP * k
        f = f.merge(nxt, on=["pidp", "wave"], how="left").merge(lag, on=["pidp", "wave"], how="left").merge(c, on=["pidp", "wave"], how="left")
        f["death_wave"] = f["pidp"].map(dw)
        f["died_next"] = np.where(f["wave_he"] >= LAST_WAVE, np.nan, (f["death_wave"] == f["wave_he"]).astype(float))
        f = f.join(per, on="pidp")
        f["a"] = (f["age"] - 55) / 10
        frames[k] = f
    SETS = {
        "age": [],
        "+ observables": OBS,
        "+ observables, region, ethnicity": OBS + REGETH,
        "+ K-means type": ["km_1", "km_2"],
        "+ theta(t)": ["theta"],
        "+ theta(t), theta(t-1)": ["theta", "y_lag"],
        "+ K-means forecast": ["km_fc"],
        "+ K-means type, theta(t)": ["km_1", "km_2", "theta"],
        "+ all": ["km_1", "km_2", "theta"] + OBS,
    }
    return frames, SETS


def main() -> int:
    frames, SETS = build()
    base_cols = ["a", "theta"] + OBS + ["km_1", "km_fc"]
    rows, by_age = [], []
    for oc, olab in (("y_next", r"health, $\theta$"), ("cost_next", "cost proxy (£)")):
        for k in (1, 2):
            s = frames[k].dropna(subset=[oc] + base_cols)
            for r in M33.score_sets(s, oc, "r2", SETS):
                rows.append({"outcome": oc, "outcome_label": olab, "horizon": k, "metric": "r2", **r})
            by_age += [{"outcome": oc, "horizon": k, **r} for r in M33.scores_by_age(s.dropna(subset=["y_lag"]), oc, SETS, BY_AGE_SETS)]
            print(f"{oc} t+{k}: {len(s):,} people")
    t = pd.DataFrame(rows)
    t["gain"] = t["score"] - t.groupby(["outcome", "horizon"])["score"].transform("first")
    t.to_csv(DESC / "prediction.csv", index=False)
    pd.DataFrame(by_age).to_csv(DESC / "prediction_by_age.csv", index=False)
    s = frames[3].dropna(subset=["died_next"] + base_cols)
    death = pd.DataFrame([{"outcome": "died_next", **r} for r in M33.score_sets(s, "died_next", "auc", SETS)])
    death["gain"] = death["score"] - death["score"].iloc[0]
    death.to_csv(DESC / "prediction_death_heldout.csv", index=False)
    print(f"held-out death: {len(s):,} people, event rate {s['died_next'].mean():.3%}")
    print(t[t["outcome"] == "y_next"].pivot(index="predictors", columns="horizon", values="gain").round(3).to_string())
    print(t[t["outcome"] == "cost_next"].pivot(index="predictors", columns="horizon", values="gain").round(4).to_string())
    print(death[["predictors", "n", "score", "gain"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
