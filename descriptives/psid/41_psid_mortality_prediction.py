"""Figure 11 on PSID: predicting death, AUC gain over an age-only quadratic, theta and K-means only.

descriptives/41_paper_mortality_prediction.py without the Bayesian and cohort rows:
  (a) every person-wave at risk (waves 2003-2021, people on the PSID contract), death within two
      years (before the next wave); the K-means type is the full-sample lifecycle typology, which
      never saw the death record. Predictor sets: + observables; + K-means type; + theta(t);
      + theta(t), theta(t-1) (the previous wave, two years earlier); + K-means type, theta(t);
      + all (K-means type, theta(t), observables). Each with a quadratic in age.
  (b) the held-out comparison of 33_psid_prediction.py: the K-means types fitted with each person's
      last two rows held out, predicting from the last fitted row whether a person dies within two
      years of their last interview or leaves the panel alive (people whose last row is before 2023).
People split 70/30 by person on the UKHLS hash; logit fitted on the 70, AUC on the 30.

Outputs: descriptives/psid/figures/main/fig_prediction_mortality.png,
artifacts/descriptives/psid/mortality_prediction.csv
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, FIG, K, OBS, WAVE_STEP, contract, hash_test, load_labels, mortality_rows, obs_dummies, observables, scores  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, apply_style  # noqa: E402

M33 = importlib.import_module("33_paper_prediction")      # logistic_fit, auc

SETS = {"age": [], "+ observables": OBS, "+ K-means type": ["km_1", "km_2"], "+ theta(t)": ["theta"],
        "+ theta(t), theta(t-1)": ["theta", "y_lag"], "+ K-means type, theta(t)": ["km_1", "km_2", "theta"],
        "+ all": ["km_1", "km_2", "theta"] + OBS}
HELD_SHOW = ["age", "+ observables", "+ observables, region, ethnicity", "+ K-means type", "+ theta(t)",
             "+ theta(t), theta(t-1)", "+ K-means forecast", "+ K-means type, theta(t)", "+ all"]


def main() -> int:
    apply_style()
    people = contract()["pidp"].unique()
    d = scores()
    d = d[d["pidp"].isin(people)].merge(mortality_rows(), on=["pidp", "wave"])
    lag = scores()[["pidp", "wave", "theta"]].rename(columns={"theta": "y_lag"}); lag["wave"] += WAVE_STEP
    d = d.merge(lag, on=["pidp", "wave"], how="left")
    L = load_labels("theta")
    d["km_type"] = d["pidp"].map(L)
    for c in range(K - 1):
        d[f"km_{c + 1}"] = (d["km_type"] == c).astype(float)
    per = observables(people)
    d = d.join(per[["income_rank_mean", "female"]], on="pidp").join(obs_dummies(per)[[c for c in OBS if c.startswith("educ_")]], on="pidp")
    d["a"] = (d["age"] - 55) / 10
    s = d.dropna(subset=["died", "a", "theta", "km_type"] + OBS).reset_index(drop=True)
    test = hash_test(s["pidp"]); y = s["died"].to_numpy(float)
    print(f"{len(s):,} person-waves at risk, {s['pidp'].nunique():,} people, {int(y.sum()):,} deaths ({y.mean():.3%})")
    out = []
    for name, cols in SETS.items():
        keep = s[cols].notna().all(axis=1).to_numpy() if cols else np.ones(len(s), bool)
        s2, t2, y2 = s[keep], test[keep], y[keep]
        X = np.column_stack([np.ones(len(s2)), s2["a"], s2["a"] ** 2] + [s2[c].to_numpy(float) for c in cols])
        b = M33.logistic_fit(X[~t2], y2[~t2]); p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        out.append({"predictors": name, "n": len(s2), "n_test": int(t2.sum()), "deaths": int(y2.sum()), "score": M33.auc(y2[t2], p[t2])})
    t = pd.DataFrame(out); t["gain"] = t["score"] - t["score"].iloc[0]
    t.to_csv(DESC / "mortality_prediction.csv", index=False)
    print(t.round(3).to_string(index=False))
    held = pd.read_csv(DESC / "prediction_death_heldout.csv")
    held = held[held["predictors"].isin(HELD_SHOW)]
    colmap = {"+ observables": INK2, "+ observables, region, ethnicity": INK2, "+ K-means type": "#b8860b", "+ K-means forecast": "#b8860b",
              "+ K-means type, theta(t)": CLUSTER[1], "+ all": GREEN}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.4), gridspec_kw={"hspace": 0.32})
    for r, (tab, title) in enumerate([(t, "every person-wave at risk, full-sample types"),
                                      (held, "death after the last interview vs leaving alive, held-out types")]):
        ax = axes[r]
        names = [n for n in tab["predictors"] if n != "age"]
        q = tab.set_index("predictors").reindex(names)
        base = tab[tab["predictors"] == "age"]["score"].iloc[0]
        ax.barh(np.arange(len(names)), q["gain"], color=[colmap.get(n, ORANGE) for n in names])
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([n.replace("theta(t)", r"$\theta(t)$").replace("theta(t-1)", r"$\theta(t-1)$") for n in names], fontsize=8)
        ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.grid(True, axis="x")
        ax.set_title(f"({'ab'[r]}) on $\\theta$, {title}\nAUC gain over age only ({base:.2f})", fontsize=9, loc="left")
    n0 = t.iloc[0]; h0 = pd.read_csv(DESC / "prediction_death_heldout.csv").iloc[0]
    foot = (f"PSID. Top: death within two years (before the next biennial wave) at every person-wave at risk, waves 2003-2021, people on the PSID contract "
            f"({n0['n']:,} person-waves, {n0['deaths']:,} deaths); the K-means type is the full-sample lifecycle typology, which never saw the death record; "
            f"theta(t-1) is the previous wave, two years earlier. Bottom: the K-means types fitted with each person's last two rows held out, predicting from the last "
            f"fitted row t whether a person dies within two years of their last interview (t+2) or leaves the panel alive ({h0['n']:,} people whose last row is before 2023, "
            f"{h0['events']:.1%} deaths); the K-means forecast is the type's mean theta at the outcome's age. Both: split 70/30 by person, fitted on the 70, scored on the 30. "
            "The paper's Bayesian, multidimensional and cohort rows are not fitted on PSID.")
    fig.text(0.01, 0.02, "\n".join(textwrap.wrap(foot, 118)), fontsize=7.2, color=INK2, va="top")
    fig.subplots_adjust(top=0.95, bottom=0.2, left=0.36, right=0.98)
    fig.savefig(FIG / "fig_prediction_mortality.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
