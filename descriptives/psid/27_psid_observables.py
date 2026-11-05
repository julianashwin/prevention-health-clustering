"""Figure 12 on PSID: lifecycle trajectories of theta by health type and by observable groups.

descriptives/27_paper_observables.py (logic copied) on the PSID clustered people, with --no-bayes:
  (a) the K-means types (the paper adds the AR(1) plus "spike" class paths, not fitted here)
  (b) observable partitions: education (4) x sex (2) x white / other (2) x income tercile (3)
  (c) education, four US groups (educ_best: the best level recorded at 25 or over)
  (d) deciles of the income rank (the person's mean within-wave rank of equivalised family income)
Cell means by age, cells with fewer than 30 people suppressed, centred five-year rolling means;
(b) and (d) stop at 70. White is non-Hispanic white; other is everyone else with a recorded group.
The between-group share of the variance of theta at 30, 50, 70 and the ARI with the types go to a csv.

Outputs: descriptives/psid/figures/main/fig_observables.png, artifacts/descriptives/psid/observables.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from _psid_common import DESC, EDUC_ORDER, FIG, K, MAX_AGE, MIN_AGE, contract, load_labels, observables  # noqa: E402
from _style import CLUSTER, INK2, apply_style  # noqa: E402

M27 = importlib.import_module("27_paper_observables")      # cell_means, between_share
INC_ORDER = ["bottom third", "middle third", "top third"]
INCOME_MAX_AGE, ROLL = 70, 5


def main() -> int:
    apply_style()
    v = "theta"
    rollm = lambda q: q.assign(mean=q["mean"].rolling(ROLL, center=True, min_periods=1).mean())  # noqa: E731
    d = contract()
    L = load_labels(v)
    per = observables(L.index)
    per["sex"] = np.where(per["female"] == 1, "women", "men")
    per["ethnic"] = np.where(per["race_group"] == "white", "white", "other")
    per.loc[per["race_group"].isna(), "ethnic"] = np.nan
    per["income_group"] = pd.qcut(per["income_rank_mean"], 3, labels=INC_ORDER)
    per["income_decile"] = pd.qcut(per["income_rank_mean"], 10, labels=[f"D{i}" for i in range(1, 11)])
    per["educ_group"] = pd.Categorical(per["educ_group"], EDUC_ORDER, ordered=True)
    keys = ["educ_group", "sex", "ethnic", "income_group"]
    per["partition"] = per[keys].astype(str).agg(" / ".join, axis=1)
    per.loc[per[keys].isna().any(axis=1), "partition"] = np.nan
    print(f"clustered people {len(L):,}; with all four observables {per['partition'].notna().sum():,}; partitions {per['partition'].nunique()}")
    s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    s = s.join(per[["educ_group", "income_group", "income_decile", "partition"]], on="pidp")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True, gridspec_kw={"hspace": 0.32, "wspace": 0.08})
    cm = M27.cell_means
    ax = axes[0, 0]
    t = cm(s, "cluster", v)
    for c in range(K):
        q = rollm(t[t["cluster"] == c])
        ax.plot(q["age"], q["mean"], color=CLUSTER[c], lw=1.8, label=f"K-means type {c + 1} ({float((L == c).mean()):.0%})")
    ax.set_title("(a) health types", loc="left"); ax.legend(loc="lower left", fontsize=8)
    ax = axes[0, 1]
    t = cm(s.dropna(subset=["partition"]), "partition", v); t = t[t["age"] <= INCOME_MAX_AGE]
    cmap = plt.get_cmap("winter")
    for k, (g, q) in enumerate(t.groupby("partition")):
        q = rollm(q.sort_values("age"))
        ax.plot(q["age"], q["mean"], color=cmap(k / max(t["partition"].nunique() - 1, 1)), lw=0.8, alpha=0.8)
    ax.set_title(f"(b) observable partitions ({t['partition'].nunique()} groups)", loc="left")
    ax = axes[1, 0]
    t = cm(s.dropna(subset=["educ_group"]), "educ_group", v)
    for k, g in enumerate(EDUC_ORDER):
        q = rollm(t[t["educ_group"] == g].sort_values("age"))
        ax.plot(q["age"], q["mean"], color=cmap(k / 3), lw=1.8, label=g)
    ax.set_title("(c) highest qualification", loc="left"); ax.legend(loc="lower left", fontsize=8)
    ax = axes[1, 1]
    t = cm(s.dropna(subset=["income_decile"]), "income_decile", v); t = t[t["age"] <= INCOME_MAX_AGE]
    for k in range(10):
        q = rollm(t[t["income_decile"] == f"D{k + 1}"].sort_values("age"))
        ax.plot(q["age"], q["mean"], color=cmap(k / 9), lw=1.4, label=f"D{k + 1}")
    ax.set_title("(d) income deciles", loc="left"); ax.legend(loc="lower left", fontsize=7, ncols=2)
    for ax in axes.flat:
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    for ax in axes[1]:
        ax.set_xlabel("age")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mean $\theta$")
    fig.text(0.01, -0.01,
             f"PSID clustered people ({len(L):,}: the PSID contract, at least three observed ages 20-89). Every series is the age-specific mean of theta, cells with fewer than 30 "
             "people suppressed, as a centred five-year rolling mean.\nPartitions in (b): education (4) x sex (2) x white (non-Hispanic) / other (2) x income tercile (3). "
             "Income is the person's mean within-wave percentile rank of equivalised family income\n(previous calendar year, before tax); education the best level recorded at 25 or over. "
             f"Panels (b) and (d) involve income and stop at age {INCOME_MAX_AGE}. Panel (a) shows the K-means types only (no Bayesian classes are fitted on PSID).",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_observables.png", bbox_inches="tight")
    recs = []
    per_l = per.join(L.rename("cluster"))
    for col, name in [("educ_group", "education (4 groups)"), ("income_group", "income tercile"), ("income_decile", "income decile"),
                      ("partition", "education x sex x ethnicity x income"), ("cluster", "health types (K = 3)")]:
        bs = M27.between_share(s, col, v)
        ok = per_l.dropna(subset=[col, "cluster"])
        ari = 1.0 if col == "cluster" else adjusted_rand_score(ok[col].astype(str), ok["cluster"])
        recs.append({"partition": name, **{f"between_share_{a}": bs[a] for a in M27.CHECK_AGES}, "ari_vs_types": ari})
    out = pd.DataFrame(recs); out.to_csv(DESC / "observables.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
