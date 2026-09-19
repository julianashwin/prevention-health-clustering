"""Figure 8: lifecycle health trajectories by health type and by observable groups.

Four panels on the clustered people (h, at least three observed ages 20-90),
each series an age-specific cell mean of h, cells with fewer than 30 people
suppressed:
  (a) the three health types
  (b) observable partitions: every cell of highest qualification (4) x sex (2)
      x white / other ethnic group (2) x tercile of the person's mean
      within-wave rank of equivalised household net income (3), 48 groups
  (c) highest qualification
  (d) deciles of the income rank
Education and income come from data_cleaning/06_build_observables.py; sex and
ethnic group from the processed panel (first non-missing value per person).

The table (tab_observables.tex) records the between-group share of the
cross-sectional variance of h at ages 30, 50 and 70 for each partition, and
the adjusted Rand index of the observable partitions with the types.

Outputs: paper/figures/fig_observables.png, paper/tables/tab_observables.tex,
         artifacts/descriptives/paper_observables.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, K, MAX_AGE, MIN_AGE, TAB, load_labels, load_measure, write_table  # noqa: E402
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402

EDUC_ORDER = ["other/none", "GCSE", "A-level", "degree or higher"]
INC_ORDER = ["bottom third", "middle third", "top third"]
CHECK_AGES = [30, 50, 70]
MIN_CELL = 30


def cell_means(s: pd.DataFrame, group, v: str) -> pd.DataFrame:
    t = s.groupby([*group, "age"] if isinstance(group, list) else [group, "age"])[v].agg(["mean", "count"]).reset_index()
    return t[t["count"] >= MIN_CELL]


def between_share(s: pd.DataFrame, group: str, v: str) -> dict:
    out = {}
    for a in CHECK_AGES:
        w = s[s["age"].between(a - 2, a + 2)].dropna(subset=[group])
        g = w.groupby(group, observed=True)[v].agg(["mean", "size"])
        gm = np.average(g["mean"], weights=g["size"])
        out[a] = float((g["size"] * (g["mean"] - gm) ** 2).sum() / g["size"].sum() / w[v].var(ddof=0))
    return out


def main() -> int:
    apply_style()
    v = "h"
    d = load_measure([v])
    L = load_labels(v)
    per = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet")
    per = per[per["pidp"].isin(L.index)].set_index("pidp")
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "sex", "racel_dv"])
    panel = panel[panel["pidp"].isin(L.index)].sort_values(["pidp", "wave"])
    first = panel.groupby("pidp").agg(sex=("sex", "first"), racel=("racel_dv", "first"))
    first["sex"] = first["sex"].map({1: "men", 2: "women"})
    first["ethnic"] = pd.Series(np.where(first["racel"] <= 4, "white", "other"), index=first.index).where(
        first["racel"].notna() & (first["racel"] <= 96))
    per = per.join(first[["sex", "ethnic"]])
    per["income_group"] = pd.qcut(per["income_rank_mean"], 3, labels=INC_ORDER)
    per["income_decile"] = pd.qcut(per["income_rank_mean"], 10, labels=[f"D{i}" for i in range(1, 11)])
    per["educ_group"] = pd.Categorical(per["educ_group"], EDUC_ORDER, ordered=True)
    per["partition"] = per[["educ_group", "sex", "ethnic", "income_group"]].astype(str).agg(" / ".join, axis=1)
    per.loc[per[["educ_group", "sex", "ethnic", "income_group"]].isna().any(axis=1), "partition"] = np.nan
    print(f"clustered people {len(L):,}; with all four observables {per['partition'].notna().sum():,}; "
          f"partitions {per['partition'].nunique()}")
    s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    s = s.join(per[["educ_group", "income_group", "income_decile", "partition"]], on="pidp")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True, gridspec_kw={"hspace": 0.32, "wspace": 0.08})
    # (a) types
    ax = axes[0, 0]
    t = cell_means(s, "cluster", v)
    for c in range(K):
        q = t[t["cluster"] == c]
        ax.plot(q["age"], q["mean"], color=CLUSTER[c], lw=1.8, label=f"type {c + 1} ({float((L == c).mean()):.0%})")
    ax.set_title("(a) health types", loc="left"); ax.legend(loc="lower left", fontsize=8)
    # (b) partitions
    ax = axes[0, 1]
    t = cell_means(s.dropna(subset=["partition"]), "partition", v)
    cmap = plt.get_cmap("winter")
    for k, (g, q) in enumerate(t.groupby("partition")):
        ax.plot(q["age"], q["mean"], color=cmap(k / max(t["partition"].nunique() - 1, 1)), lw=0.8, alpha=0.8)
    ax.set_title(f"(b) observable partitions ({t['partition'].nunique()} groups)", loc="left")
    # (c) education
    ax = axes[1, 0]
    t = cell_means(s.dropna(subset=["educ_group"]), "educ_group", v)
    cmap = plt.get_cmap("winter")
    for k, g in enumerate(EDUC_ORDER):
        q = t[t["educ_group"] == g]
        ax.plot(q["age"], q["mean"], color=cmap(k / 3), lw=1.8, label=g)
    ax.set_title("(c) highest qualification", loc="left"); ax.legend(loc="lower left", fontsize=8)
    # (d) income deciles
    ax = axes[1, 1]
    t = cell_means(s.dropna(subset=["income_decile"]), "income_decile", v)
    for k in range(10):
        q = t[t["income_decile"] == f"D{k + 1}"]
        ax.plot(q["age"], q["mean"], color=cmap(k / 9), lw=1.4, label=f"D{k + 1}")
    ax.set_title("(d) income deciles", loc="left"); ax.legend(loc="lower left", fontsize=7, ncols=2)
    for ax in axes.flat:
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    for ax in axes[1]:
        ax.set_xlabel("age")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean $h$")
    fig.text(0.01, -0.01,
             f"Clustered people ({len(L):,}, at least three observed ages 20-90). Every series is the age-specific mean "
             "of h, cells with fewer than 30 people suppressed. Partitions in (b): highest qualification (4) x sex (2) x\n"
             "white / other ethnic group (2) x income tercile (3). Income is the person's mean within-wave percentile "
             "rank of equivalised household net income; education the highest qualification ever recorded.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_observables.png")

    rows, recs = [], []
    per_l = per.join(L.rename("cluster"))
    for col, name in [("educ_group", "education (4 groups)"), ("income_group", "income tercile"),
                      ("income_decile", "income decile"), ("partition", "education x sex x ethnicity x income (48)"),
                      ("cluster", "health types (K = 3)")]:
        bs = between_share(s, col, v)
        ok = per_l.dropna(subset=[col, "cluster"])
        ari = 1.0 if col == "cluster" else adjusted_rand_score(ok[col].astype(str), ok["cluster"])
        rows.append([name] + [f"{bs[a]:.3f}" for a in CHECK_AGES] + [f"{ari:.3f}"])
        recs.append({"partition": name, **{f"between_share_{a}": bs[a] for a in CHECK_AGES}, "ari_vs_types": ari})
    write_table(TAB / "tab_observables.tex",
                ["partition"] + [f"between share, age {a}" for a in CHECK_AGES] + ["ARI vs types"], rows)
    out = pd.DataFrame(recs)
    out.to_csv(DESC / "paper_observables.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("wrote fig_observables.png and tab_observables.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
