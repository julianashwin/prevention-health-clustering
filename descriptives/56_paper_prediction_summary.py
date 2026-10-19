"""Main-text summary of the two prediction exercises, 2 x 2.

Left column, predicting ahead (from 33_paper_prediction.py, paper_prediction.csv, on theta):
out-of-sample R-squared at t+1 (full bar) and t+2 (light bar) for (a) the measure and (c) the
cost proxy, from the observables, theta(t), theta(t) and its lag, and the forecasts of partial
K-means, the AR(1) plus "spike" mixture and the multidimensional fit with birth-decade shifts,
each on top of a quadratic in age; dotted, age alone.

Right column, predicting the type (from 43_paper_type_from_initial.py --first2,
paper_type_from_initial_first2.csv): (b) macro one-vs-rest AUC for the type of the fits with
each person's first two rows held out, from the observables, the first held reading and both
held readings, for K-means, AR(1) plus "spike" and the multidimensional fit; dotted, age
alone. (d) How the observables are distributed across those types: the share of each
education group, income third and sex in the worst type, against the worst type's overall
share (dotted). The per-person types are recomputed with 43's own functions and cached to
paper_prediction_summary_types.csv.

The multidimensional typology in (b) and (d) is the fit without birth-decade shifts: there is
no first-two-rows-held-out fit with them (theta-ssm-mort-cohort-first2ho).

Outputs: paper/figures/fig_prediction_summary.png,
         artifacts/descriptives/paper_prediction_summary_types.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, MENTAL_COL, MULTIDIM_CONTRACT, load_labels  # noqa: E402
from _style import CLUSTER, INK2, PURPLE, apply_style  # noqa: E402

KM_COL, SSM_COL, MD_COL = "#b8860b", CLUSTER[0], PURPLE
TYPES_CSV = DESC / "paper_prediction_summary_types.csv"
EDUC = [("other/none", "No qualifications"), ("GCSE", "GCSE"), ("A-level", "A-level"), ("degree or higher", "Degree or higher")]


def person_types() -> pd.DataFrame:
    """Modal type of each first-two-rows-held-out typology, with the observables."""
    if TYPES_CSV.exists() and "--recompute" not in sys.argv:
        return pd.read_csv(TYPES_CSV)
    argv = sys.argv; sys.argv = ["43", "--first2"]
    M43 = importlib.import_module("43_paper_type_from_initial")
    sys.argv = argv
    sub, first = M43.first_rows()
    L = M43.load_labels("theta_first2held")
    out = [first.merge(pd.DataFrame({"pidp": L.index, "type": L.to_numpy()}), on="pidp").assign(typology="kmeans")]
    out.append(M43.bayes_types("ssm", sub, first).assign(typology="ssm"))
    sub_md, first_md = M43.first_rows(MULTIDIM_CONTRACT, extra=(MENTAL_COL,), kheld=M43.MD_KHELD)
    out.append(M43.multidim_types(sub_md, first_md.rename(columns={MENTAL_COL: "m0"})).assign(typology="md"))
    t = pd.concat([o[["typology", "pidp", "type", "y0", "educ_group", "income_rank_mean", "female"]] for o in out])
    t.to_csv(TYPES_CSV, index=False)
    return t


def main() -> int:
    apply_style()
    p = pd.read_csv(DESC / "paper_prediction.csv"); p = p[p["variant"] == "theta"]
    q = pd.read_csv(DESC / "paper_type_from_initial_first2.csv")
    types = person_types()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    # ---- left: predicting ahead
    LEFT = [("Observables", "+ observables", INK2), (r"$\theta_t$", "+ theta(t)", "#1f6fb4"),
            (r"$\theta_t,\ \theta_{t-1}$", "+ theta(t), theta(t-1)", "#1f6fb4"),
            ("K-means forecast", "+ K-means forecast", KM_COL), ('AR(1)+"spike" forecast', '+ AR(1)+"spike" forecast', SSM_COL),
            ("Multidimensional forecast\n(cohort shifts)", "+ multidim cohort forecast", MD_COL)]
    for ax, oc, title in ((axes[0, 0], "y_next", r"(a) Predicting health ahead, $\theta$"),
                          (axes[1, 0], "cost_next", "(c) Predicting the cost proxy ahead")):
        s = p[p["outcome"] == oc].set_index(["predictors", "horizon"])["score"]
        y = np.arange(len(LEFT))[::-1].astype(float); h = 0.38
        for yi, (lab, pr, c) in zip(y, LEFT):
            for k, off, alpha in ((1, h / 2, 1.0), (2, -h / 2, 0.4)):
                v = s.loc[(pr, k)]
                ax.barh(yi + off, v, height=h * 0.95, color=c, alpha=alpha)
                ax.text(v + 0.004 * ax.get_xlim()[1], yi + off, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        for k, ls in ((1, (0, (1, 1.5))), (2, (0, (1, 3.5)))):
            ax.axvline(s.loc[("age", k)], color="k", lw=1, ls=ls)
        ax.text(s.loc[("age", 1)], y[0] + 0.62, " age only", fontsize=7.5, va="bottom")
        ax.set_yticks(y, [l for l, _, _ in LEFT]); ax.set_ylim(-0.6, y[0] + 0.9)
        ax.set_xlim(0, s.max() * 1.12); ax.set_xlabel("out-of-sample $R^2$", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2), plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.4)],
                  labels=["$t+1$", "$t+2$"], fontsize=8, frameon=False, loc="upper right")
    # ---- (b) predicting the type
    ax = axes[0, 1]
    TYP = [("kmeans", "K-means", KM_COL), ("ssm", 'AR(1)+"spike"', SSM_COL), ("md", "Multidimensional", MD_COL)]
    INFO = [("Observables", "+ education, income, sex"), (r"$\theta_{t-2}$", "+ theta(0)"), (r"$\theta_{t-2},\ \theta_{t-1}$", "+ theta(0), theta(1)")]
    w, gap = 0.26, 0.5
    centres = np.arange(len(INFO))[::-1] * (len(TYP) * w + gap)
    for k, (tag, lab, c) in enumerate(TYP):
        qs = q[q["typology"] == tag].set_index("predictors")["auc"]
        yk = centres + (1 - k) * w
        ax.barh(yk, [qs.loc[pr] for _, pr in INFO], height=w * 0.92, color=c, label=lab)
        for yy in yk:   # age alone, dotted, across the typology's own bar
            ax.plot([qs.loc["age(0)"]] * 2, [yy - w / 2, yy + w / 2], color="k", lw=1.2, ls=(0, (1, 1.2)))
        for yy, (_, pr) in zip(yk, INFO):
            ax.text(qs.loc[pr] + 0.003, yy, f"{qs.loc[pr]:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(centres, [l for l, _ in INFO]); ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("macro one-vs-rest AUC for the type (dotted: age only)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(b) Predicting the type from the first two rows", loc="left", fontsize=10)
    # ---- (d) the observables across the types: share in the worst type
    ax = axes[1, 1]
    t = types.dropna(subset=["educ_group", "income_rank_mean"]).copy()
    t["inc3"] = pd.qcut(t.groupby("pidp")["income_rank_mean"].transform("first"), 3, labels=["Bottom income third", "Middle income third", "Top income third"])
    GROUPS = [("educ_group", g, lab) for g, lab in EDUC] + [("inc3", g, g) for g in ["Bottom income third", "Middle income third", "Top income third"]] \
        + [("female", 0.0, "Men"), ("female", 1.0, "Women")]
    blocks = [0] * 4 + [1] * 3 + [2] * 2
    centres = -(np.arange(len(GROUPS)) * (len(TYP) * w + 0.25) + np.array(blocks) * 0.6)
    for k, (tag, lab, c) in enumerate(TYP):
        d = t[t["typology"] == tag]; worst = d["type"].eq(0)
        vals = [worst[d[col] == g].mean() for col, g, _ in GROUPS]
        yk = centres + (1 - k) * w
        ax.barh(yk, vals, height=w * 0.92, color=c, label=lab)
        ax.axvline(worst.mean(), color=c, lw=1, ls=(0, (1, 1.5)))
    ax.set_yticks(centres, [l for _, _, l in GROUPS], fontsize=8)
    ax.set_xlabel("share in the worst type (dotted: the worst type's overall share)", fontsize=9)
    ax.set_title("(d) The observables across the types", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_prediction_summary.png")
    print("wrote fig_prediction_summary.png")
    for tag, _, _ in TYP:
        d = t[t["typology"] == tag]
        print(tag, len(d), "type shares", d["type"].value_counts(normalize=True).sort_index().round(3).tolist(),
              "mean y0 by type", d.groupby("type")["y0"].mean().round(2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
