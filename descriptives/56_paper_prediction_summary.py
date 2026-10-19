"""Main-text summary of the two prediction exercises, 2 x 2.

Left column, predicting ahead (from 33_paper_prediction.py, paper_prediction.csv, on theta):
(a) the gain in out-of-sample R-squared over age alone at t+1 (full bar) and t+2 (light bar)
for the measure, from the observables (education, income rank, sex,
region, ethnic group), theta(t), theta(t) and its lag, and the forecasts of partial K-means,
the AR(1) plus "spike" mixture and the multidimensional fit with birth-decade shifts, each on
top of a quadratic in age. Every fit behind them held out each person's last two rows.
(c) The same gain within age bands at t (paper_prediction_by_age.csv), for theta(t) and the
AR(1) plus "spike" forecast, on the people with a lag observed. The cost-proxy version of
(a) is a separate appendix figure, fig_prediction_cost.png.

Right column, predicting the type (from 43_paper_type_from_initial.py,
paper_type_from_initial_first2.csv): (b) balanced accuracy (the mean per-type recall; 1/3 by
chance) for the type of the fits with each person's first two rows held out, from the same
observables, the closest held reading theta(t-1), and both held readings, for K-means,
AR(1) plus "spike" and the multidimensional fit. (d) Who is in each AR(1) plus "spike" type:
the distribution of education, income third and sex within each of the three types. The
per-person types are recomputed with 43's own functions and cached to
paper_prediction_summary_types.csv (--recompute to refresh).

The multidimensional typology in (b) is 43's: the fit with birth-decade shifts
(theta-ssm-mort-cohort-first2ho) once it has finished, the fit without them until then.
Rerun 43 and then this script with --recompute when it lands.

Outputs: paper/figures/fig_prediction_summary.png, paper/figures/fig_prediction_cost.png,
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
    OBS_AHEAD, OBS_TYPE = "+ observables, region, ethnicity", "+ education, income, sex, region, ethnicity"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    # ---- left: predicting ahead, gain over age alone
    LEFT = [("Observables", OBS_AHEAD, INK2), (r"$\theta_t$", "+ theta(t)", "#1f6fb4"),
            (r"$\theta_t,\ \theta_{t-1}$", "+ theta(t), theta(t-1)", "#1f6fb4"),
            ("K-means forecast", "+ K-means forecast", KM_COL), ('AR(1)+"spike" forecast', '+ AR(1)+"spike" forecast', SSM_COL),
            ("Multidimensional forecast\n(cohort shifts)", "+ multidim cohort forecast", MD_COL)]
    def ahead(ax, oc, title):
        s = p[p["outcome"] == oc].set_index(["predictors", "horizon"])["gain"]
        y = np.arange(len(LEFT))[::-1].astype(float); h = 0.38
        xmax = max(s.loc[(pr, k)] for _, pr, _ in LEFT for k in (1, 2)) * 1.14
        for yi, (lab, pr, c) in zip(y, LEFT):
            for k, off, alpha in ((1, h / 2, 1.0), (2, -h / 2, 0.4)):
                v = s.loc[(pr, k)]
                ax.barh(yi + off, v, height=h * 0.95, color=c, alpha=alpha)
                ax.text(v + 0.006 * xmax, yi + off, f"+{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y, [l for l, _, _ in LEFT]); ax.set_ylim(-0.6, y[0] + 0.7)
        ax.set_xlim(0, xmax); ax.set_xlabel("gain in out-of-sample $R^2$ over age alone", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2), plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.4)],
                  labels=["$t+1$", "$t+2$"], fontsize=8, frameon=False, loc="upper right")

    ahead(axes[0, 0], "y_next", r"(a) Predicting health ahead, $\theta$")
    # ---- (c) predicting ahead by age at t
    ax = axes[1, 0]
    ba = pd.read_csv(DESC / "paper_prediction_by_age.csv")
    ba = ba[(ba["variant"] == "theta") & (ba["outcome"] == "y_next")]
    bands = list(dict.fromkeys(ba["band"]))
    SER = [(r"$\theta_t$", "+ theta(t)", "#1f6fb4"), ('AR(1)+"spike" forecast', '+ AR(1)+"spike" forecast', SSM_COL)]
    bw = 0.2; x = np.arange(len(bands))
    for j, (lab, pr, c) in enumerate(SER):
        for k, alpha in ((1, 1.0), (2, 0.4)):
            g = ba[(ba["predictors"] == pr) & (ba["horizon"] == k)].set_index("band").reindex(bands)["gain"]
            xs = x + (2 * j + (k - 1) - 1.5) * bw
            ax.bar(xs, g, width=bw * 0.92, color=c, alpha=alpha, label=f"{lab}, $t+{k}$")
            for xx, gg in zip(xs, g):
                ax.text(xx, gg + 0.005, f"{gg:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [f"{b.replace('-', '–')}" for b in bands]); ax.set_xlabel("age at $t$", fontsize=9)
    ax.set_ylabel("gain in out-of-sample $R^2$ over age alone", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, ba[ba["predictors"].isin([pr for _, pr, _ in SER])]["gain"].max() * 1.25)
    ax.set_title(r"(c) Predicting health ahead, by age", loc="left", fontsize=10)
    # ---- (b) predicting the type, balanced accuracy
    ax = axes[0, 1]
    TYP = [("kmeans", "K-means", KM_COL), ("ssm", 'AR(1)+"spike"', SSM_COL), ("md", "Multidimensional", MD_COL)]
    INFO = [("Observables", OBS_TYPE), (r"$\theta_{t-1}$", "+ theta(1)"), (r"$\theta_{t-1},\ \theta_{t-2}$", "+ theta(0), theta(1)")]
    w, gap = 0.26, 0.5
    centres = np.arange(len(INFO))[::-1] * (len(TYP) * w + gap)
    for k, (tag, lab, c) in enumerate(TYP):
        qs = q[q["typology"] == tag].set_index("predictors")["balanced_accuracy"]
        yk = centres + (1 - k) * w
        ax.barh(yk, [qs.loc[pr] for _, pr in INFO], height=w * 0.92, color=c, label=lab)
        for yy, (_, pr) in zip(yk, INFO):
            ax.text(qs.loc[pr] + 0.004, yy, f"{qs.loc[pr]:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(centres, [l for l, _ in INFO]); ax.set_xlim(0, 0.8)
    ax.set_xlabel("balanced accuracy for the type (mean per-type recall; 1/3 by chance)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(b) Predicting the type from the two rows before the window", loc="left", fontsize=10)
    # ---- (d) who is in each AR(1) + "spike" type
    ax = axes[1, 1]
    t = types[types["typology"] == "ssm"].dropna(subset=["educ_group", "income_rank_mean"]).copy()
    t["inc3"] = pd.qcut(t["income_rank_mean"], 3, labels=["bottom third", "middle third", "top third"])
    t["sexl"] = np.where(t["female"] == 1, "women", "men")
    VARS = [("Education", "educ_group", [g for g, _ in EDUC], [l for _, l in EDUC], plt.cm.Blues(np.linspace(0.35, 0.9, 4))),
            ("Income", "inc3", ["bottom third", "middle third", "top third"], ["Bottom third", "Middle third", "Top third"], plt.cm.Greens(np.linspace(0.35, 0.85, 3))),
            ("Sex", "sexl", ["men", "women"], ["Men", "Women"], [plt.cm.Oranges(0.45), plt.cm.Oranges(0.8)])]
    TYPE_LAB = ["worst", "middle", "best"]
    shares = t["type"].value_counts(normalize=True).sort_index()
    bw, x0, ticks, ticklabels, handles = 0.8, 0.0, [], [], []
    for vi, (vname, col, cats, labs, colors) in enumerate(VARS):
        xs = x0 + np.arange(3)
        for c in range(3):
            d = t[t["type"] == c][col].value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
            bottom = 0.0
            for j, (sh, colr) in enumerate(zip(d, colors)):
                bar = ax.bar(xs[c], sh, bottom=bottom, width=bw, color=colr, edgecolor="white", lw=0.6)
                if sh > 0.07:
                    ax.text(xs[c], bottom + sh / 2, f"{sh:.0%}", ha="center", va="center", fontsize=6.8,
                            color="white" if j >= len(cats) / 2 else "k")
                bottom += sh
                if c == 0:
                    handles.append((bar, labs[j], vi))
            ticks.append(xs[c]); ticklabels.append(f"{TYPE_LAB[c]}\n{shares.loc[c]:.0%}")
        ax.text(xs[1], 1.03, vname, ha="center", va="bottom", fontsize=9)
        x0 += 3 + 0.8
    ax.set_xticks(ticks, ticklabels, fontsize=7.5); ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel('share within the AR(1)+"spike" type', fontsize=9)
    ax.set_title('(d) Who is in each AR(1)+"spike" type', loc="left", fontsize=10)
    leg = [h for h, _, _ in handles]; legl = [l for _, l, _ in handles]
    ax.legend(leg, legl, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_prediction_summary.png")
    # the cost-proxy twin of (a), for the appendix
    figc, axc = plt.subplots(figsize=(6.2, 4.3))
    ahead(axc, "cost_next", "Predicting the cost proxy ahead")
    figc.tight_layout(); figc.savefig(FIG / "fig_prediction_cost.png")
    print("wrote fig_prediction_summary.png, fig_prediction_cost.png")
    print("ssm types", len(t), "shares", shares.round(3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
