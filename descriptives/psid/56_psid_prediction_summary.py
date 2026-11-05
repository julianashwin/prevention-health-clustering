"""Figure 13 on PSID: predicting ahead and predicting the type, theta, K-means only.

The 2 x 2 of descriptives/56_paper_prediction_summary.py (layout and drawing copied) on the PSID
csvs of 33_psid_prediction.py and 43_psid_type_from_initial.py, without the AR(1) plus "spike" and
multidimensional bars:
  (a) gain in out-of-sample R-squared over age alone at t+1 (full bar) and t+2 (light bar), one PSID
      wave being two years: observables (education, income rank, sex, region, ethnic group), theta_t,
      theta_t and theta_{t-1}, the K-means forecast; every fit held out each person's last two rows.
  (c) the same at t+1 within age bands at t, for theta_t and the K-means forecast.
  (b) balanced accuracy for the K-means type of the fit with the first two rows held out, from the
      observables, the closest held reading theta_{t-1}, and both held readings.
  (d) who is in each K-means type of (b): education, income third and sex within each type (the
      paper draws the AR(1) plus "spike" types here).

Output: descriptives/psid/figures/main/fig_prediction_summary.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, EDUC_ORDER, FIG  # noqa: E402
from _style import INK2, apply_style  # noqa: E402

KM_COL = "#b8860b"
OBS_AHEAD, OBS_TYPE = "+ observables, region, ethnicity", "+ education, income, sex, region, ethnicity"
EDUC_LAB = ["Less than high school", "High school", "Some college", "Bachelor's or more"]


def main() -> int:
    apply_style()
    P = pd.read_csv(DESC / "prediction.csv")
    q = pd.read_csv(DESC / "type_from_initial.csv")
    types = pd.read_parquet(DESC / "type_first2_people.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    LEFT = [("Observables", OBS_AHEAD, INK2), (r"$\theta_t$", "+ theta(t)", "#1f6fb4"),
            (r"$\theta_t,\ \theta_{t-1}$", "+ theta(t), theta(t-1)", "#1f6fb4"), ("K-means forecast", "+ K-means forecast", KM_COL)]
    ax = axes[0, 0]
    s = P[P["outcome"] == "y_next"].set_index(["predictors", "horizon"])["gain"]
    y = np.arange(len(LEFT))[::-1].astype(float); h = 0.38
    xmax = max(s.loc[(pr, k)] for _, pr, _ in LEFT for k in (1, 2)) * 1.14
    for yi, (lab, pr, c) in zip(y, LEFT):
        for k, off, alpha in ((1, h / 2, 1.0), (2, -h / 2, 0.4)):
            v = s.loc[(pr, k)]
            ax.barh(yi + off, v, height=h * 0.95, color=c, alpha=alpha)
            ax.text(v + 0.006 * xmax, yi + off, f"+{v:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [l for l, _, _ in LEFT]); ax.set_ylim(-0.6, y[0] + 0.7); ax.set_xlim(0, xmax)
    ax.set_xlabel("gain in out-of-sample $R^2$ over age alone", fontsize=9)
    ax.set_title(r"(a) Predicting health ahead, $\theta$", loc="left", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2), plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.4)],
              labels=["$t+1$ (2 years)", "$t+2$ (4 years)"], fontsize=8, frameon=False, loc="upper right")
    # (c)
    ax = axes[1, 0]
    ba = pd.read_csv(DESC / "prediction_by_age.csv")
    b = ba[(ba["horizon"] == 1) & (ba["outcome"] == "y_next")]
    SER = [(r"$\theta_t$", "+ theta(t)", "#1f6fb4"), ("K-means forecast", "+ K-means forecast", KM_COL)]
    bands = list(dict.fromkeys(b["band"])); bw = 0.34; x = np.arange(len(bands))
    for j, (lab, pr, c) in enumerate(SER):
        g = b[b["predictors"] == pr].set_index("band").reindex(bands)["gain"]
        xs = x + (j - 0.5) * bw
        ax.bar(xs, g, width=bw * 0.92, color=c, label=f"{lab}, $t+1$")
        for xx, gg in zip(xs, g):
            ax.text(xx, gg + 0.005 * g.max() / 0.7, f"{gg:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [bb.replace("-", "–") for bb in bands]); ax.set_xlabel("age at $t$", fontsize=9)
    ax.set_ylabel("gain in out-of-sample $R^2$ over age alone", fontsize=9); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, b[b["predictors"].isin([pr for _, pr, _ in SER])]["gain"].max() * 1.22)
    ax.set_title("(c) Predicting health ahead, by age", loc="left", fontsize=10)
    # (b)
    ax = axes[0, 1]
    INFO = [("Observables", OBS_TYPE), (r"$\theta_{t-1}$", "+ theta(1)"), (r"$\theta_{t-1},\ \theta_{t-2}$", "+ theta(0), theta(1)")]
    qs = q.set_index("predictors")["balanced_accuracy"]
    yk = np.arange(len(INFO))[::-1].astype(float)
    ax.barh(yk, [qs.loc[pr] for _, pr in INFO], height=0.5, color=KM_COL, label="K-means")
    for yy, (_, pr) in zip(yk, INFO):
        ax.text(qs.loc[pr] + 0.004, yy, f"{qs.loc[pr]:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(1 / 3, color=INK2, lw=0.7, ls=":")
    ax.set_yticks(yk, [l for l, _ in INFO]); ax.set_xlim(0, 0.8)
    ax.set_xlabel("balanced accuracy for the type (mean per-type recall; 1/3 by chance)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(b) Predicting the type from the two rows before the window", loc="left", fontsize=10)
    # (d)
    ax = axes[1, 1]
    t = types.dropna(subset=["educ_group", "income_rank_mean"]).copy()
    t["inc3"] = pd.qcut(t["income_rank_mean"], 3, labels=["bottom third", "middle third", "top third"])
    t["sexl"] = np.where(t["female"] == 1, "women", "men")
    VARS = [("Education", "educ_group", EDUC_ORDER, EDUC_LAB, plt.cm.Blues(np.linspace(0.35, 0.9, 4))),
            ("Income", "inc3", ["bottom third", "middle third", "top third"], ["Bottom third", "Middle third", "Top third"], plt.cm.Greens(np.linspace(0.35, 0.85, 3))),
            ("Sex", "sexl", ["men", "women"], ["Men", "Women"], [plt.cm.Oranges(0.45), plt.cm.Oranges(0.8)])]
    TYPE_LAB = ["worst", "middle", "best"]
    shares = t["type"].value_counts(normalize=True).sort_index()
    bw, x0, ticks, ticklabels, handles = 0.8, 0.0, [], [], []
    for vname, col, cats, labs, colors in VARS:
        xs = x0 + np.arange(3)
        for c in range(3):
            dd = t[t["type"] == c][col].value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
            bottom = 0.0
            for j, (sh, colr) in enumerate(zip(dd, colors)):
                bar = ax.bar(xs[c], sh, bottom=bottom, width=bw, color=colr, edgecolor="white", lw=0.6)
                if sh > 0.07:
                    ax.text(xs[c], bottom + sh / 2, f"{sh:.0%}", ha="center", va="center", fontsize=6.8, color="white" if j >= len(cats) / 2 else "k")
                bottom += sh
                if c == 0:
                    handles.append((bar, labs[j]))
            ticks.append(xs[c]); ticklabels.append(f"{TYPE_LAB[c]}\n{shares.loc[c]:.0%}")
        ax.text(xs[1], 1.03, vname, ha="center", va="bottom", fontsize=9)
        x0 += 3 + 0.8
    ax.set_xticks(ticks, ticklabels, fontsize=7.5); ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share within the K-means type", fontsize=9)
    ax.set_title("(d) Who is in each K-means type", loc="left", fontsize=10)
    ax.legend([h_ for h_, _ in handles], [l_ for _, l_ in handles], fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_prediction_summary.png")
    print("wrote fig_prediction_summary.png; type shares", shares.round(3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
