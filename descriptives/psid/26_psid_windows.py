"""Figure 10 on PSID, adapted: types over limited age windows, partial K-means.

The paper's Figure 10 is the Bayesian windows figure (50_paper_windows_bayes.py); its K-means twin is
the appendix figure fig:windows_kmeans (descriptives/26_paper_windows.py), which this replicates on
PSID theta with the same layout and logic (copied, pointed at the PSID outputs of 22_psid_kmeans.py):
  (a) six sliding twenty-year windows, each refitted at K = 3 on the people observed at least three
      times inside it, over the lifecycle types in grey; (b) windows expanding backwards from 90;
  (c) the co-assignment correlation of each window's typology with the lifecycle one;
  (d), (e) the worst type's share of the person-waves at each age, by window; (f) the worst type's
      share of each window's people.

Outputs: descriptives/psid/figures/main/fig_windows.png, artifacts/descriptives/psid/windows_agreement.csv
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from _psid_common import DESC, FIG, K, KM_COMP, MAX_AGE, MIN_AGE, load_labels, load_trajectories  # noqa: E402
from _paper_common import coassignment_corr, onehot  # noqa: E402
from _style import CLUSTER, INK2, INK3, apply_style, share_ylim  # noqa: E402

SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(70, 90), (60, 90), (50, 90), (40, 90), (30, 90), (20, 90)]
MIN_SUPPORT = 25


def main() -> int:
    apply_style()
    v = "theta"
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), gridspec_kw={"wspace": 0.25, "hspace": 0.42})
    rows = []
    comp_all = pd.read_csv(KM_COMP)
    cmap = plt.get_cmap("viridis")
    full_lab = load_labels(v); full_tr = load_trajectories(v)
    for col, (wins, name) in enumerate([(SLIDING, "sliding twenty-year windows"), (BACKWARD, "windows expanding backwards from 90")]):
        ax = axes[0, col]
        for c in range(K):
            t = full_tr[(full_tr["cluster"] == c) & (full_tr["count"] >= MIN_SUPPORT)]
            ax.plot(t["age"], t["mean"], color=INK3, lw=3.0, alpha=0.35, zorder=1)
        for lo, hi in wins:
            if (lo, hi) == (MIN_AGE, MAX_AGE):
                continue
            tr = load_trajectories(v, lo, hi); lab = load_labels(v, lo, hi)
            for c in range(K):
                t = tr[(tr["cluster"] == c) & (tr["count"] >= MIN_SUPPORT)]
                ax.plot(t["age"], t["mean"], color=CLUSTER[c], lw=1.4, zorder=2)
                ax.plot([t["age"].iloc[0]], [t["mean"].iloc[0]], color=CLUSTER[c], marker="|", ms=7)
            both = lab.index.intersection(full_lab.index)
            rows.append({"kind": "sliding" if col == 0 else "backward", "lo": lo, "hi": hi, "people": len(lab), "shared": len(both),
                         "ari_vs_full": adjusted_rand_score(lab.loc[both], full_lab.loc[both]),
                         "coassign_corr": coassignment_corr(onehot(lab), onehot(full_lab)),
                         "same_vs_full": float((lab.loc[both] == full_lab.loc[both]).mean()), "worst_share": float((lab == 0).mean())})
        ax.set_title(rf"({'ab'[col]}) $\theta$: {name}", loc="left", fontsize=9.5)
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
    r = pd.DataFrame(rows)
    ax = axes[0, 2]
    for kind, mk, c in [("sliding", "o", CLUSTER[1]), ("backward", "s", CLUSTER[0])]:
        q = r[r["kind"] == kind]
        x = q["lo"] + (q["hi"] - q["lo"]) / 2 if kind == "sliding" else q["lo"]
        ax.plot(x, q["coassign_corr"], marker=mk, color=c, lw=1.4,
                label=f"{kind} windows" + (" (x = window midpoint)" if kind == "sliding" else " (x = start age)"))
    ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    ax.set_title(r"(c) $\theta$: correlation with lifecycle types", loc="left", fontsize=9.5); ax.legend(loc="lower left", fontsize=7.5); ax.set_xlabel("age")
    comp_v = comp_all[(comp_all["variant"] == v) & (comp_all["cluster"] == 0)]
    full_c = comp_v[(comp_v["lo"] == MIN_AGE) & (comp_v["hi"] == MAX_AGE)].set_index("age")["share"].sort_index()
    for col, wins in enumerate([SLIDING, BACKWARD]):
        ax = axes[1, col]
        ax.plot(full_c.index, full_c.rolling(5, center=True, min_periods=1).mean(), color=INK3, lw=3.0, alpha=0.35, zorder=1, label="lifecycle typology")
        wl = [w for w in wins if w != (MIN_AGE, MAX_AGE)]
        for i_w, (lo, hi) in enumerate(wl):
            q = comp_v[(comp_v["lo"] == lo) & (comp_v["hi"] == hi)].set_index("age")["share"].sort_index()
            ax.plot(q.index, q.rolling(5, center=True, min_periods=1).mean(), color=cmap(0.85 * i_w / max(len(wl) - 1, 1)), lw=1.4, zorder=2, label=f"{lo}-{hi}")
        ax.set_title(f"({'de'[col]}) worst type's share by age: {'sliding' if col == 0 else 'backward'}", loc="left", fontsize=9.5)
        ax.set_ylim(0, 0.5); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age"); ax.legend(fontsize=6.8, ncols=2, loc="upper left")
    ax = axes[1, 2]
    for kind, mk, c in [("sliding", "o", CLUSTER[1]), ("backward", "s", CLUSTER[0])]:
        q = r[r["kind"] == kind]
        x = q["lo"] + (q["hi"] - q["lo"]) / 2 if kind == "sliding" else q["lo"]
        ax.plot(x, q["worst_share"], marker=mk, color=c, lw=1.4, label=f"{kind} windows" + (" (x = midpoint)" if kind == "sliding" else " (x = start age)"))
    ax.axhline((full_lab == 0).mean(), color=INK3, lw=1.4, ls="--", label="lifecycle typology")
    ax.set_ylim(0, 0.5); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    ax.set_title("(f) worst type's share of a window's people", loc="left", fontsize=9.5); ax.legend(loc="upper left", fontsize=7.5); ax.set_xlabel("age")
    footer = ("PSID theta. Each window refits partial K-means at K = 3 on the people observed at least three times inside it (with biennial waves, at least four years of a "
              "twenty-year window). Coloured lines: the window's type means over its own age range (tick at the start); grey: the full 20-90 typology. Panel (c): over pairs of "
              "people in both the window and the lifecycle typology (a random 3,000), the correlation between the same-type indicator under the window and under the lifecycle "
              "typology (the adjusted Rand index is in the csv). Panels (d), (e): the share of the person-waves observed at each age that the window assigns to its worst type "
              "(five-year rolling mean), against the lifecycle typology in grey; (f) the worst type's share of each window's people. Types are ordered worst to best within each window.")
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(footer, 175)), fontsize=7.4, color=INK2, va="top")
    share_ylim([axes[0, 0], axes[0, 1]])
    fig.savefig(FIG / "fig_windows.png", bbox_inches="tight")
    r.to_csv(DESC / "windows_agreement.csv", index=False)
    print(r.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
