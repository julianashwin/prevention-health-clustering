"""Figure 7: types over limited age windows, and over windows expanding backwards.

Partial K-means, K = 3, refitted on the people observed at least three times
inside each window (22_paper_kmeans.py). Two rows (h, theta), three columns:
  (a) six sliding twenty-year windows, each window's type means drawn over its
      own age range, with the full-lifecycle types in grey behind
  (b) windows expanding backwards from age 90: 70-90, 60-90, ..., 20-90
  (c) agreement of each window's typology with the full-lifecycle one on the
      people both contain: the co-assignment correlation (over pairs of people, the
      correlation of the two same-cluster indicators, which needs no label
      correspondence; _paper_common.coassignment_corr); the adjusted Rand index goes
      to the csv only; (f) the worst type's share

Outputs: paper/figures/fig_windows.png, artifacts/descriptives/paper_windows_agreement.csv
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, coassignment_corr, load_labels, load_trajectories, onehot  # noqa: E402
from _style import CLUSTER, INK2, INK3, apply_style, share_ylim  # noqa: E402

SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(70, 90), (60, 90), (50, 90), (40, 90), (30, 90), (20, 90)]
MIN_SUPPORT = 25


def main() -> int:
    apply_style()
    # the paper's figure is theta alone; --h the appendix version
    V_ONLY = "h" if "--h" in sys.argv else "theta"
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), gridspec_kw={"wspace": 0.25, "hspace": 0.42}, squeeze=False)
    rows = []
    comp_all = pd.read_csv(DESC / "paper_kmeans_composition.csv")
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate([V_ONLY]):
        full_lab = load_labels(v)
        full_tr = load_trajectories(v)
        for col, (wins, name) in enumerate([(SLIDING, "sliding twenty-year windows"),
                                            (BACKWARD, "windows expanding backwards from 90")]):
            ax = axes[i, col]
            for c in range(K):
                t = full_tr[(full_tr["cluster"] == c) & (full_tr["count"] >= MIN_SUPPORT)]
                ax.plot(t["age"], t["mean"], color=INK3, lw=3.0, alpha=0.35, zorder=1)
            for lo, hi in wins:
                if (lo, hi) == (MIN_AGE, MAX_AGE):
                    continue
                tr = load_trajectories(v, lo, hi)
                lab = load_labels(v, lo, hi)
                for c in range(K):
                    t = tr[(tr["cluster"] == c) & (tr["count"] >= MIN_SUPPORT)]
                    ax.plot(t["age"], t["mean"], color=CLUSTER[c], lw=1.4, zorder=2)
                    ax.plot([t["age"].iloc[0]], [t["mean"].iloc[0]], color=CLUSTER[c], marker="|", ms=7)
                both = lab.index.intersection(full_lab.index)
                rows.append({"variant": v, "kind": "sliding" if col == 0 else "backward", "lo": lo, "hi": hi,
                             "people": len(lab), "shared": len(both),
                             "ari_vs_full": adjusted_rand_score(lab.loc[both], full_lab.loc[both]),
                             "coassign_corr": coassignment_corr(onehot(lab), onehot(full_lab)),
                             "same_vs_full": float((lab.loc[both] == full_lab.loc[both]).mean()),
                             "worst_share": float((lab == 0).mean())})
            ax.set_title(f"({'ab'[col]}) {LABEL[v]}: {name}", loc="left", fontsize=9.5)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            ax.set_xlabel("age")
        ax = axes[i, 2]
        r = pd.DataFrame(rows)
        r = r[r["variant"] == v]
        for kind, mk, c in [("sliding", "o", CLUSTER[1]), ("backward", "s", CLUSTER[0])]:
            q = r[r["kind"] == kind]
            x = q["lo"] + (q["hi"] - q["lo"]) / 2 if kind == "sliding" else q["lo"]
            ax.plot(x, q["coassign_corr"], marker=mk, color=c, lw=1.4,
                    label=f"{kind} windows" + (" (x = window midpoint)" if kind == "sliding" else " (x = start age)"))
        ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        ax.set_title(f"(c) {LABEL[v]}: correlation with lifecycle types", loc="left", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=7.5)
        ax.set_xlabel("age")
        # ---- lower row: the worst type's share of the person-waves observed at each age, window by window
        comp_v = comp_all[(comp_all["variant"] == v) & (comp_all["cluster"] == 0)]
        full_c = comp_v[(comp_v["lo"] == MIN_AGE) & (comp_v["hi"] == MAX_AGE)].set_index("age")["share"].sort_index()
        for col, (wins, name) in enumerate([(SLIDING, "sliding twenty-year windows"), (BACKWARD, "windows expanding backwards from 90")]):
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
        ax.set_title("(f) worst type's share of a window's people", loc="left", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7.5); ax.set_xlabel("age")
    footer = ("Each window refits K = 3 on the people observed at least three times inside it. Coloured lines: the window's type means over its own age range (tick at the start); grey: the full 20-90 typology. Panel (c): over pairs of people in both the window and the lifecycle typology (a random 3,000), the correlation between the same-type indicator under the window and under the lifecycle typology, which needs no correspondence between the two sets of labels (the adjusted Rand index is in the csv). Panels (d), (e): the share of the person-waves observed at each age that the window assigns to its worst type (five-year rolling mean), one line per window, against the lifecycle typology in grey; (f) the worst type's share of each window's people. Types are ordered worst to best within each window.top")
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(footer, 175)), fontsize=7.4, color=INK2, va="top")
    sfx = "" if V_ONLY == "theta" else "_h"
    share_ylim([axes[0, 0], axes[0, 1]])      # the two trajectory panels show the same measure
    fig.savefig(FIG / f"fig_windows{sfx}.png", bbox_inches="tight")
    out = pd.DataFrame(rows)
    out.to_csv(DESC / f"paper_windows_agreement{sfx}.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
