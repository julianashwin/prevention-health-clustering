"""Figure 7: types over limited age windows, and over windows expanding backwards.

Partial K-means, K = 3, refitted on the people observed at least three times
inside each window (22_paper_kmeans.py). Two rows (h, theta), three columns:
  (a) six sliding twenty-year windows, each window's type means drawn over its
      own age range, with the full-lifecycle types in grey behind
  (b) windows expanding backwards from age 90: 70-90, 60-90, ..., 20-90
  (c) agreement of each window's typology with the full-lifecycle one on the
      people both contain (adjusted Rand index), and the worst type's share

Outputs: paper/figures/fig_windows.png, artifacts/descriptives/paper_windows_agreement.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, load_labels, load_trajectories  # noqa: E402
from _style import CLUSTER, INK2, INK3, apply_style  # noqa: E402

SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(70, 90), (60, 90), (50, 90), (40, 90), (30, 90), (20, 90)]
MIN_SUPPORT = 25


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), gridspec_kw={"wspace": 0.25, "hspace": 0.35})
    rows = []
    for i, v in enumerate(["h", "theta"]):
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
                             "same_vs_full": float((lab.loc[both] == full_lab.loc[both]).mean()),
                             "worst_share": float((lab == 0).mean())})
            ax.set_title(f"({'ab'[col]}) {LABEL[v]}: {name}", loc="left", fontsize=9.5)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if i == 1:
                ax.set_xlabel("age")
        ax = axes[i, 2]
        r = pd.DataFrame(rows)
        r = r[r["variant"] == v]
        for kind, mk, c in [("sliding", "o", CLUSTER[1]), ("backward", "s", CLUSTER[0])]:
            q = r[r["kind"] == kind]
            x = q["lo"] + (q["hi"] - q["lo"]) / 2 if kind == "sliding" else q["lo"]
            ax.plot(x, q["ari_vs_full"], marker=mk, color=c, lw=1.4,
                    label=f"{kind}: ARI vs lifecycle types" + (" (x = window midpoint)" if kind == "sliding" else " (x = start age)"))
        ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        ax.set_title(f"(c) {LABEL[v]}: agreement with the lifecycle typology", loc="left", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=7.5)
        if i == 1:
            ax.set_xlabel("age")
    fig.text(0.01, -0.01,
             "Each window refits K = 3 on the people observed at least three times inside it. Coloured lines: the "
             "window's type means over its own age range (tick at the start); grey: the full 20-90 typology.\n"
             "Panel (c): adjusted Rand index between a window's assignment and the lifecycle assignment, on the "
             "people both contain.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_windows.png")
    out = pd.DataFrame(rows)
    out.to_csv(DESC / "paper_windows_agreement.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
