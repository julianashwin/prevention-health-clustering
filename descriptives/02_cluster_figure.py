"""Partial K-means trajectory clusters across measures (Figure 5 analogue).

Re-runs the predecessor pipeline's partial K-means (K = 3, 50 seeded inits)
on each measure over ages 20-90 (the GRM window; the original figure used
16-90), holding the person sample fixed at those eligible on the baseline
metric (>= 3 observed ages of sf12pcs_dv). Panels show cluster mean
trajectories where a cluster has >= 25 people observed at that age, with a
composition strip underneath; the agreement table records ARI/AMI against the
baseline solution.

Outputs: docs/figures/fig_cluster_trajectories.png,
         artifacts/descriptives/cluster_agreement.csv (+ labels parquet)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR
from prevention_health_clustering.trajectories.partial_kmeans import partial_kmeans

MIN_AGE, MAX_AGE, MIN_OBS, K, MIN_SUPPORT = 20, 90, 3, 3, 25

MEASURES = {
    "sf12pcs_dv": "UKHLS PCS (baseline)",
    "PCS_phys_only": "Physical subscales only",
    "sf6d_utility": "SF-6D utility",
    "grm_theta": "GRM physical, 4 testlets (original)",
    "theta_phys_full": "GRM-2 physical (+function, +conditions)",
    "theta_ment": "GRM-2 mental",
}


def main() -> int:
    apply_style()
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=["pidp", "age"] + list(MEASURES))
    panel = panel[panel["age"].notna()]
    panel["age"] = panel["age"].astype(int)
    panel = panel[panel["age"].between(MIN_AGE, MAX_AGE)]

    def wide(metric):
        w = panel.pivot_table(index="pidp", columns="age", values=metric,
                              aggfunc="mean")
        return w.reindex(columns=range(MIN_AGE, MAX_AGE + 1))

    base_wide = wide("sf12pcs_dv")
    common = base_wide.index[base_wide.notna().sum(axis=1) >= MIN_OBS]
    print(f"eligible on baseline: {len(common):,} people")

    labels, trajs, comps = {}, {}, {}
    for metric in MEASURES:
        # reindex, not .loc: inventory-gated measures lack some baseline-
        # eligible people entirely; they stay as all-NaN rows and are dropped
        # by the >= 1 observation guard below.
        w = wide(metric).reindex(common)
        observed = w.notna().sum(axis=1)
        keep = observed > 0
        lab = pd.Series(index=w.index, dtype=float)
        lab.loc[keep] = partial_kmeans(w.loc[keep], n_clusters=K)
        # order clusters worst -> best by person mean level (SF-6D and GRMs
        # are increasing-in-health, as are all measures here)
        pm = w.mean(axis=1)
        order = pm.groupby(lab).mean().sort_values().index
        lab = lab.map({old: new for new, old in enumerate(order)})
        labels[metric] = lab
        long = panel[["pidp", "age", metric]].dropna()
        long = long[long["pidp"].isin(w.index[keep])]
        long["cluster"] = long["pidp"].map(lab)
        g = long.groupby(["cluster", "age"])[metric]
        trajs[metric] = g.agg(["mean", "count"]).reset_index()
        comp = (long.groupby(["age", "cluster"]).size().unstack(fill_value=0))
        comps[metric] = comp.div(comp.sum(axis=1), axis=0)
        sizes = lab.value_counts().sort_index()
        print(f"  {metric:18s} clusters " +
              " / ".join(f"{int(sizes.get(c, 0)):,}" for c in range(K)))

    # ---- agreement table ----------------------------------------------------
    rows = []
    base = labels["sf12pcs_dv"]
    for metric in MEASURES:
        both = base.notna() & labels[metric].notna()
        rows.append({
            "measure": metric,
            "n": int(both.sum()),
            "ARI_vs_baseline": adjusted_rand_score(base[both], labels[metric][both]),
            "AMI_vs_baseline": adjusted_mutual_info_score(base[both], labels[metric][both]),
            "pct_same": float((base[both] == labels[metric][both]).mean()),
        })
    agree = pd.DataFrame(rows)
    out_dir = ARTIFACTS_DIR / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)
    agree.to_csv(out_dir / "cluster_agreement.csv", index=False)
    pd.DataFrame(labels).to_parquet(out_dir / "cluster_labels.parquet")
    print(agree.round(3).to_string(index=False))

    # ---- figure -------------------------------------------------------------
    n = len(MEASURES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(11.5, 4.1 * nrows))
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3.2, 0.75] * nrows,
                          hspace=0.45, wspace=0.25)
    for i, (metric, title) in enumerate(MEASURES.items()):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[2 * r, c])
        axc = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        t = trajs[metric]
        for cl in range(K):
            s = t[(t["cluster"] == cl) & (t["count"] >= MIN_SUPPORT)]
            ax.plot(s["age"], s["mean"], color=CLUSTER[cl], lw=1.8,
                    label=f"cluster {cl + 1}")
        ax.set_title(title)
        ax.grid(True, axis="y")
        if i == 0:
            ax.legend(loc="lower left", ncols=1)
        comp = comps[metric].reindex(columns=range(K), fill_value=0)
        axc.stackplot(comp.index, *[comp[cl] for cl in range(K)],
                      colors=CLUSTER, alpha=0.9)
        axc.set_ylim(0, 1)
        axc.set_yticks([])
        axc.set_xlabel("age")
        axc.set_xlim(MIN_AGE, MAX_AGE)
        for spine in ("left",):
            axc.spines[spine].set_visible(False)
    fig.suptitle("Partial K-means trajectory clusters, K = 3, common sample",
                 fontweight="bold", y=1.005)
    fig.text(0.01, -0.01,
             "Sample fixed at people with ≥ 3 observed ages of the baseline PCS, ages 20–90. "
             "Cluster curves shown where ≥ 25 members observed; strips show the cluster "
             "composition of the observed sample at each age.",
             fontsize=7.5, color=INK2)
    fig_path = ROOT_DIR / "docs" / "figures" / "fig_cluster_trajectories.png"
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
