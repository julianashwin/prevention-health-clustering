"""Partial K-means, K = 3, on the twelve limitation-bank scores, h scale.

Settings of descriptives/02_cluster_figure.py: ages 20-90, people with at least
three observed ages, 50 seeded starts, clusters ordered worst to best by person
mean. Trajectories are cluster means by age with member counts; composition is
each cluster's share of the observed sample at each age. The twelve scores share
P-FUNC's person-waves, so the same people are clustered on every score.

Writes, in data/processed/baseline_measures/ (gitignored):
  limitation_cluster_labels.parquet, limitation_cluster_trajectories.csv,
  limitation_cluster_composition.csv, limitation_cluster_agreement.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
import time
from multiprocessing import Pool

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from prevention_health_clustering.config import PROCESSED_DATA_DIR
from prevention_health_clustering.trajectories.partial_kmeans import partial_kmeans

MIN_AGE, MAX_AGE, MIN_OBS, K = 20, 90, 3, 3
SLUG = {"P-FUNC": "pfunc", "P-LIM1": "plim1", "P-LIM": "plim", "P-LIM4": "plim4",
        "P-LIM3": "plim3", "P-LIM3+O": "plim3o"}
MEASURES = [f"h_{s}_{m}" for s in SLUG.values() for m in ("grm", "gpcm")] + \
    [f"h_{s}{cd}_grm" for b, s in SLUG.items() if b != "P-FUNC" for cd in ("cc", "cg")]
OUT = PROCESSED_DATA_DIR / "baseline_measures"

panel = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "limitation_scores.parquet",
                        columns=["pidp", "age", *MEASURES])
panel = panel[panel["age"].between(MIN_AGE, MAX_AGE)].copy()
panel["age"] = panel["age"].astype(int)


def wide(metric: str) -> pd.DataFrame:
    w = panel.pivot_table(index="pidp", columns="age", values=metric, aggfunc="mean")
    return w.reindex(columns=range(MIN_AGE, MAX_AGE + 1))


def run(metric: str):
    t0 = time.time()
    w = wide(metric)
    w = w[w.notna().sum(axis=1) >= MIN_OBS]
    lab = pd.Series(partial_kmeans(w, n_clusters=K), index=w.index)
    order = w.mean(axis=1).groupby(lab).mean().sort_values().index
    lab = lab.map({old: new for new, old in enumerate(order)})
    long = panel[["pidp", "age", metric]].dropna()
    long = long[long["pidp"].isin(w.index)].assign(cluster=lambda x: x["pidp"].map(lab))
    traj = (long.groupby(["cluster", "age"])[metric].agg(["mean", "count"]).reset_index()
            .assign(measure=metric))
    comp = long.groupby(["age", "cluster"]).size().unstack(fill_value=0)
    comp = (comp.div(comp.sum(axis=1), axis=0).reset_index()
            .melt(id_vars="age", var_name="cluster", value_name="share").assign(measure=metric))
    print(f"  {metric:16s} {len(w):,} people, sizes "
          + " / ".join(f"{int(n):,}" for n in lab.value_counts().sort_index())
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    return metric, lab.rename(metric), traj, comp


def main() -> int:
    # incremental: scores already clustered are read back rather than refitted
    # (partial_kmeans is seeded, so a refit reproduces them); --all refits
    lab_f, traj_f, comp_f = (OUT / "limitation_cluster_labels.parquet",
                             OUT / "limitation_cluster_trajectories.csv",
                             OUT / "limitation_cluster_composition.csv")
    done = [] if "--all" in sys.argv or not lab_f.exists() else \
        [m for m in pd.read_parquet(lab_f).columns if m in MEASURES]
    todo = [m for m in MEASURES if m not in done]
    print(f"clustering {len(todo)} scores; reusing {len(done)}", flush=True)
    with Pool(6) as pool:
        res = pool.map(run, todo) if todo else []
    labels = pd.concat([pd.read_parquet(lab_f)[done]] + [r[1] for r in res], axis=1) if done else \
        pd.concat([r[1] for r in res], axis=1)
    traj = pd.concat(([pd.read_csv(traj_f).query("measure in @done")] if done else []) + [r[2] for r in res])
    comp = pd.concat(([pd.read_csv(comp_f).query("measure in @done")] if done else []) + [r[3] for r in res])
    traj.to_csv(traj_f, index=False)
    comp.to_csv(comp_f, index=False)
    labels.to_parquet(lab_f)
    rows = []
    base = labels["h_pfunc_grm"]
    for bank, s in SLUG.items():
        g, p = labels[f"h_{s}_grm"], labels[f"h_{s}_gpcm"]
        rows.append({"bank": bank, "people": len(g),
                     "ari_grm_vs_gpcm": adjusted_rand_score(g, p),
                     "same_cluster_grm_vs_gpcm": float((g == p).mean()),
                     "ari_grm_vs_pfunc_grm": adjusted_rand_score(base, g),
                     "ari_gpcm_vs_pfunc_grm": adjusted_rand_score(base, p),
                     **{f"share_grm_{c + 1}": float((g == c).mean()) for c in range(K)},
                     **{f"share_gpcm_{c + 1}": float((p == c).mean()) for c in range(K)}})
    agree = pd.DataFrame(rows)
    agree.to_csv(OUT / "limitation_cluster_agreement.csv", index=False)
    print(agree.round(3).to_string(index=False))
    rows = []
    for bank, s in SLUG.items():
        if bank == "P-FUNC":
            continue
        cc, cg, base_b = labels[f"h_{s}cc_grm"], labels[f"h_{s}cg_grm"], labels[f"h_{s}_grm"]
        both = cc.notna() & cg.notna() & base_b.notna()
        rows.append({"bank": bank, "people": int((cc.notna()).sum()),
                     "ari_cc_vs_cg": adjusted_rand_score(cc[both], cg[both]),
                     "same_cluster_cc_vs_cg": float((cc[both] == cg[both]).mean()),
                     "ari_cc_vs_no_conditions": adjusted_rand_score(cc[both], base_b[both]),
                     "ari_cg_vs_no_conditions": adjusted_rand_score(cg[both], base_b[both]),
                     **{f"share_cc_{c + 1}": float((cc == c).sum() / cc.notna().sum()) for c in range(K)},
                     **{f"share_cg_{c + 1}": float((cg == c).sum() / cg.notna().sum()) for c in range(K)}})
    agree_c = pd.DataFrame(rows)
    agree_c.to_csv(OUT / "limitation_cluster_agreement_conditions.csv", index=False)
    print(agree_c.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
