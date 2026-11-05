"""Partial K-means on PSID theta: the paper's non-parametric types, replicated.

Same algorithm and settings as descriptives/22_paper_kmeans.py (whose run() this copies,
pointed at PSID rows): prevention_health_clustering.trajectories.partial_kmeans with 50
seeded starts, K = 3, on the PSID contract (ages 20-89, one row per person-age, people with
at least three observed ages), types ordered worst to best by person mean. Runs:

  theta                  the lifecycle typology (keyed lo, hi = 20, 90)
  theta_k4, theta_k5     the same with K = 4 and 5 (the K-means stand-in for Figure 7)
  windows                six sliding twenty-year windows and windows expanding backwards
                         from 90, each on the people observed at least three times inside it
  theta_first2held       each person's first two rows dropped, people with at least five rows
                         (the types of the type-prediction and targeting exercises)
  theta_last2held        each person's last two rows dropped, people with at least five rows
                         (the K-means forecast of the predicting-ahead exercise)

Outputs, artifacts/descriptives/psid/: psid_kmeans_labels.parquet, psid_kmeans_trajectories.csv,
psid_kmeans_composition.csv, psid_kmeans_runs.csv.
"""

from __future__ import annotations

import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from _psid_common import KM_COMP, KM_LABELS, KM_RUNS, KM_TRAJ, contract  # noqa: E402
from prevention_health_clustering.trajectories.partial_kmeans import partial_kmeans  # noqa: E402

MIN_OBS, K = 3, 3
FULL = (20, 90)
SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(60, 90), (50, 90), (40, 90), (30, 90)]
HELD_MIN_OBS = 5
KVAR = re.compile(r"^theta_k(\d)$")


def rows(variant: str) -> pd.DataFrame:
    p = contract()[["pidp", "age", "theta"]]
    if variant in ("theta_first2held", "theta_last2held"):
        n = p.groupby("pidp")["age"].transform("size")
        pos = p.groupby("pidp").cumcount()
        keep = (n >= HELD_MIN_OBS) & ((pos >= 2) if variant == "theta_first2held" else (pos < n - 2))
        p = p[keep]
    return p.rename(columns={"theta": variant})


def run(task):
    variant, lo, hi = task
    t0 = time.time()
    kv = KVAR.match(variant)
    k = int(kv.group(1)) if kv else K
    p = rows(variant)
    p = p[p["age"].between(lo, hi)]
    w = p.pivot_table(index="pidp", columns="age", values=variant, aggfunc="mean").reindex(columns=range(lo, hi + 1))
    w = w[w.notna().sum(axis=1) >= MIN_OBS]
    lab = pd.Series(partial_kmeans(w, n_clusters=k), index=w.index)
    order = w.mean(axis=1).groupby(lab).mean().sort_values().index          # type 1 (0) is worst health
    lab = lab.map({old: new for new, old in enumerate(order)}).astype(int)
    long = p[p["pidp"].isin(w.index)].assign(cluster=lambda x: x["pidp"].map(lab))
    traj = long.groupby(["cluster", "age"])[variant].agg(["mean", "std", "count"]).reset_index().rename(columns={"std": "sd"})
    comp = long.groupby(["age", "cluster"]).size().unstack(fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0).reset_index().melt(id_vars="age", var_name="cluster", value_name="share")
    shares = lab.value_counts(normalize=True).sort_index()
    meta = {"variant": variant, "lo": lo, "hi": hi}
    print(f"  {variant:17s} {lo}-{hi}: {len(w):,} people, shares " + " / ".join(f"{s:.3f}" for s in shares)
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    return (pd.DataFrame({**meta, "pidp": lab.index, "cluster": lab.to_numpy()}), traj.assign(**meta), comp.assign(**meta),
            pd.DataFrame([{**meta, "people": len(w), "seconds": round(time.time() - t0),
                           **{f"share_{c + 1}": float(shares.get(c, 0.0)) for c in range(k)}}]))


def main() -> int:
    t0 = time.time()
    tasks = [("theta", *FULL), ("theta_k4", *FULL), ("theta_k5", *FULL), ("theta_first2held", *FULL), ("theta_last2held", *FULL)]
    tasks += [("theta", lo, hi) for lo, hi in SLIDING + BACKWARD]
    with Pool(int(os.environ.get("KMEANS_PROCS", "6"))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    out = [pd.concat([r[i] for r in res]) for i in range(4)]
    out[0].to_parquet(KM_LABELS, index=False)
    out[1].to_csv(KM_TRAJ, index=False)
    out[2].to_csv(KM_COMP, index=False)
    out[3].to_csv(KM_RUNS, index=False)
    print(out[3].round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
