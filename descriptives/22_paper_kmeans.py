"""Partial K-means on the project's health measure: the paper's non-parametric types.

K = 3 over ages 20-90 on each of the three reported variants of the measure
(theta, h, fi10), people with at least three observed ages, 50 seeded starts,
clusters ordered worst -> best by person mean. The same routine is then run on
limited age windows for the window figure: sliding twenty-year windows and
windows that expand backwards from age 90, on h and theta. Every run is a
separate task so the whole set parallelises.

Outputs, artifacts/descriptives/:
  paper_kmeans_labels.parquet        long: variant, lo, hi, pidp, cluster
  paper_kmeans_trajectories.csv      variant, lo, hi, cluster, age, mean, sd, count
  paper_kmeans_composition.csv       variant, lo, hi, age, cluster, share
  paper_kmeans_runs.csv              one row per run: people, shares, seconds

--full-only runs the three lifecycle fits; --windows-only the windows.
Nothing here reads anything but health_measure.parquet.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from multiprocessing import Pool

import pandas as pd

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.trajectories.partial_kmeans import partial_kmeans

MIN_OBS, K = 3, 3
FULL = (20, 90)
SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(60, 90), (50, 90), (40, 90), (30, 90)]
VARIANTS = ["h", "theta", "fi10"]
OUT = ARTIFACTS_DIR / "descriptives"

_panel: pd.DataFrame | None = None


def panel() -> pd.DataFrame:
    global _panel
    if _panel is None:
        p = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                            columns=["pidp", "age", *VARIANTS])
        _panel = p[p["age"].between(*FULL)].assign(age=lambda x: x["age"].astype(int))
    return _panel


def run(task: tuple[str, int, int]):
    variant, lo, hi = task
    t0 = time.time()
    p = panel()
    p = p[p["age"].between(lo, hi)]
    w = p.pivot_table(index="pidp", columns="age", values=variant, aggfunc="mean")
    w = w.reindex(columns=range(lo, hi + 1))
    w = w[w.notna().sum(axis=1) >= MIN_OBS]
    lab = pd.Series(partial_kmeans(w, n_clusters=K), index=w.index)
    order = w.mean(axis=1).groupby(lab).mean().sort_values().index
    lab = lab.map({old: new for new, old in enumerate(order)}).astype(int)
    long = p[p["pidp"].isin(w.index)].assign(cluster=lambda x: x["pidp"].map(lab))
    traj = (long.groupby(["cluster", "age"])[variant].agg(["mean", "std", "count"])
            .reset_index().rename(columns={"std": "sd"}))
    comp = long.groupby(["age", "cluster"]).size().unstack(fill_value=0)
    comp = (comp.div(comp.sum(axis=1), axis=0).reset_index()
            .melt(id_vars="age", var_name="cluster", value_name="share"))
    shares = lab.value_counts(normalize=True).sort_index()
    meta = {"variant": variant, "lo": lo, "hi": hi}
    print(f"  {variant:6s} {lo}-{hi}: {len(w):,} people, shares "
          + " / ".join(f"{s:.3f}" for s in shares) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return (pd.DataFrame({**meta, "pidp": lab.index, "cluster": lab.to_numpy()}),
            traj.assign(**meta), comp.assign(**meta),
            pd.DataFrame([{**meta, "people": len(w), "seconds": round(time.time() - t0),
                           **{f"share_{c + 1}": float(shares.get(c, 0.0)) for c in range(K)}}]))


def main() -> int:
    tasks = []
    if "--windows-only" not in sys.argv:
        tasks += [(v, *FULL) for v in VARIANTS]
    if "--full-only" not in sys.argv:
        tasks += [(v, lo, hi) for v in ("h", "theta") for lo, hi in dict.fromkeys(SLIDING + BACKWARD)]
    print(f"{len(tasks)} runs", flush=True)
    with Pool(int(os.environ.get("KMEANS_PROCS", "3"))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"labels": OUT / "paper_kmeans_labels.parquet",
             "traj": OUT / "paper_kmeans_trajectories.csv",
             "comp": OUT / "paper_kmeans_composition.csv",
             "runs": OUT / "paper_kmeans_runs.csv"}
    new = [pd.concat([r[i] for r in res]) for i in range(4)]
    keys = ["variant", "lo", "hi"]
    done = new[3][keys].drop_duplicates()
    for i, name in enumerate(("labels", "traj", "comp", "runs")):
        f = files[name]
        if f.exists():   # keep runs not repeated this time
            old = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            old = old.merge(done.assign(_r=1), on=keys, how="left")
            old = old[old["_r"].isna()].drop(columns="_r")
            new[i] = pd.concat([old, new[i]])
        (new[i].to_parquet(f, index=False) if f.suffix == ".parquet"
         else new[i].to_csv(f, index=False))
    print(new[3].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
