"""Partial K-means on the project's health measure: the paper's non-parametric types.

K = 3 on both reported variants of the measure (theta, h),
on exactly the rows the Bayesian mixtures use: the health contract
(health_lifecycle_20_89_minobs3_v1), ages 20-89, people with at least three
observed ages, one row per person-age (a person interviewed twice at the same
integer age has the two scores averaged). 50 seeded starts, clusters ordered
worst -> best by person mean. The lifecycle run is keyed (lo, hi) = (20, 90)
for the consumers' sake; its rows stop at 89 like the contract. The same routine is then run on
limited age windows for the window figure: sliding twenty-year windows and
windows that expand backwards from age 90, on h and theta. Every run is a
separate task so the whole set parallelises.

Outputs, artifacts/descriptives/:
  paper_kmeans_labels.parquet        long: variant, lo, hi, pidp, cluster
  paper_kmeans_trajectories.csv      variant, lo, hi, cluster, age, mean, sd, count
  paper_kmeans_composition.csv       variant, lo, hi, age, cluster, share
  paper_kmeans_runs.csv              one row per run: people, shares, seconds

A fourth lifecycle run clusters each person's predicted cost, the pooled cost curve
on h evaluated at every contract row (_paper_common.load_predicted_cost_rows): the
cost-anchored cardinalisation of the measure, for the appendix comparison.

--last-held runs theta and h with each person's last two rows held out, for the forecasts of
33_paper_prediction.py. --full-only runs the lifecycle fits; --windows-only the windows; --variants h,cost
restricts either to the named variants. --frailty runs only the lifecycle fits on the
comparison measures, the 31-deficit frailty index and log(frailty + 1/31) (contract columns
``frailty``, ``logfrailty``; higher = frailer, so type 1 is the highest), and adds
them to the same output files under their own variant keys.
Nothing here reads anything but the contract's long.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _paper_common import load_predicted_cost_rows  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from prevention_health_clustering.trajectories.partial_kmeans import partial_kmeans

MIN_OBS, K = 3, 3
FULL = (20, 90)
SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(60, 90), (50, 90), (40, 90), (30, 90)]
VARIANTS = ["h", "theta"]
EXTRA = ["predcost"]             # lifecycle run only
FRAILTY = ["frailty", "logfrailty"]   # comparison measures, --frailty only
# --first-held [k]: theta and h with each person's first k rows (by age) dropped (k = 1 or 2), people
# with at least k + 3 rows, the rows of the first-rows-held-out Bayesian fits
# (clustering/runs/health_firstheld_queue.py, frailty_firstheld_queue.py); stored as variants
# "theta_firstheld", "h_firstheld", "theta_first2held", "h_first2held" for 43_paper_type_from_initial.py
FIRSTHELD = {"theta_firstheld": ("theta", 1), "h_firstheld": ("h", 1),
             "theta_first2held": ("theta", 2), "h_first2held": ("h", 2)}
# --last-held: theta and h with each person's last two rows (by age) dropped, people with at least
# five rows, the rows of the last-rows-held-out Bayesian fits (artifacts/health-next/*-ho) that
# 33_paper_prediction.py forecasts from; stored as variants "theta_last2held", "h_last2held"
LASTHELD = {"theta_last2held": ("theta", 2), "h_last2held": ("h", 2)}
LASTHELD_MIN_OBS = 5
# --denoise: theta with the one-period "spike" taken out before clustering, the K-means analogue of
# reading the covariance matrix off the diagonal (29_paper_moments.py). Two versions, stored as variants:
#   theta_nbr  each observed row replaced by the mean of the person's adjacent observed rows (the one
#              before and the one after; a single neighbour at the ends): the type and the carried
#              component survive (attenuated by rho over the gap), the spike does not
#   theta_ma3  a centred three-row rolling mean within person including the row itself: the spike's
#              variance is cut to a third, the signal kept whole
#   theta_ma5  the same over five rows
#   theta_fit  each person's rows replaced by their own least-squares path in age, linear with three to
#              five rows and quadratic with six or more: the strongest per-person smoothing, the
#              K-means counterpart of a growth mixture's person-level path
DENOISE = {"theta_nbr": "nbr", "theta_ma3": "ma3", "theta_ma5": "ma5", "theta_fit": "fit"}
# --k 4,5: lifecycle K-means on theta with that many types, stored as variants "theta_k4", "theta_k5"
# (53_paper_worst_type.py); the column is theta, the class count is read from the variant name
import re
KVAR = re.compile(r"^(theta|h)_k(\d)$")
# --mental: the mental GRM alone, on the multidim contract's rows (the rows the Bayesian mental fits use)
MENTAL_CONTRACT = PROCESSED_DATA_DIR / "contracts" / "multidim_health_20_89_minobs3_v1"
WORSE_HIGH = {"predcost", "frailty", "logfrailty"}
OUT = ARTIFACTS_DIR / "descriptives"

_panel: dict[str, pd.DataFrame] = {}


def panel(which: str = "health") -> pd.DataFrame:
    if which not in _panel:
        if which == "mental":
            p = pd.read_csv(MENTAL_CONTRACT / "long.csv", usecols=["pidp", "age", "theta_ment_nodepr"]).rename(columns={"theta_ment_nodepr": "mental"})
        elif which in DENOISE:
            p = pd.read_csv(PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1" / "long.csv",
                            usecols=["pidp", "age", "theta"]).sort_values(["pidp", "age"]).reset_index(drop=True)
            g = p.groupby("pidp")["theta"]
            kind = DENOISE[which]
            if kind == "nbr":
                prev, nxt = g.shift(1), g.shift(-1)
                p[which] = pd.concat([prev, nxt], axis=1).mean(axis=1)        # mean of the neighbours present
            elif kind in ("ma3", "ma5"):
                win = 3 if kind == "ma3" else 5
                p[which] = g.transform(lambda x: x.rolling(win, center=True, min_periods=2).mean())
            else:
                import numpy as np
                def fit_person(df):
                    a = df["age"].to_numpy(float); y = df["theta"].to_numpy(float)
                    deg = 2 if len(a) >= 6 else 1 if len(a) >= 3 else 0
                    return pd.Series(np.polyval(np.polyfit(a - a.mean(), y, deg), a - a.mean()), index=df.index)
                p[which] = p.groupby("pidp", group_keys=False)[["age", "theta"]].apply(fit_person)
            p = p.dropna(subset=[which]).drop(columns="theta")
        elif which in FIRSTHELD:
            v, k = FIRSTHELD[which]
            p = pd.read_csv(PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1" / "long.csv",
                            usecols=["pidp", "age", v]).sort_values(["pidp", "age"])
            n = p.groupby("pidp")["age"].transform("size")
            p = p[(n >= k + 3) & (p.groupby("pidp").cumcount() >= k)].rename(columns={v: which})
        elif which in LASTHELD:
            v, k = LASTHELD[which]
            p = pd.read_csv(PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1" / "long.csv",
                            usecols=["pidp", "age", v]).sort_values(["pidp", "age"])
            n = p.groupby("pidp")["age"].transform("size")
            p = p[(n >= LASTHELD_MIN_OBS) & (p.groupby("pidp").cumcount() < n - k)].rename(columns={v: which})
        else:
            p = pd.read_csv(PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1" / "long.csv",
                            usecols=["pidp", "age", *VARIANTS, *FRAILTY])
        _panel[which] = p[p["age"].between(*FULL)].assign(age=lambda x: x["age"].astype(int))
    return _panel[which]


def run(task: tuple[str, int, int]):
    variant, lo, hi = task
    t0 = time.time()
    kv = KVAR.match(variant)
    col, k = (kv.group(1), int(kv.group(2))) if kv else (variant, K)
    p = load_predicted_cost_rows("h") if variant == "predcost" else panel(variant if variant == "mental" or variant in FIRSTHELD or variant in LASTHELD or variant in DENOISE else "health")
    p = p[p["age"].between(lo, hi)]
    if kv:
        p = p.rename(columns={col: variant})
    w = p.pivot_table(index="pidp", columns="age", values=variant, aggfunc="mean")
    w = w.reindex(columns=range(lo, hi + 1))
    w = w[w.notna().sum(axis=1) >= MIN_OBS]
    lab = pd.Series(partial_kmeans(w, n_clusters=k), index=w.index)
    # type 1 is worst health: lowest score, or for the cost index the highest cost
    order = w.mean(axis=1).groupby(lab).mean().sort_values(ascending=(variant not in WORSE_HIGH)).index
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
                           **{f"share_{c + 1}": float(shares.get(c, 0.0)) for c in range(k)}}]))


def main() -> int:
    only = None
    if "--variants" in sys.argv:
        only = sys.argv[sys.argv.index("--variants") + 1].split(",")
    tasks = []
    if "--frailty" in sys.argv:
        tasks = [(v, *FULL) for v in FRAILTY]
    elif "--mental" in sys.argv:
        tasks = [("mental", *FULL)]
    elif "--denoise" in sys.argv:
        tasks = [(v, *FULL) for v in DENOISE if only is None or v in only]
    elif "--k" in sys.argv:
        tasks = [(f"theta_k{k}", *FULL) for k in sys.argv[sys.argv.index("--k") + 1].split(",")]
    elif "--first-held" in sys.argv:
        # --first-held [k] [--variants theta,h]: default both measures, k = 1
        nxt = sys.argv[sys.argv.index("--first-held") + 1:]
        k = int(nxt[0]) if nxt and nxt[0].isdigit() else 1
        tasks = [(f"{v}_first{'' if k == 1 else k}held", *FULL) for v in ("theta", "h") if only is None or v in only]
    elif "--last-held" in sys.argv:
        tasks = [(f"{v}_last2held", *FULL) for v in ("theta", "h") if only is None or v in only]
    elif "--windows-only" not in sys.argv:
        tasks += [(v, *FULL) for v in VARIANTS + EXTRA if only is None or v in only]
    if "--full-only" not in sys.argv and not any(f in sys.argv for f in ("--frailty", "--mental", "--first-held", "--last-held", "--denoise", "--k")):
        tasks += [(v, lo, hi) for v in ("h", "theta") if only is None or v in only for lo, hi in dict.fromkeys(SLIDING + BACKWARD)]
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
            old = old[old["_r"].isna() & (old["variant"] != "cost")].drop(columns="_r")   # realised-cost runs retired
            new[i] = pd.concat([old, new[i]])
        (new[i].to_parquet(f, index=False) if f.suffix == ".parquet"
         else new[i].to_csv(f, index=False))
    print(new[3].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
