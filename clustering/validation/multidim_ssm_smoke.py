"""Convergence smoke for the multidimensional state-space specification.

The first attempt at this -- 2,351 people, 400 warmup, rho initialised at 0.5
-- returned a max R-hat of 1.53 concentrated entirely in rho, with tiny
within-chain standard deviations. That pattern is chains settling in
different tight places rather than a badly behaved posterior, and the
suspected cause was the initialisation: under ar_mode 2 the persistence
posterior sits near 0.95, so starting every chain at 0.5 leaves a long climb
that different chains finish at different points.

This runs the same subsample with the corrected init and a full-length
warmup, and reports rho PER CHAIN, which is what distinguishes the two
explanations. If the chains agree, the earlier failure was the init.

    PYTHONPATH=src .venv/bin/python clustering/validation/multidim_ssm_smoke.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.config import PROCESSED_DATA_DIR
from prevention_health_clustering.runner.fit import compile_model
from prevention_health_clustering.runner.multidim import (
    build_multidim_payload, multidim_inits)

STAN = (Path(__file__).resolve().parents[2] / "src"
        / "prevention_health_clustering" / "models" / "stan"
        / "mixture_multidim_panel.stan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sampling", type=int, default=400)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--threads-per-chain", type=int, default=2)
    ap.add_argument("--mortality", action="store_true", default=True)
    a = ap.parse_args(argv)

    long = pd.read_csv(PROCESSED_DATA_DIR / "contracts"
                       / "multidim_lifecycle_20_89_minobs3_v1" / "long.csv")
    keep = long["pidp"].drop_duplicates().sample(a.people, random_state=7)
    long = long[long["pidp"].isin(set(keep))]
    pl = build_multidim_payload(long, ar_mode=2, min_person_obs=5,
                                holdout_last_k=None,
                                use_mortality=a.mortality)
    d = pl.data
    inits = multidim_inits(pl, n_chains=a.chains)
    print(f"smoke: {d['N_person']:,} people, {d['N_obs']:,} rows, "
          f"mortality={d['use_mortality']}, "
          f"mort events={sum(d['mort_y'])}")
    print(f"rho init per chain: "
          f"{[round(i['rho'][0], 3) for i in inits]}")

    m = compile_model(STAN)
    fit = m.sample(data=d, chains=a.chains, parallel_chains=a.chains,
                   threads_per_chain=a.threads_per_chain,
                   iter_warmup=a.warmup, iter_sampling=a.sampling, seed=99,
                   inits=inits, show_progress=False,
                   adapt_delta=0.95, max_treedepth=12,
                   output_dir="artifacts/multidim-ssm-smoke")

    dr = fit.draws()          # (draws, chains, params)
    names = fit.column_names
    print(f"\n{'param':>14s} " + " ".join(f"{'chain'+str(c+1):>9s}"
                                          for c in range(a.chains))
          + f" {'R-hat':>8s}")
    s = fit.summary()
    for pname in ([f"rho[{k}]" for k in (1, 2, 3)]
                  + [f"theta[{k}]" for k in (1, 2, 3)]
                  + ["sigma_meas[1]", "sigma_meas[2]"]):
        col = names.index(pname.replace("[", ".").replace("]", "")
                          .replace(",", "."))
        per = [dr[:, c, col].mean() for c in range(a.chains)]
        print(f"{pname:>14s} " + " ".join(f"{v:9.4f}" for v in per)
              + f" {s.loc[pname, 'R_hat']:8.4f}")
    core = [i for i in s.index if i.startswith(("theta[", "rho[", "sigma[",
                                                "sigma_meas[", "coef_mort["))]
    worst = s.loc[core, "R_hat"].max()
    div = int(fit.method_variables()["divergent__"].sum())
    print(f"\nmax core R-hat {worst:.4f}   divergences {div}")
    print("VERDICT: " + ("chains agree, the earlier failure was the init"
                         if worst < 1.05 else
                         "still not mixing -- do NOT launch the full batch"))
    return 0 if worst < 1.05 else 1


if __name__ == "__main__":
    sys.exit(main())
