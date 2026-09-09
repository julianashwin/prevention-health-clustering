"""Generic registry-model fit on a frozen contract, with optional holdout.

One invocation = one fit. Designed for unattended batch runs: partition
inits with jitter (the honest dispersed recipe), structural-parameter
diagnostics computed WITHOUT invoking stansummary over person-level columns,
and per-person held-out predictive densities extracted directly when a
holdout is requested.

    python clustering/run_fit.py --model pcs-ar1 \
        --contract data/processed/contracts/pcs_lifecycle_20_89_minobs3_v1 \
        --output artifacts/overnight/pcs-ar1 \
        [--holdout-last-k 2 --holdout-min-obs 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import (
    assignment_inits,
    build_payload,
    fit_model,
)


def split_rhat(x: np.ndarray) -> float:
    """Classic split-R-hat over (chains, draws); no rank-normalisation."""
    c, d = x.shape
    half = d // 2
    parts = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    m = parts.mean(axis=1)
    w = parts.var(axis=1, ddof=1).mean()
    b = half * m.var(ddof=1)
    var_plus = (half - 1) / half * w + b / half
    return float(np.sqrt(var_plus / w)) if w > 0 else np.inf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--parallel-chains", type=int, default=0,
                        help="Chains to run at once; 0 means all of them. "
                             "Use to cap the CPU footprint on a shared machine.")
    parser.add_argument("--init-jitter", type=float, default=0.15)
    parser.add_argument("--holdout-last-k", type=int, default=None)
    parser.add_argument("--holdout-min-obs", type=int, default=None)
    args = parser.parse_args(argv)

    t0 = time.time()
    spec = get_model(args.model)
    long = pd.read_csv(args.contract / "long.csv")
    holdout = args.holdout_last_k is not None
    payload = build_payload(
        spec, long,
        holdout_last_k=args.holdout_last_k,
        holdout_min_person_obs=args.holdout_min_obs,
        emit_person_quantities=holdout,
    )
    d = payload.data
    n_held = sum(
        d["hold_end"][i] - d["hold_start"][i] + 1
        for i in range(d["N_person"]) if d["hold_end"][i] > 0
    )
    print(f"[{args.model}] {d['N_person']:,} persons, {d['N_obs']:,} rows, "
          f"ar_mode={d['ar_mode']}, held-out rows {n_held:,}, "
          f"dropped {d.get('_dropped_no_history', 0):,}", flush=True)

    inits = assignment_inits(spec, payload, jitter=args.init_jitter)
    print(f"  partition init theta {[round(v, 3) for v in inits[0]['theta']]}",
          flush=True)
    fit = fit_model(
        spec, payload, inits=inits, output_dir=args.output,
        chains=args.chains, iter_warmup=args.warmup,
        iter_sampling=args.sampling,
        threads_per_chain=args.threads_per_chain,
        **({"parallel_chains": args.parallel_chains}
           if args.parallel_chains > 0 else {}),
    )
    wall = time.time() - t0
    print(f"  sampling done in {wall / 3600:.2f} h", flush=True)

    # -- structural diagnostics without touching person-level columns --------
    k, c = spec.n_classes, spec.n_channels
    rho_names = ([f"rho[{i}]" for i in range(1, k + 1)]
                 if d["ar_mode"] != 0 else [])
    # ar_mode 2 adds one measurement-error scale per channel.
    meas_names = ([f"sigma_meas[{ci}]" for ci in range(1, c + 1)]
                  if d["ar_mode"] == 2 else [])
    names = ([f"theta[{i}]" for i in range(1, k + 1)]
             + [f"coef[{ci},{i},{p}]" for ci in range(1, c + 1)
                for i in range(1, k + 1) for p in range(1, 4)]
             + [f"sigma[1,{ci}]" for ci in range(1, c + 1)]
             + rho_names + meas_names + ["lp__"])
    draws = fit.draws_pd(vars=["theta", "coef", "sigma", "lp__"]
                         + (["rho"] if d["ar_mode"] != 0 else [])
                         + (["sigma_meas"] if d["ar_mode"] == 2 else []))
    n_chain, n_draw = args.chains, args.sampling
    summary = {}
    worst_rhat, worst_name = 0.0, ""
    for name in names:
        col = draws[name].to_numpy().reshape(n_chain, n_draw)
        r = split_rhat(col)
        summary[name] = {"mean": float(col.mean()), "sd": float(col.std()),
                         "rhat": r}
        if name != "lp__" and r > worst_rhat:
            worst_rhat, worst_name = r, name
    print(f"  max structural R-hat {worst_rhat:.4f} ({worst_name})", flush=True)
    print("  theta " + " ".join(
        f"{summary[f'theta[{i}]']['mean']:.3f}" for i in range(1, k + 1)))
    if d["ar_mode"] != 0:
        print("  rho " + " ".join(
            f"{summary[n]['mean']:.3f}" for n in rho_names))
    if d["ar_mode"] == 2:
        print("  sigma_meas " + " ".join(
            f"{summary[n]['mean']:.3f}" for n in meas_names))
        # How much of the observed year-to-year variance is persistent signal
        # rather than measurement noise, per class. This is the number the
        # specification exists to produce.
        sig = summary[f"sigma[1,1]"]["mean"]
        mea = summary[meas_names[0]]["mean"]
        shares = [(sig ** 2 / (1 - summary[n]["mean"] ** 2))
                  / (sig ** 2 / (1 - summary[n]["mean"] ** 2) + mea ** 2)
                  for n in rho_names]
        print("  signal share " + " ".join(f"{x:.3f}" for x in shares))

    result = {
        "model": args.model, "contract": str(args.contract),
        "n_person": int(d["N_person"]), "n_obs": int(d["N_obs"]),
        "held_rows": int(n_held), "wall_hours": wall / 3600,
        "chains": args.chains, "warmup": args.warmup,
        "sampling": args.sampling, "max_structural_rhat": worst_rhat,
        "worst_param": worst_name, "params": summary,
    }

    # -- held-out predictive density -----------------------------------------
    if holdout:
        ll = fit.stan_variable("log_lik_heldout")      # (draws, N_person)
        lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
        out = pd.DataFrame({"pidp": payload.person_ids, "lpd_heldout": lpd})
        out.to_parquet(args.output / "heldout_lpd.parquet", index=False)
        result["heldout_mean_lpd"] = float(lpd.mean())
        result["heldout_total_lpd"] = float(lpd.sum())
        print(f"  held-out mean LPD {lpd.mean():.4f} over "
              f"{len(lpd):,} persons", flush=True)

    (args.output / "run_summary.json").write_text(json.dumps(result, indent=1))
    ok = worst_rhat < 1.05
    print(f"[{args.model}] {'CONVERGED' if ok else 'R-HAT FLAG'} "
          f"in {wall / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
