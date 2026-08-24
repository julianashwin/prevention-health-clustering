"""Check the sufficient-statistic model agrees with the row-loop model, and time both.

Correctness first: both programs must give the same log-density at the same
parameter values, up to floating point. Only then is the timing meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_inits, build_payload

STAN_DIR = (
    Path(__file__).resolve().parents[1]
    / "src/prevention_health_clustering/models/stan"
)


def suffstat_payload(full: dict) -> dict:
    """Project the full data block onto the sufficient-statistic model's inputs."""
    keep = [
        "N_obs", "N_person", "K", "C", "P", "y", "X",
        "fit_start", "fit_end", "anchor_channel", "homosigma",
        "person_weight_power", "channel_n_obs",
        "alpha_prior_scale", "coef_prior_scale",
        "sigma_prior_location", "sigma_prior_scale",
        "theta_prior_concentration", "grainsize",
    ]
    return {key: full[key] for key in keep}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--persons", type=int, default=6000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--sampling", type=int, default=300)
    parser.add_argument("--threads-per-chain", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args(argv)

    spec = get_model("pcs-headline")
    long = pd.read_csv(args.contract / "long.csv")
    keep = long["pidp"].drop_duplicates().head(args.persons)
    long = long[long["pidp"].isin(keep)]

    payload = build_payload(spec, long)
    n_person = payload.data["N_person"]
    n_obs = payload.data["N_obs"]
    print(
        f"{n_person:,} persons, {n_obs:,} rows, "
        f"mean {n_obs / n_person:.2f} obs/person, K={payload.data['K']}"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    full_json = args.output / "data_full.json"
    ss_json = args.output / "data_ss.json"
    full_json.write_text(json.dumps(payload.data))
    ss_json.write_text(json.dumps(suffstat_payload(payload.data)))

    results = {}
    for label, stan_name, data_path in (
        ("row-loop", "mixture_gaussian_panel.stan", full_json),
        ("suff-stat", "mixture_gaussian_suffstat.stan", ss_json),
    ):
        model = CmdStanModel(
            stan_file=str(STAN_DIR / stan_name),
            cpp_options={"STAN_THREADS": True},
        )
        inits = build_inits(spec, payload)
        start = time.time()
        fit = model.sample(
            data=str(data_path),
            inits=inits,
            chains=args.chains,
            iter_warmup=args.warmup,
            iter_sampling=args.sampling,
            adapt_delta=spec.adapt_delta,
            max_treedepth=spec.max_treedepth,
            seed=spec.seed,
            threads_per_chain=args.threads_per_chain,
            output_dir=str(args.output / label),
            show_progress=False,
        )
        elapsed = time.time() - start
        draws = fit.draws_pd()
        results[label] = {
            "seconds": elapsed,
            "lp_mean": float(draws["lp__"].mean()),
            "theta": [float(draws[f"theta[{i}]"].mean()) for i in (1, 2, 3)],
            "alpha": [float(draws[f"coef[1,{i},1]"].mean()) for i in (1, 2, 3)],
            "beta": [float(draws[f"coef[1,{i},2]"].mean()) for i in (1, 2, 3)],
            "gamma": [float(draws[f"coef[1,{i},3]"].mean()) for i in (1, 2, 3)],
            "sigma": float(draws["sigma[1,1]"].mean()),
            "max_rhat": float(fit.summary()["R_hat"].dropna().max()),
        }
        print(f"  {label:10s} {elapsed:7.1f}s   lp__ mean {results[label]['lp_mean']:.2f}")

    a, b = results["row-loop"], results["suff-stat"]
    print("\n=== agreement ===")
    print(f"{'param':10s} {'row-loop':>12s} {'suff-stat':>12s} {'delta':>10s}")
    worst = 0.0
    for key in ("theta", "alpha", "beta", "gamma"):
        for i in range(3):
            delta = b[key][i] - a[key][i]
            worst = max(worst, abs(delta))
            print(f"{key + str(i + 1):10s} {a[key][i]:12.5f} {b[key][i]:12.5f} {delta:10.5f}")
    d_sigma = b["sigma"] - a["sigma"]
    worst = max(worst, abs(d_sigma))
    print(f"{'sigma':10s} {a['sigma']:12.5f} {b['sigma']:12.5f} {d_sigma:10.5f}")
    print(f"\nmax |delta| across parameters: {worst:.6f}")
    print(f"lp__ difference: {b['lp_mean'] - a['lp_mean']:.4f}")
    print(f"max R-hat: row-loop {a['max_rhat']:.5f} | suff-stat {b['max_rhat']:.5f}")

    print("\n=== timing ===")
    print(f"  row-loop   {a['seconds']:7.1f} s")
    print(f"  suff-stat  {b['seconds']:7.1f} s")
    print(f"  speedup    {a['seconds'] / b['seconds']:7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
