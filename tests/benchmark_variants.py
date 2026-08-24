"""Paired A/B benchmark of Stan formulations of the same likelihood.

Correctness is checked before timing: the variants must agree on the posterior,
otherwise a speed comparison is meaningless.

Times are wall-clock and therefore sensitive to what else is running. Run the
variants back to back under the same conditions and read the RATIO, not the
absolute seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from cmdstanpy import CmdStanModel

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_inits, build_payload

STAN_DIR = (
    Path(__file__).resolve().parents[1]
    / "src/prevention_health_clustering/models/stan"
)

# Fields each variant's data block accepts.
COMMON = [
    "N_obs", "N_person", "K", "C", "P", "y", "X", "fit_start", "fit_end",
    "homosigma", "anchor_channel", "person_weight_power", "channel_n_obs",
    "alpha_prior_scale", "coef_prior_scale", "sigma_prior_location",
    "sigma_prior_scale", "theta_prior_concentration", "emit_person_quantities", "grainsize",
]

VARIANTS = {
    "row-loop": (
        "mixture_gaussian_panel.stan",
        COMMON + [
            "hold_start", "hold_end", "ar_mode", "age_gap", "N_cohort",
            "cohort_id", "cohort_by_class", "cohort_prior_scale",
            "rho_prior_alpha", "rho_prior_beta",
        ],
    ),
    "vectorised": (
        "mixture_gaussian_vec.stan",
        COMMON + ["N_cohort", "cohort_id", "cohort_prior_scale"],
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--persons", type=int, default=6000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--sampling", type=int, default=250)
    parser.add_argument("--threads-per-chain", type=int, default=2)
    parser.add_argument("--grainsize", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench-variants"))
    args = parser.parse_args(argv)

    spec = get_model("pcs-headline")
    long = pd.read_csv(args.contract / "long.csv")
    keep = long["pidp"].drop_duplicates().head(args.persons)
    long = long[long["pidp"].isin(keep)]
    payload = build_payload(spec, long, grainsize=args.grainsize)
    inits = build_inits(spec, payload)

    n_person, n_obs = payload.data["N_person"], payload.data["N_obs"]
    print(
        f"{n_person:,} persons, {n_obs:,} rows, "
        f"{n_obs / n_person:.2f} obs/person, K={payload.data['K']}, "
        f"grainsize={payload.data['grainsize']}, "
        f"{args.chains} chain(s) x {args.threads_per_chain} threads"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, (stan_name, fields) in VARIANTS.items():
        data = {key: payload.data[key] for key in fields if key in payload.data}
        data_path = args.output / f"data_{label}.json"
        data_path.write_text(json.dumps(data))
        model = CmdStanModel(
            stan_file=str(STAN_DIR / stan_name), cpp_options={"STAN_THREADS": True}
        )
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
            "lp": float(draws["lp__"].mean()),
            "params": {
                name: [float(draws[f"coef[1,{i},{p}]"].mean()) for i in (1, 2, 3)]
                for p, name in ((1, "alpha"), (2, "beta"), (3, "gamma"))
            },
            "theta": [float(draws[f"theta[{i}]"].mean()) for i in (1, 2, 3)],
            "sigma": float(draws["sigma[1,1]"].mean()),
        }
        print(f"  {label:12s} {elapsed:7.1f}s   lp__ {results[label]['lp']:.2f}")

    base = results["row-loop"]
    print("\n=== agreement vs row-loop ===")
    worst = 0.0
    for label, res in results.items():
        if label == "row-loop":
            continue
        for key in ("alpha", "beta", "gamma"):
            for i in range(3):
                worst = max(worst, abs(res["params"][key][i] - base["params"][key][i]))
        for i in range(3):
            worst = max(worst, abs(res["theta"][i] - base["theta"][i]))
        worst = max(worst, abs(res["sigma"] - base["sigma"]))
        print(f"  {label:12s} max |delta| {worst:.6f}   lp__ delta {res['lp'] - base['lp']:+.4f}")

    print("\n=== timing ===")
    for label, res in results.items():
        ratio = base["seconds"] / res["seconds"]
        print(f"  {label:12s} {res['seconds']:7.1f}s   {ratio:5.2f}x vs row-loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
