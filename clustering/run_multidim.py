"""Fit the multidimensional mixture (physical, mental, mortality) on a frozen contract.

    python clustering/run_multidim.py --variant ssm --physical theta \
        --with-mortality --output artifacts/multidim-health/theta-ssm-mort

Variants: baseline | holdout | ar1 | ar1-holdout | ssm | ssm-holdout.
Channels: ``--physical`` picks the physical ruler (theta or h on the paper's
contract; theta_phys_func on the archived one) and ``--mental`` the mental
score; ``--with-chronic`` adds the negative-binomial condition count where the
contract carries it. Mortality is Gompertz-Makeham (see the Stan header).
Structural diagnostics are computed without touching person-level columns;
per-person held-out densities are written when a holdout is requested.

The earlier four-channel fits in artifacts/multidim* were run by the previous
version of this script (logit-quadratic hazard, one persistence per class);
their stan_data.json files no longer match the current Stan source.
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

from prevention_health_clustering.config import PROCESSED_DATA_DIR
from prevention_health_clustering.runner.fit import compile_model
from prevention_health_clustering.runner.multidim import (
    build_multidim_payload,
    multidim_inits,
    write_multidim_data,
)

STAN = (Path(__file__).resolve().parents[1] / "src"
        / "prevention_health_clustering" / "models" / "stan"
        / "mixture_multidim_panel.stan")
CONTRACT = (PROCESSED_DATA_DIR / "contracts"
            / "multidim_health_20_89_minobs3_v1")
VARIANTS = {
    "baseline":    dict(ar_mode=0, holdout=False),
    "holdout":     dict(ar_mode=0, holdout=True),
    "ar1":         dict(ar_mode=1, holdout=False),
    "ar1-holdout": dict(ar_mode=1, holdout=True),
    # AR(1) latent state plus a one-period "spike" on the Gaussian channels;
    # the chronic count and mortality hazard are unaffected.
    "ssm":         dict(ar_mode=2, holdout=False),
    "ssm-holdout": dict(ar_mode=2, holdout=True),
}


def split_rhat(x: np.ndarray) -> float:
    c, d = x.shape
    half = d // 2
    parts = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    m = parts.mean(axis=1)
    w = parts.var(axis=1, ddof=1).mean()
    b = half * m.var(ddof=1)
    return float(np.sqrt(((half - 1) / half * w + b / half) / w)) if w > 0 else np.inf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--physical", default="theta")
    parser.add_argument("--mental", default="theta_ment_nodepr")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--parallel-chains", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--holdout-last-k", type=int, default=2)
    parser.add_argument("--min-obs", type=int, default=None,
                        help="minimum rows per person, applied to every variant; "
                             "holdout variants default to holdout-last-k + 3")
    parser.add_argument("--no-mortality", action="store_true", default=True)
    parser.add_argument("--with-mortality", dest="no_mortality", action="store_false")
    parser.add_argument("--with-chronic", action="store_true")
    parser.add_argument("--init-jitter", type=float, default=0.15)
    parser.add_argument("--max-persons", type=int, default=0, help="smoke test: first N people only")
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = VARIANTS[args.variant]
    long = pd.read_csv(args.contract / "long.csv")
    if args.max_persons > 0:
        keep = long["pidp"].drop_duplicates().iloc[: args.max_persons]
        long = long[long["pidp"].isin(keep)]
    min_obs = args.min_obs if args.min_obs is not None else (args.holdout_last_k + 3 if cfg["holdout"] else None)
    channels = (args.physical, args.mental)
    payload = build_multidim_payload(
        long, gauss_channels=channels, n_classes=args.classes, ar_mode=cfg["ar_mode"],
        holdout_last_k=args.holdout_last_k if cfg["holdout"] else None,
        min_person_obs=min_obs, use_chronic=args.with_chronic,
        use_mortality=not args.no_mortality,
        emit_person_quantities=cfg["holdout"])
    d = payload.data
    n_held = sum(d["hold_end"][i] - d["hold_start"][i] + 1
                 for i in range(d["N_person"]) if d["hold_end"][i] > 0)
    print(f"[{args.variant} {'+'.join(channels)}] {d['N_person']:,} persons, {d['N_obs']:,} rows, "
          f"ar_mode={d['ar_mode']}, held-out {n_held:,}, "
          f"chronic {'on' if d['use_chronic'] else 'off'}, "
          f"mortality {'on' if d['use_mortality'] else 'OFF'} ({int(np.sum(d['mort_y'])):,} events)",
          flush=True)

    inits = multidim_inits(payload, jitter=args.init_jitter, n_chains=args.chains)
    print(f"  partition init theta {[round(v, 3) for v in inits[0]['theta']]}", flush=True)
    if d["use_mortality"]:
        print(f"  partition init log_b_raw {[round(v, 2) for v in inits[0]['log_b_raw']]}", flush=True)
    model = compile_model(STAN)
    data_path = write_multidim_data(payload, args.output)
    fit = model.sample(
        data=str(data_path), inits=inits, chains=args.chains,
        iter_warmup=args.warmup, iter_sampling=args.sampling,
        adapt_delta=0.9, max_treedepth=12, seed=20260826,
        threads_per_chain=args.threads_per_chain,
        **({"parallel_chains": args.parallel_chains} if args.parallel_chains > 0 else {}),
        output_dir=str(args.output / "chains"), show_progress=False)
    wall = time.time() - t0
    print(f"  sampling done in {wall / 3600:.2f} h", flush=True)

    K, C, P = d["K"], d["C"], d["P"]
    names = ([f"theta[{k}]" for k in range(1, K + 1)]
             + [f"coef[{c},{k},{p}]" for c in range(1, C + 1)
                for k in range(1, K + 1) for p in range(1, P + 1)]
             + [f"sigma[1,{c}]" for c in range(1, C + 1)]
             + ([f"coef_chronic[{k},{p}]" for k in range(1, K + 1) for p in range(1, P + 1)]
                + ["phi_chronic[1]"] if d["use_chronic"] else [])
             + ([f"log_b[{k}]" for k in range(1, K + 1)] + [f"gomp_slope[{k}]" for k in range(1, K + 1)]
                + ["makeham[1]"] if d["use_mortality"] else [])
             + ([f"rho[{k},{c}]" for k in range(1, K + 1) for c in range(1, C + 1)] if d["ar_mode"] else [])
             + ([f"sigma_meas[{c}]" for c in range(1, C + 1)] if d["ar_mode"] == 2 else [])
             + ["lp__"])
    draws = fit.draws_pd(vars=["theta", "coef", "sigma", "lp__"]
                         + (["coef_chronic", "phi_chronic"] if d["use_chronic"] else [])
                         + (["log_b", "gomp_slope", "makeham"] if d["use_mortality"] else [])
                         + (["rho"] if d["ar_mode"] else [])
                         + (["sigma_meas"] if d["ar_mode"] == 2 else []))
    summary, worst, worst_name = {}, 0.0, ""
    per_chain = {}
    for n in names:
        col = draws[n].to_numpy().reshape(args.chains, args.sampling)
        r = split_rhat(col)
        summary[n] = {"mean": float(col.mean()), "sd": float(col.std()), "rhat": r}
        per_chain[n] = [float(v) for v in col.mean(axis=1)]
        if n != "lp__" and r > worst:
            worst, worst_name = r, n
    print(f"  max structural R-hat {worst:.4f} ({worst_name})", flush=True)
    g = lambda n: summary[n]["mean"]  # noqa: E731
    print("  theta " + " ".join(f"{g(f'theta[{k}]'):.3f}" for k in range(1, K + 1)))
    print("  per-chain lp__ " + " ".join(f"{v:.1f}" for v in per_chain["lp__"]))
    hazard = {}
    if d["use_mortality"]:
        print("  Gompertz log level at 55 by class " + " ".join(f"{g(f'log_b[{k}]'):+.2f}" for k in range(1, K + 1)))
        print("  Gompertz slope by class " + " ".join(f"{g(f'gomp_slope[{k}]'):.3f}" for k in range(1, K + 1))
              + f"; Makeham {g('makeham[1]'):.5f}")
        for age in (35, 55, 75, 85):
            hz = [g("makeham[1]") + np.exp(g(f"log_b[{k}]") + g(f"gomp_slope[{k}]") * (age - 55)) for k in range(1, K + 1)]
            hazard[age] = hz
            print(f"  yearly hazard at {age}: " + " ".join(f"{h:.4f}" for h in hz)
                  + f"  (class 1 / class {K}: {hz[0] / hz[-1]:.1f}x)")
    if d["use_chronic"]:
        print("  chronic log-mean by class " + " ".join(f"{g(f'coef_chronic[{k},1]'):+.2f}" for k in range(1, K + 1)))
    if d["ar_mode"]:
        for c in range(1, C + 1):
            print(f"  rho, channel {c} ({payload.channels[c - 1]}): "
                  + " ".join(f"{g(f'rho[{k},{c}]'):.3f}" for k in range(1, K + 1)))
    if d["ar_mode"] == 2:
        print("  sigma_meas " + " ".join(f"{g(f'sigma_meas[{c}]'):.3f}" for c in range(1, C + 1)))
        # Share of each Gaussian channel's observation variance that is
        # persistent state rather than "spike", per class.
        for c in range(1, C + 1):
            sig, mea = g(f"sigma[1,{c}]"), g(f"sigma_meas[{c}]")
            shares = [(sig ** 2 / (1 - g(f"rho[{k},{c}]") ** 2))
                      / (sig ** 2 / (1 - g(f"rho[{k},{c}]") ** 2) + mea ** 2) for k in range(1, K + 1)]
            print(f"  signal share, channel {c} ({payload.channels[c - 1]}): "
                  + " ".join(f"{x:.3f}" for x in shares))

    result = {"variant": args.variant, "channels": list(payload.channels),
              "contract": str(args.contract),
              "n_person": int(d["N_person"]), "n_obs": int(d["N_obs"]), "held_rows": int(n_held),
              "use_chronic": int(d["use_chronic"]), "use_mortality": int(d["use_mortality"]),
              "mort_events": int(np.sum(d["mort_y"])),
              "chains": args.chains, "warmup": args.warmup, "sampling": args.sampling,
              "wall_hours": wall / 3600, "max_structural_rhat": worst, "worst_param": worst_name,
              "channel_moments": payload.channel_moments, "params": summary,
              "per_chain_means": per_chain, "hazard_by_age": {str(a): v for a, v in hazard.items()}}
    if cfg["holdout"]:
        for label, var in (("ar_conditional", "log_lik_heldout"), ("class_only", "log_lik_heldout_marginal")):
            ll = fit.stan_variable(var)
            lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
            result[f"heldout_{label}_mean"] = float(lpd.mean())
            if label == "ar_conditional":
                pd.DataFrame({"pidp": payload.person_ids, "lpd_heldout": lpd}).to_parquet(
                    args.output / "heldout_lpd.parquet", index=False)
            print(f"  held-out {label}: {lpd.mean():.4f} per person", flush=True)
    (args.output / "run_summary.json").write_text(json.dumps(result, indent=1))
    print(f"[{args.variant}] {'CONVERGED' if worst < 1.05 else 'R-HAT FLAG'} in {wall / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
