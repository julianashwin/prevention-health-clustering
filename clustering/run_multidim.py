"""Fit the four-channel multidimensional mixture on the frozen contract.

    python clustering/run_multidim.py --variant ar1-holdout \
        --output artifacts/multidim/ar1-holdout

Variants: baseline | holdout | ar1 | ar1-holdout. Structural diagnostics are
computed without touching person-level columns; per-person held-out densities
are written when a holdout is requested.
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
    GAUSS_CHANNELS,
    build_multidim_payload,
    multidim_inits,
    write_multidim_data,
)

STAN = (Path(__file__).resolve().parents[1] / "src"
        / "prevention_health_clustering" / "models" / "stan"
        / "mixture_multidim_panel.stan")
CONTRACT = (PROCESSED_DATA_DIR / "contracts"
            / "multidim_lifecycle_20_89_minobs3_v1")
VARIANTS = {
    "baseline":    dict(ar_mode=0, holdout=False),
    "holdout":     dict(ar_mode=0, holdout=True),
    "ar1":         dict(ar_mode=1, holdout=False),
    "ar1-holdout": dict(ar_mode=1, holdout=True),
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
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--holdout-last-k", type=int, default=2)
    parser.add_argument("--min-obs", type=int, default=5,
                        help="applied to EVERY variant so the four differ by "
                             "specification alone")
    parser.add_argument("--no-mortality", action="store_true", default=True)
    parser.add_argument("--with-mortality", dest="no_mortality",
                        action="store_false")
    parser.add_argument("--init-jitter", type=float, default=0.15)
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = VARIANTS[args.variant]
    long = pd.read_csv(args.contract / "long.csv")
    payload = build_multidim_payload(
        long, n_classes=args.classes, ar_mode=cfg["ar_mode"],
        holdout_last_k=args.holdout_last_k if cfg["holdout"] else None,
        min_person_obs=args.min_obs,
        use_mortality=not args.no_mortality,
        emit_person_quantities=cfg["holdout"])
    d = payload.data
    n_held = sum(d["hold_end"][i] - d["hold_start"][i] + 1
                 for i in range(d["N_person"]) if d["hold_end"][i] > 0)
    print(f"[{args.variant}] {d['N_person']:,} persons, {d['N_obs']:,} rows, "
          f"ar_mode={d['ar_mode']}, held-out {n_held:,}, "
          f"chronic obs {np.mean(d['chronic_obs']):.1%}, "
          f"mortality {'on' if d['use_mortality'] else 'OFF'}", flush=True)

    inits = multidim_inits(payload, jitter=args.init_jitter,
                           n_chains=args.chains)
    print(f"  partition init theta {[round(v,3) for v in inits[0]['theta']]}",
          flush=True)
    model = compile_model(STAN)
    data_path = write_multidim_data(payload, args.output)
    fit = model.sample(
        data=str(data_path), inits=inits, chains=args.chains,
        iter_warmup=args.warmup, iter_sampling=args.sampling,
        adapt_delta=0.9, max_treedepth=12, seed=20260826,
        threads_per_chain=args.threads_per_chain,
        output_dir=str(args.output / "chains"), show_progress=False)
    wall = time.time() - t0
    print(f"  sampling done in {wall/3600:.2f} h", flush=True)

    K, C, P = d["K"], d["C"], d["P"]
    names = ([f"theta[{k}]" for k in range(1, K + 1)]
             + [f"coef[{c},{k},{p}]" for c in range(1, C + 1)
                for k in range(1, K + 1) for p in range(1, P + 1)]
             + [f"sigma[1,{c}]" for c in range(1, C + 1)]
             + [f"coef_chronic[{k},{p}]" for k in range(1, K + 1)
                for p in range(1, P + 1)]
             + ["phi_chronic"]
             + ([f"coef_mort[{k},{p}]" for k in range(1, K + 1)
                 for p in range(1, P + 1)] if d["use_mortality"] else [])
             + ([f"rho[{k}]" for k in range(1, K + 1)] if d["ar_mode"] else [])
             + ["lp__"])
    draws = fit.draws_pd(vars=["theta", "coef", "sigma", "coef_chronic",
                               "phi_chronic", "lp__"]
                         + (["coef_mort"] if d["use_mortality"] else [])
                         + (["rho"] if d["ar_mode"] else []))
    summary, worst, worst_name = {}, 0.0, ""
    for n in names:
        col = draws[n].to_numpy().reshape(args.chains, args.sampling)
        r = split_rhat(col)
        summary[n] = {"mean": float(col.mean()), "sd": float(col.std()),
                      "rhat": r}
        if n != "lp__" and r > worst:
            worst, worst_name = r, n
    print(f"  max structural R-hat {worst:.4f} ({worst_name})", flush=True)
    print("  theta " + " ".join(f"{summary[f'theta[{k}]']['mean']:.3f}"
                                for k in range(1, K + 1)))
    if d["use_mortality"]:
        print("  mortality logit intercept by class " + " ".join(
            f"{summary[f'coef_mort[{k},1]']['mean']:+.2f}"
            for k in range(1, K + 1)))
    print("  chronic log-mean by class " + " ".join(
        f"{summary[f'coef_chronic[{k},1]']['mean']:+.2f}" for k in range(1, K + 1)))
    if d["ar_mode"]:
        print("  rho " + " ".join(f"{summary[f'rho[{k}]']['mean']:.3f}"
                                  for k in range(1, K + 1)))

    result = {"variant": args.variant, "n_person": int(d["N_person"]),
              "n_obs": int(d["N_obs"]), "held_rows": int(n_held),
              "use_mortality": int(d["use_mortality"]),
              "mort_events": int(np.sum(d["mort_y"])),
              "wall_hours": wall / 3600, "max_structural_rhat": worst,
              "worst_param": worst_name,
              "channel_moments": payload.channel_moments, "params": summary}
    if cfg["holdout"]:
        for label, var in (("ar_conditional", "log_lik_heldout"),
                           ("class_only", "log_lik_heldout_marginal")):
            ll = fit.stan_variable(var)
            lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
            result[f"heldout_{label}_mean"] = float(lpd.mean())
            if label == "ar_conditional":
                pd.DataFrame({"pidp": payload.person_ids,
                              "lpd_heldout": lpd}).to_parquet(
                    args.output / "heldout_lpd.parquet", index=False)
            print(f"  held-out {label}: {lpd.mean():.4f} per person",
                  flush=True)
    (args.output / "run_summary.json").write_text(json.dumps(result, indent=1))
    print(f"[{args.variant}] {'CONVERGED' if worst < 1.05 else 'R-HAT FLAG'} "
          f"in {wall/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
