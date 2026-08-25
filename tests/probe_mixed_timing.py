"""Timing probe for the empirical five-channel mixed fit.

The predecessor's mixed_outcomes model was 'intractable in practice' —
~15 iterations/hour at the full roster on 8 VM cores. This probe measures the
rebuilt model on real data at two subsample sizes, checks the scaling is
roughly linear in rows, and extrapolates to the full roster at production
settings. A probe, not a production fit: short chains, convergence not gated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import compile_model
from prevention_health_clustering.runner.mixed import (
    build_mixed_payload,
    mixed_assignment_inits,
    write_mixed_data,
)

FULL_ROSTER_ROWS = 484_300
PRODUCTION_ITERATIONS = 2_000  # 1000 warmup + 1000 sampling


def chain_seconds(fit) -> list[tuple[float, float]]:
    """Per-chain (warmup, sampling) seconds from the CmdStan CSV comments.

    Format:  #  Elapsed Time: 2256.26 seconds (Warm-up)
             #                290.394 seconds (Sampling)
    """
    out = []
    for path in fit.runset.csv_files:
        warm = samp = 0.0
        for line in open(path):
            if "(Warm-up)" in line:
                warm = float(line.replace("#", "").split()[2])
            elif "(Sampling)" in line and "seconds" in line:
                samp = float(line.replace("#", "").split()[0])
        out.append((warm, samp))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 6000])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--sampling", type=int, default=200)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=None,
                        help="initial step size; skips the expensive early search")
    parser.add_argument("--frame", type=Path,
                        default=Path("/tmp/phc_mixed_frame_full.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mixed-probe"))
    args = parser.parse_args(argv)

    spec = get_model("mixed-health")
    full = pd.read_parquet(args.frame)
    model = compile_model(spec.stan_file)
    iters = args.warmup + args.sampling

    results = []
    for size in args.sizes:
        keep = full["pidp"].drop_duplicates().sample(
            size, random_state=20260825
        )
        frame = full[full["pidp"].isin(keep)]
        payload = build_mixed_payload(spec, frame)
        inits = mixed_assignment_inits(spec, payload, jitter=0.15)
        rows = sum(payload.data[k] for k in ("N_srh", "N_chronic", "N_adl")) \
            + sum(payload.data["N_gauss"])
        out = args.output / f"n{size}"
        data_path = write_mixed_data(payload, out)

        extra = {}
        if args.step_size is not None:
            extra["step_size"] = args.step_size
        start = time.time()
        fit = model.sample(
            data=str(data_path), inits=inits,
            chains=args.chains, iter_warmup=args.warmup,
            iter_sampling=args.sampling,
            adapt_delta=spec.adapt_delta, max_treedepth=spec.max_treedepth,
            seed=spec.seed, threads_per_chain=args.threads_per_chain,
            output_dir=str(out / "chains"), show_progress=False, **extra,
        )
        wall = time.time() - start
        per_chain = chain_seconds(fit)
        slowest = max(w + s for w, s in per_chain)
        warm_slow = max(w for w, _ in per_chain)
        samp_slow = max(s for _, s in per_chain)
        sec_per_iter = slowest / iters
        print(f"    warmup slowest {warm_slow:7.1f}s ({warm_slow/args.warmup:5.2f} s/it)  "
              f"sampling slowest {samp_slow:7.1f}s ({samp_slow/args.sampling:5.2f} s/it)")
        rhat = float(fit.summary()["R_hat"].dropna().max())
        results.append(
            {"persons": size, "channel_rows": rows, "wall_s": wall,
             "slowest_chain_s": slowest, "sec_per_iter": sec_per_iter,
             "rhat": rhat}
        )
        print(f"n={size:>6,}  rows={rows:>9,}  wall {wall:7.1f}s  "
              f"slowest chain {slowest:7.1f}s  {sec_per_iter:6.3f} s/iter  "
              f"R-hat {rhat:.3f}")

    print("\n=== scaling and extrapolation ===")
    if len(results) >= 2:
        a, b = results[0], results[-1]
        row_ratio = b["channel_rows"] / a["channel_rows"]
        time_ratio = b["sec_per_iter"] / a["sec_per_iter"]
        print(f"rows x{row_ratio:.2f} -> time x{time_ratio:.2f} "
              f"({'~linear' if 0.6 * row_ratio <= time_ratio <= 1.4 * row_ratio else 'NONLINEAR'})")
    ref = results[-1]
    scale = FULL_ROSTER_ROWS / ref["channel_rows"]
    full_sec_per_iter = ref["sec_per_iter"] * scale
    hours = full_sec_per_iter * PRODUCTION_ITERATIONS / 3600
    print(f"full roster ({FULL_ROSTER_ROWS:,} channel rows): "
          f"~{full_sec_per_iter:.2f} s/iter "
          f"-> ~{hours:.1f} h for {PRODUCTION_ITERATIONS} iterations")
    print(f"predecessor at full roster: ~15 iter/hour = 240 s/iter "
          f"-> speedup ~{240 / full_sec_per_iter:.0f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
