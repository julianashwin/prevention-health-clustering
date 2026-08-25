"""Parameter recovery for the five-channel mixed-health model.

Truth values are grounded in the predecessor's (quarantined) mixed fit,
converted to this repo's convention — class 1 = worst physical health, PCS/MCS
on the raw standardised scale — and moved OFF the bounds where the old fit was
pinned: its alpha_pcs[3] sat at the 1.5 tanh cap (posterior sd 0.00025) and
phi_chronic at its upper bound of 20. Truth here uses -1.55 and 12.0, values a
capped model could not have recovered.

Channel missingness mirrors the real panel: SRH ~95%, chronic ~90%, ADL ~8%
(a scheduled sub-module). ADL recovery from sparse coverage is the honest
difficulty, not an artefact of the test.
"""

from __future__ import annotations

import argparse
import sys
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

AGE_CENTER, AGE_SCALE = 55, 10.0
RAW = {
    "sf12pcs_dv": (49.1684051, 11.1724567),
    "sf12mcs_dv": (48.8403343, 10.2822509),
}

# Class order: 1 = worst physical health. Derived from the quarantined fit by
# reversing its (best-first) class order and negating its bad-health scale.
TRUTH = {
    "theta": np.array([0.162, 0.405, 0.433]),
    "pcs": {"alpha": [-1.55, -0.063, 0.547],
            "beta": [-0.349, -0.326, -0.126],
            "gamma": [-0.030, -0.020, 0.010],
            "sigma": 0.652},
    "mcs": {"alpha": [-0.709, -0.069, 0.346],
            "beta": [0.178, 0.145, 0.056],
            "gamma": [0.010, 0.015, 0.020],
            "sigma": 0.904},
    # SRH latent mean (higher = worse); cutpoints centred as the model defines.
    "srh": {"eta": [2.199, 0.351, -2.061],
            "beta": [0.367, 0.394, 0.371],
            "cuts": [-3.0, -0.9, 1.3, 2.6]},
    "chronic": {"log_mu": [0.789, 0.456, 0.296],
                "beta": [0.069, 0.074, 0.052],
                "phi": 12.0},
    "adl": {"zero_logit": [0.470, -1.413, -2.267],
            "zero_beta": [0.403, 0.437, 0.335],
            "count_log_mu": [0.60, 0.20, 0.00],
            "phi": 5.0},
}

MISSINGNESS = {"srh": 0.95, "chronic": 0.90, "adl": 0.08}

TOLERANCES = {
    "theta": 0.05,
    "gauss": 0.20,       # alpha/beta/gamma on either gaussian channel
    "srh": 0.30,         # latent-scale quantities
    "chronic": 0.25,
    "adl": 0.60,         # 8% coverage: wide by honest necessity
    "sigma": 0.06,
}


def simulate(n_persons: int, seed: int = 20260825) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = rng.choice(3, size=n_persons, p=TRUTH["theta"])
    cuts = np.asarray(TRUTH["srh"]["cuts"])
    records = []
    for person, k in enumerate(classes):
        n_obs = int(np.clip(rng.poisson(6) + 3, 3, 15))
        start = rng.integers(20, 89 - n_obs)
        ages = np.arange(start, start + n_obs)
        a = (ages - AGE_CENTER) / AGE_SCALE
        rows = {"pidp": person, "age": ages}

        for name in ("pcs", "mcs"):
            t = TRUTH[name]
            mu = t["alpha"][k] + t["beta"][k] * a + t["gamma"][k] * a**2
            z = mu + rng.normal(0, t["sigma"], n_obs)
            mean, sd = RAW[f"sf12{name}_dv"]
            rows[f"sf12{name}_dv"] = z * sd + mean

        eta = TRUTH["srh"]["eta"][k] + TRUTH["srh"]["beta"][k] * a
        u = np.log(rng.random((n_obs, 1)) / (1 - rng.random((n_obs, 1))))
        # ordered logit draw: category = 1 + number of cutpoints below eta+logistic
        latent = eta[:, None] + np.random.default_rng(seed + person).logistic(size=(n_obs, 1))
        srh = 1 + (latent > cuts[None, :]).sum(axis=1).ravel()
        rows["general_health_combined"] = np.where(
            rng.random(n_obs) < MISSINGNESS["srh"], srh, np.nan
        )

        log_mu = TRUTH["chronic"]["log_mu"][k] + TRUTH["chronic"]["beta"][k] * a
        mu_c = np.exp(log_mu)
        phi = TRUTH["chronic"]["phi"]
        chronic = rng.negative_binomial(phi, phi / (phi + mu_c))
        rows["chronic_condition_count"] = np.where(
            rng.random(n_obs) < MISSINGNESS["chronic"], chronic, np.nan
        )

        z_logit = TRUTH["adl"]["zero_logit"][k] + TRUTH["adl"]["zero_beta"][k] * a
        any_lim = rng.random(n_obs) < 1 / (1 + np.exp(-z_logit))
        mu_a = np.exp(TRUTH["adl"]["count_log_mu"][k])
        phi_a = TRUTH["adl"]["phi"]
        counts = np.zeros(n_obs, dtype=int)
        for j in np.where(any_lim)[0]:
            draw = 0
            while draw == 0:
                draw = rng.negative_binomial(phi_a, phi_a / (phi_a + mu_a))
            counts[j] = draw
        rows["adl_limitation_count"] = np.where(
            rng.random(n_obs) < MISSINGNESS["adl"], counts.astype(float), np.nan
        )

        frame = pd.DataFrame(rows)
        records.append(frame)
    out = pd.concat(records, ignore_index=True)
    out["age"] = out["age"].astype(int)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=4000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--sampling", type=int, default=500)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--init-jitter", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("artifacts/recovery-mixed"))
    args = parser.parse_args(argv)

    spec = get_model("mixed-health")
    frame = simulate(args.persons)
    counts = {c: int(frame[c].notna().sum()) for c in
              ("sf12pcs_dv", "general_health_combined",
               "chronic_condition_count", "adl_limitation_count")}
    print(f"mixed-health: {frame['pidp'].nunique():,} persons, {len(frame):,} rows")
    print(f"  channel rows: pcs/mcs {counts['sf12pcs_dv']:,}, "
          f"srh {counts['general_health_combined']:,}, "
          f"chronic {counts['chronic_condition_count']:,}, "
          f"adl {counts['adl_limitation_count']:,}")

    payload = build_mixed_payload(spec, frame)
    inits = mixed_assignment_inits(spec, payload, jitter=args.init_jitter)
    print(f"  partition init theta start "
          f"{[round(v, 3) for v in inits[0]['theta']]}")

    model = compile_model(spec.stan_file)
    data_path = write_mixed_data(payload, args.output)
    fit = model.sample(
        data=str(data_path), inits=inits,
        chains=args.chains, iter_warmup=args.warmup,
        iter_sampling=args.sampling,
        adapt_delta=spec.adapt_delta, max_treedepth=spec.max_treedepth,
        seed=spec.seed, threads_per_chain=args.threads_per_chain,
        output_dir=str(args.output / "chains"), show_progress=False,
    )
    draws = fit.draws_pd()
    k = spec.n_classes

    worst: dict[str, float] = {}

    def record(group: str, got: float, truth: float, label: str) -> None:
        delta = got - truth
        worst[group] = max(worst.get(group, 0.0), abs(delta))
        print(f"  {label:22s} {got:9.4f} {truth:9.4f} {delta:+9.4f}")

    print(f"\n  {'parameter':22s} {'recovered':>9s} {'truth':>9s} {'delta':>9s}")
    for i in range(k):
        record("theta", float(draws[f"theta[{i+1}]"].mean()),
               TRUTH["theta"][i], f"theta{i+1}")
    for ci, name in ((1, "pcs"), (2, "mcs")):
        for col, pname in ((1, "alpha"), (2, "beta"), (3, "gamma")):
            for i in range(k):
                record("gauss",
                       float(draws[f"coef_gauss[{ci},{i+1},{col}]"].mean()),
                       TRUTH[name][pname][i], f"{name}.{pname}{i+1}")
        record("sigma", float(draws[f"sigma_gauss[{ci}]"].mean()),
               TRUTH[name]["sigma"], f"{name}.sigma")
    for i in range(k):
        record("srh", float(draws[f"coef_srh[{i+1},1]"].mean()),
               TRUTH["srh"]["eta"][i], f"srh.eta{i+1}")
        record("srh", float(draws[f"coef_srh[{i+1},2]"].mean()),
               TRUTH["srh"]["beta"][i], f"srh.beta{i+1}")
    for i in range(k):
        record("chronic", float(draws[f"coef_chronic[{i+1},1]"].mean()),
               TRUTH["chronic"]["log_mu"][i], f"chronic.mu{i+1}")
    for i in range(k):
        record("adl", float(draws[f"coef_adl_zero[{i+1},1]"].mean()),
               TRUTH["adl"]["zero_logit"][i], f"adl.zero{i+1}")
        record("adl", float(draws[f"coef_adl_count[{i+1},1]"].mean()),
               TRUTH["adl"]["count_log_mu"][i], f"adl.count{i+1}")

    summary = fit.summary()
    rhat = float(summary["R_hat"].dropna().max())
    ess = float(summary["ESS_bulk"].dropna().min())
    print(f"\ndiagnostics: max R-hat {rhat:.5f}, min ESS bulk {ess:.0f}")

    print("\n=== verdict ===")
    ok = True
    for key, tol in TOLERANCES.items():
        good = worst.get(key, 0.0) <= tol
        ok &= good
        print(f"  {key:8s} max |delta| {worst.get(key, 0.0):.4f}  "
              f"tol {tol:.2f}  {'PASS' if good else 'FAIL'}")
    conv = rhat < 1.05
    ok &= conv
    print(f"  {'R-hat':8s} {rhat:.5f} < 1.05  {'PASS' if conv else 'FAIL'}")
    print(f"\n{'RECOVERY OK' if ok else 'RECOVERY FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
