"""Exercise every data toggle in the consolidated Stan program.

The design claim is that channels, AR persistence, cohort effects, class count
and the held-out window are all selected by DATA rather than by separate model
files. That claim is only worth anything if every cell actually runs.

This is a smoke matrix, not an inference test: small samples and short chains.
It checks that each configuration builds a valid payload, samples, produces
finite parameters in the right shapes, and respects the ordering constraint.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import ModelSpec, get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model


def cells() -> list[tuple[str, dict, dict]]:
    """(label, ModelSpec overrides, build_payload kwargs)."""
    return [
        ("C1 baseline",          {}, {}),
        ("C2 bivariate",         {"channels": ("sf12pcs_dv", "sf12mcs_dv")}, {}),
        ("C1 + AR(1)",           {"ar_mode": 1}, {}),
        ("C1 + cohort decade",   {"cohort": "decade"}, {}),
        ("C1 + class sigma",     {"homosigma": False}, {}),
        ("C1 K=2",               {"n_classes": 2, "alpha_prior_scale": 1.5}, {}),
        ("C1 K=4",               {"n_classes": 4}, {}),
        ("C1 held-out 70+",      {}, {"holdout_min_age": 70, "emit_person_quantities": True}),
        ("C2 + AR + cohort",     {"channels": ("sf12pcs_dv", "sf12mcs_dv"),
                                  "ar_mode": 1, "cohort": "decade"}, {}),
        ("C2 MCS-anchored",      {"channels": ("sf12pcs_dv", "sf12mcs_dv"),
                                  "anchor_channel": "sf12mcs_dv"}, {}),
    ]


def check(fit, spec: ModelSpec, payload, holdout: bool) -> list[str]:
    """Structural checks. Returns a list of problems, empty if fine."""
    problems = []
    draws = fit.draws_pd()
    k, c, p = spec.n_classes, spec.n_channels, spec.design_width

    theta = np.array([draws[f"theta[{i}]"].mean() for i in range(1, k + 1)])
    if not np.isfinite(theta).all():
        problems.append("theta not finite")
    if abs(theta.sum() - 1.0) > 1e-6:
        problems.append(f"theta sums to {theta.sum():.6f}")

    # Ordering must hold on the anchor channel's intercept.
    a = spec.anchor_index
    anchor = np.array([draws[f"coef[{a},{i},1]"].mean() for i in range(1, k + 1)])
    if not np.all(np.diff(anchor) > 0):
        problems.append(f"anchor intercepts not ordered: {np.round(anchor, 3)}")

    for ch in range(1, c + 1):
        for i in range(1, k + 1):
            for col in range(1, p + 1):
                v = draws[f"coef[{ch},{i},{col}]"].mean()
                if not np.isfinite(v):
                    problems.append(f"coef[{ch},{i},{col}] not finite")

    n_sigma_rows = 1 if spec.homosigma else k
    for r in range(1, n_sigma_rows + 1):
        for ch in range(1, c + 1):
            v = draws[f"sigma[{r if not spec.homosigma else 1},{ch}]"].mean()
            if not (np.isfinite(v) and v > 0):
                problems.append(f"sigma[{r},{ch}] invalid: {v}")

    if spec.ar_mode:
        rho = np.array([draws[f"rho[{i}]"].mean() for i in range(1, k + 1)])
        if not np.isfinite(rho).all() or (rho < 0).any() or (rho > 0.99).any():
            problems.append(f"rho out of range: {np.round(rho, 3)}")

    if spec.cohort != "none":
        n_cohort = payload.data["N_cohort"]
        if n_cohort < 2:
            problems.append(f"cohort requested but N_cohort={n_cohort}")

    if holdout:
        if "log_lik_heldout[1]" not in draws.columns:
            problems.append("held-out log density not emitted")
        else:
            held = np.array(
                [draws[f"log_lik_heldout[{i}]"].mean()
                 for i in range(1, min(200, payload.data["N_person"]) + 1)]
            )
            if not np.isfinite(held).all():
                problems.append("held-out log density not finite")
            if (held == 0).all():
                problems.append("held-out log density identically zero")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--persons", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--sampling", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("artifacts/toggle-matrix"))
    args = parser.parse_args(argv)

    long = pd.read_csv(args.contract / "long.csv")
    keep = long["pidp"].drop_duplicates().head(args.persons)
    long = long[long["pidp"].isin(keep)]
    base = get_model("pcs-headline")

    print(f"{len(cells())} cells, {long['pidp'].nunique():,} persons, "
          f"{args.chains} chains x {args.warmup}+{args.sampling}\n")
    print(f"{'cell':22s} {'C':>2s} {'K':>2s} {'ar':>3s} {'coh':>4s} {'result':>9s}  notes")
    print("-" * 88)

    failures = 0
    for label, overrides, payload_kwargs in cells():
        spec = dataclasses.replace(base, **overrides)
        holdout = bool(payload_kwargs.get("holdout_min_age"))
        try:
            payload = build_payload(spec, long, **payload_kwargs)
            fit = fit_model(
                spec, payload,
                output_dir=args.output / label.replace(" ", "_").replace("+", ""),
                chains=args.chains, iter_warmup=args.warmup,
                iter_sampling=args.sampling, show_console=False,
            )
            problems = check(fit, spec, payload, holdout)
            status = "PASS" if not problems else "FAIL"
            failures += bool(problems)
            note = "; ".join(problems[:2]) if problems else ""
        except Exception as exc:  # noqa: BLE001 - smoke test reports, never masks
            status, failures, note = "ERROR", failures + 1, str(exc).split("\n")[0][:60]
            if "--traceback" in (argv or sys.argv):
                traceback.print_exc()
        print(
            f"{label:22s} {spec.n_channels:2d} {spec.n_classes:2d} "
            f"{spec.ar_mode:3d} {spec.cohort[:4]:>4s} {status:>9s}  {note}"
        )

    print("-" * 88)
    print(f"{len(cells()) - failures}/{len(cells())} cells passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
