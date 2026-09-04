"""Build the Stan payload, run CmdStan, and write results.

Python talks to CmdStan directly through cmdstanpy. The predecessor project
routed every fit through an R layer, which added a version-drift surface
(cmdstanr, posterior, R itself) for no modelling benefit.

Compiled binaries are cached by the hash of the Stan source, so a repeated fit
does not pay the ~15 s compile the predecessor paid on every run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from prevention_health_clustering.config import (
    ARTIFACTS_DIR,
    DEFAULT_AGE_CENTER,
    DEFAULT_AGE_SCALE,
)
from prevention_health_clustering.models.registry import ModelSpec

COMPILE_CACHE = ARTIFACTS_DIR / "stan-cache"


@dataclass
class StanPayload:
    data: dict
    person_ids: np.ndarray
    channel_moments: dict[str, dict[str, float]]


def _cohort_ids(birthy: pd.Series, mode: str) -> tuple[np.ndarray, int]:
    """Map birth year to a one-based cohort index. Index 1 is the reference."""
    if mode == "none":
        return np.ones(len(birthy), dtype=int), 1
    if mode != "decade":
        raise ValueError(f"Unknown cohort mode: {mode!r}")
    decade = (birthy.astype("float").fillna(-1) // 10 * 10).astype(int)
    levels = sorted(value for value in decade.unique() if value >= 0)
    lookup = {value: index + 1 for index, value in enumerate(levels)}
    return decade.map(lookup).fillna(1).astype(int).to_numpy(), len(levels)


def build_payload(
    spec: ModelSpec,
    long: pd.DataFrame,
    *,
    age_center: int = DEFAULT_AGE_CENTER,
    age_scale: float = DEFAULT_AGE_SCALE,
    holdout_min_age: int | None = None,
    holdout_last_k: int | None = None,
    holdout_min_person_obs: int | None = None,
    emit_person_quantities: bool = False,
    grainsize: int = 0,
) -> StanPayload:
    """Assemble the Stan data block from a contract's long frame.

    ``holdout_min_age`` moves each person's rows at or above that age into the
    held-out window. Those rows do not enter the likelihood; they are scored in
    generated quantities as a conditional predictive density.

    ``holdout_last_k`` instead holds out each person's LAST k observations (by
    age). ``holdout_min_person_obs`` restricts the sample to people with at
    least that many observations first, so every retained person keeps
    ``min_obs - k`` fitted rows. The two holdout modes are mutually exclusive.
    Because held-out rows are a within-person suffix, the AR(1) age-gap
    recursion bridges the fit/hold boundary correctly in either mode.
    """
    spec.validate()
    if holdout_min_age is not None and holdout_last_k is not None:
        raise ValueError("holdout_min_age and holdout_last_k are exclusive.")
    if holdout_last_k is not None:
        min_obs = holdout_min_person_obs or (holdout_last_k + 1)
        if min_obs <= holdout_last_k:
            raise ValueError("holdout_min_person_obs must exceed holdout_last_k.")
    missing = [c for c in spec.channels if c not in long.columns]
    if missing:
        raise ValueError(f"Contract is missing channels: {', '.join(missing)}")

    frame = long.sort_values(["pidp", "age"]).reset_index(drop=True)

    # A held-out age threshold can leave a person with no fitted rows at all
    # (everyone whose observations begin at or after the threshold). Such a
    # person carries no history to condition on, so they are not part of this
    # design's population. Drop them here and record how many, rather than
    # failing on a legitimate data condition.
    dropped_no_history = 0
    if holdout_last_k is not None:
        n_obs_per = frame.groupby("pidp")["age"].size()
        keep_ids = n_obs_per[n_obs_per >= min_obs].index
        dropped_no_history = int((n_obs_per < min_obs).sum())
        frame = frame[frame["pidp"].isin(keep_ids)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"holdout_min_person_obs={min_obs} leaves nobody in the sample."
            )
    if holdout_min_age is not None:
        has_history = frame.groupby("pidp")["age"].min() < holdout_min_age
        keep_ids = has_history[has_history].index
        dropped_no_history = int((~has_history).sum())
        frame = frame[frame["pidp"].isin(keep_ids)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"holdout_min_age={holdout_min_age} leaves no person with "
                "fitted rows."
            )

    # Person row windows. Rows are contiguous per person after the sort.
    codes, person_ids = pd.factorize(frame["pidp"], sort=True)
    frame = frame.assign(_person=codes)
    bounds = frame.groupby("_person").apply(
        lambda g: pd.Series({"lo": g.index.min() + 1, "hi": g.index.max() + 1}),
        include_groups=False,
    )

    if holdout_min_age is None and holdout_last_k is None:
        fit_start = bounds["lo"].to_numpy()
        fit_end = bounds["hi"].to_numpy()
        hold_start = np.zeros(len(bounds), dtype=int)
        hold_end = np.zeros(len(bounds), dtype=int)
    else:
        if holdout_last_k is not None:
            # the last k rows of each person's age-sorted block are held out
            held = frame.groupby("_person").cumcount(ascending=False) < holdout_last_k
        else:
            held = frame["age"] >= holdout_min_age
        fit_start, fit_end, hold_start, hold_end = [], [], [], []
        for _, group in frame.groupby("_person", sort=True):
            fitted = group.index[~held.loc[group.index]]
            holdout = group.index[held.loc[group.index]]
            if len(fitted) == 0:  # pragma: no cover - guarded above
                raise AssertionError(
                    "person retained without fitted rows after history filter"
                )
            fit_start.append(int(fitted.min()) + 1)
            fit_end.append(int(fitted.max()) + 1)
            hold_start.append(int(holdout.min()) + 1 if len(holdout) else 0)
            hold_end.append(int(holdout.max()) + 1 if len(holdout) else 0)
        fit_start = np.array(fit_start)
        fit_end = np.array(fit_end)
        hold_start = np.array(hold_start)
        hold_end = np.array(hold_end)

    # Design matrix in scaled centred age.
    age_scaled = (frame["age"].to_numpy() - age_center) / age_scale
    columns = [np.ones_like(age_scaled), age_scaled]
    if spec.design == "quadratic":
        columns.append(age_scaled**2)
    design = np.column_stack(columns)

    # Channels, optionally standardised. Moments are recorded so that the
    # posterior can be returned to the response scale exactly.
    y_matrix, moments = [], {}
    for channel in spec.channels:
        values = frame[channel].to_numpy(dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        moments[channel] = {"mean": mean, "sd": sd}
        y_matrix.append((values - mean) / sd if spec.standardize else values)

    cohort_id, n_cohort = _cohort_ids(frame["birthy"], spec.cohort)

    age_gap = np.zeros(len(frame)) if spec.ar_mode == 0 else np.concatenate(
        [[0.0], np.maximum(np.diff(frame["age"].to_numpy()), 1.0)]
    )

    n_person = len(bounds)
    data = {
        "N_obs": int(len(frame)),
        "N_person": int(n_person),
        "K": int(spec.n_classes),
        "C": int(spec.n_channels),
        "P": int(spec.design_width),
        "y": [list(map(float, row)) for row in y_matrix],
        "X": design.tolist(),
        "fit_start": fit_start.astype(int).tolist(),
        "fit_end": fit_end.astype(int).tolist(),
        "hold_start": hold_start.astype(int).tolist(),
        "hold_end": hold_end.astype(int).tolist(),
        "ar_mode": int(spec.ar_mode),
        "age_gap": age_gap.tolist() if spec.ar_mode else [],
        "N_cohort": int(n_cohort),
        "cohort_id": cohort_id.astype(int).tolist(),
        "cohort_by_class": int(spec.cohort_by_class),
        "homosigma": int(spec.homosigma),
        "anchor_channel": int(spec.anchor_index),
        "person_weight_power": float(spec.person_weight_power),
        "channel_n_obs": [int(np.isfinite(row).sum()) for row in y_matrix],
        "alpha_prior_scale": float(spec.alpha_prior_scale),
        "coef_prior_scale": [float(s) for s in spec.slope_prior_scales],
        "sigma_prior_location": float(spec.sigma_prior_location),
        "sigma_prior_scale": float(spec.sigma_prior_scale),
        "theta_prior_concentration": float(spec.theta_prior_concentration),
        "cohort_prior_scale": float(spec.cohort_prior_scale),
        "rho_prior_alpha": float(spec.rho_prior_alpha),
        "rho_prior_beta": float(spec.rho_prior_beta),
        "sigma_meas_prior_location": float(spec.sigma_meas_prior_location),
        "sigma_meas_prior_scale": float(spec.sigma_meas_prior_scale),
        "emit_person_quantities": int(emit_person_quantities),
        # not read by Stan; carried for provenance
        "_dropped_no_history": dropped_no_history,
        "grainsize": int(grainsize or max(1, n_person // (spec.chains * 4))),
    }
    return StanPayload(
        data=data, person_ids=np.asarray(person_ids), channel_moments=moments
    )



def build_inits(
    spec: ModelSpec,
    payload: StanPayload,
    *,
    seed: int | None = None,
) -> list[dict]:
    """Data-informed, per-chain dispersed initial values.

    A K-class growth mixture started from random unconstrained draws lands
    different chains in different modes; the published pipeline used
    data-informed inits for exactly this reason.

    Intercepts start at evenly spaced quantiles of the person-mean
    distribution, which respects the ordering constraint by construction.
    Slopes start near zero and sigma at the observed within-person scale.

    Each chain gets its own jitter drawn from its own seed. Starting every
    chain at one point mechanically deflates R-hat and the label-switch share
    for a multimodal posterior, so the dispersion here is deliberate.
    """
    rng = np.random.default_rng(seed if seed is not None else spec.seed)
    k, c, p = spec.n_classes, spec.n_channels, spec.design_width

    y = np.asarray(payload.data["y"], dtype=float)
    starts = np.asarray(payload.data["fit_start"]) - 1
    ends = np.asarray(payload.data["fit_end"]) - 1

    # Every channel gets a data-derived intercept ladder, not just the anchor.
    #
    # Initialising the non-anchor channels at random was a real defect: the
    # bivariate full-roster fit produced three distinct modes across four
    # chains (max R-hat 2.42) because the second channel started uninformed
    # while the anchor did not. Only one chain reached the good mode.
    quantiles = np.linspace(0.15, 0.85, k)
    anchor = payload.data["anchor_channel"] - 1

    def channel_ladder(ch: int) -> np.ndarray:
        means = np.array([y[ch][lo : hi + 1].mean() for lo, hi in zip(starts, ends)])
        return np.quantile(means, quantiles)

    # Rank every channel by the ANCHOR channel's person ordering, so the class
    # ladders are mutually consistent: class 1 is the low-anchor group in each
    # channel rather than each channel's own low group.
    anchor_means = np.array(
        [y[anchor][lo : hi + 1].mean() for lo, hi in zip(starts, ends)]
    )
    edges = np.quantile(anchor_means, np.linspace(0, 1, k + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    groups = np.digitize(anchor_means, edges[1:-1])

    ladders = {}
    for ch in range(c):
        means = np.array([y[ch][lo : hi + 1].mean() for lo, hi in zip(starts, ends)])
        ladders[ch] = np.array(
            [means[groups == g].mean() if (groups == g).any() else means.mean()
             for g in range(k)]
        )
    ladder = ladders[anchor]

    within = float(
        np.mean([y[anchor][lo : hi + 1].std() for lo, hi in zip(starts, ends)])
    )
    within = max(within, 0.1)

    inits = []
    for chain in range(spec.chains):
        jitter = rng.normal(0.0, 0.25, size=k)
        anchor_init = np.sort(ladder + jitter)
        init = {
            "theta": np.full(k, 1.0 / k),
            "anchor_intercept": anchor_init,
            "other_intercept": np.column_stack(
                [ladders[ch] + rng.normal(0.0, 0.25, size=k)
                 for ch in range(c) if ch != anchor]
            ) if c > 1 else np.zeros((k, 0)),
            "slope": [rng.normal(0.0, 0.05, size=(k, p - 1)) for _ in range(c)],
            "sigma_raw": np.full((1 if spec.homosigma else k, c), within),
        }
        if spec.ar_mode != 0:
            init["rho"] = np.full(k, 0.3)
        if spec.ar_mode == 2:
            # Open at a modest signal-to-noise split rather than at either
            # boundary: rho and sigma_meas trade off along a ridge, and
            # starting on the ridge rather than at an end of it keeps the
            # chains from setting off in opposite directions.
            init["sigma_meas"] = np.full(c, 0.35)
        # Omitted entirely when there are no cohort effects: a zero-length
        # container's declared shape differs between model variants, and Stan
        # rejects a shape mismatch even when the parameter is empty.
        n_cohort = payload.data["N_cohort"]
        if n_cohort > 1:
            rows = k if spec.cohort_by_class else 1
            init["cohort_free"] = [
                np.zeros((rows, n_cohort - 1)) for _ in range(c)
            ]
        inits.append({key: np.asarray(value).tolist() if isinstance(value, np.ndarray)
                      else value for key, value in init.items()})
    return inits




def assignment_inits(
    spec: ModelSpec,
    payload: StanPayload,
    *,
    jitter: float = 0.0,
    seed: int | None = None,
) -> list[dict]:
    """Initialise from a k-means partition, replicating the predecessor recipe.

    The published joint PCS+MCS fit was initialised from a joint LCGM's modal
    labels: per labelled class, a quadratic OLS on each channel supplied alpha,
    beta, gamma and sigma, and the label shares supplied theta. All chains
    started at that identical point. That is what made the joint fit converge -
    the partition of people into classes was decided before sampling, and the
    init encoded it fully (slopes and shares, not just intercept levels).

    Here k-means on per-person channel means plays the LCGM role. With
    jitter=0 this reproduces the predecessor behaviour (all chains at one
    point), which HIDES multimodality from R-hat: chains that never disperse
    cannot find a second mode. Use jitter > 0 to keep the diagnostic honest,
    or confirm separately with a multi-seed check.
    """
    rng = np.random.default_rng(seed if seed is not None else spec.seed)
    k, c, p = spec.n_classes, spec.n_channels, spec.design_width
    y = np.asarray(payload.data["y"], dtype=float)
    X = np.asarray(payload.data["X"], dtype=float)
    starts = np.asarray(payload.data["fit_start"]) - 1
    ends = np.asarray(payload.data["fit_end"]) - 1

    # Per-person mean per channel; k-means over the joint feature space.
    features = np.column_stack([
        [y[ch][lo : hi + 1].mean() for lo, hi in zip(starts, ends)]
        for ch in range(c)
    ])
    # Plain Lloyd's with k-means++ style farthest-point seeding; avoids a
    # sklearn dependency for a one-shot partition.
    centers = features[rng.choice(len(features), 1)]
    while len(centers) < k:
        d2 = ((features[:, None, :] - centers[None]) ** 2).sum(-1).min(1)
        centers = np.vstack([centers, features[rng.choice(len(features), p=d2 / d2.sum())]])
    for _ in range(50):
        labels = ((features[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.vstack([
            features[labels == g].mean(0) if (labels == g).any() else centers[g]
            for g in range(k)
        ])
        if np.allclose(new, centers):
            break
        centers = new

    # Per class, per channel: quadratic OLS -> alpha, beta, gamma; residual sd.
    theta = np.array([(labels == g).mean() for g in range(k)])
    coef = np.zeros((c, k, p))
    sig = np.zeros((c, k))
    for g in range(k):
        rows = np.concatenate([
            np.arange(lo, hi + 1)
            for lo, hi, lab in zip(starts, ends, labels) if lab == g
        ])
        Xg = X[rows]
        for ch in range(c):
            yg = y[ch][rows]
            b, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
            coef[ch, g, :] = b
            resid = yg - Xg @ b
            sig[ch, g] = max(float(resid.std()), 0.2)

    # Order classes by the anchor channel's intercept, matching the model.
    anchor = payload.data["anchor_channel"] - 1
    order = np.argsort(coef[anchor, :, 0])
    theta, coef, sig = theta[order], coef[:, order, :], sig[:, order]

    inits = []
    for chain in range(spec.chains):
        jit = rng.normal(0.0, jitter, size=(c, k, p)) if jitter > 0 else 0.0
        cj = coef + jit
        init = {
            "theta": np.maximum(theta, 1e-3) / np.maximum(theta, 1e-3).sum(),
            "anchor_intercept": np.sort(cj[anchor, :, 0]),
            "slope": [cj[ch, :, 1:] for ch in range(c)],
            "sigma_raw": (sig.mean(1, keepdims=True).T if spec.homosigma
                          else sig.T),
        }
        if c > 1:
            init["other_intercept"] = np.column_stack(
                [cj[ch, :, 0] for ch in range(c) if ch != anchor]
            )
        if spec.ar_mode != 0:
            init["rho"] = np.full(k, 0.3)
        if spec.ar_mode == 2:
            # Open at a modest signal-to-noise split rather than at either
            # boundary: rho and sigma_meas trade off along a ridge, and
            # starting on the ridge rather than at an end of it keeps the
            # chains from setting off in opposite directions.
            init["sigma_meas"] = np.full(c, 0.35)
        n_cohort = payload.data["N_cohort"]
        if n_cohort > 1:
            rows_ = k if spec.cohort_by_class else 1
            init["cohort_free"] = [np.zeros((rows_, n_cohort - 1)) for _ in range(c)]
        inits.append({key: np.asarray(val).tolist() if isinstance(val, np.ndarray)
                      else val for key, val in init.items()})
    return inits


def pathfinder_inits(
    spec: ModelSpec,
    payload: StanPayload,
    *,
    output_dir: Path,
    num_paths: int = 8,
    draws: int = 200,
) -> tuple[list[dict], float]:
    """Initialise from Pathfinder rather than a static data-informed ladder.

    Pathfinder runs multi-path quasi-Newton optimisation with importance
    resampling and is designed specifically to initialise MCMC. It is preferred
    over a MAP point for two reasons: the MAP of a finite mixture is degenerate
    (the density diverges as a component scale collapses onto one point), and a
    single point start deflates R-hat for a multimodal posterior. Pathfinder
    returns a set of draws, so each chain gets a distinct, data-informed start.

    Falls back to the static ladder if Pathfinder fails, which it can do on a
    badly conditioned mixture.

    Returns (inits, seconds_spent).
    """
    import time

    from cmdstanpy import CmdStanModel

    start = time.time()
    model = compile_model(spec.stan_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    stan_data = {k: v for k, v in payload.data.items() if not k.startswith("_")}
    data_path = output_dir / "stan_data.json"
    data_path.write_text(json.dumps(stan_data))
    try:
        pf = model.pathfinder(
            data=str(data_path),
            num_paths=num_paths,
            draws=draws,
            seed=spec.seed,
            # one init dict, not one per chain: num_paths need not equal chains
            inits=build_inits(spec, payload)[0],
            show_console=False,
        )
        # CmdStanPathfinder resolves attributes to Stan variables, so
        # draws_pd() is not available; pull variables explicitly.
        variables = {
            name: np.atleast_1d(pf.stan_variable(name))
            for name in ("theta", "anchor_intercept", "other_intercept",
                         "slope", "sigma_raw")
            if True
        }
    except Exception as exc:  # noqa: BLE001 - fall back, never fail the fit
        print(f"  pathfinder failed ({exc}); using the static ladder")
        return build_inits(spec, payload), time.time() - start

    # One distinct Pathfinder draw per chain, spread across the returned set.
    n_draws = len(variables["theta"])
    rows = np.linspace(0, n_draws - 1, spec.chains).astype(int)
    inits = []
    for row in rows:
        init = {
            "theta": variables["theta"][row].tolist(),
            "anchor_intercept": sorted(variables["anchor_intercept"][row].tolist()),
            "slope": variables["slope"][row].tolist(),
            "sigma_raw": np.atleast_2d(variables["sigma_raw"][row]).tolist(),
        }
        if spec.n_channels > 1:
            init["other_intercept"] = np.atleast_2d(
                variables["other_intercept"][row]
            ).tolist()
        inits.append(init)
    return inits, time.time() - start


def _source_hash(stan_file: Path) -> str:
    return hashlib.sha256(stan_file.read_bytes()).hexdigest()[:16]


def compile_model(stan_file: Path, *, cache_dir: Path = COMPILE_CACHE):
    """Compile with a content-addressed cache keyed on the Stan source."""
    from cmdstanpy import CmdStanModel

    digest = _source_hash(stan_file)
    target_dir = cache_dir / f"{stan_file.stem}-{digest}"
    target_dir.mkdir(parents=True, exist_ok=True)
    cached_src = target_dir / stan_file.name
    if not cached_src.exists():
        shutil.copy2(stan_file, cached_src)
    return CmdStanModel(stan_file=str(cached_src), cpp_options={"STAN_THREADS": True})


def fit_model(
    spec: ModelSpec,
    payload: StanPayload,
    *,
    output_dir: Path,
    threads_per_chain: int = 2,
    show_console: bool = False,
    inits: list[dict] | None = None,
    **overrides,
):
    """Run CmdStan and return the CmdStanMCMC object."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "stan_data.json"
    stan_data = {k: v for k, v in payload.data.items() if not k.startswith("_")}
    data_path.write_text(json.dumps(stan_data))

    model = compile_model(spec.stan_file)
    settings = {
        "chains": spec.chains,
        "iter_warmup": spec.iter_warmup,
        "iter_sampling": spec.iter_sampling,
        "adapt_delta": spec.adapt_delta,
        "max_treedepth": spec.max_treedepth,
        "seed": spec.seed,
    }
    settings.update(overrides)
    if inits is None:
        inits = build_inits(spec, payload)
    return model.sample(
        inits=inits,
        data=str(data_path),
        threads_per_chain=threads_per_chain,
        output_dir=str(output_dir / "chains"),
        show_console=show_console,
        **settings,
    )


__all__ = [
    "StanPayload",
    "assignment_inits",
    "build_inits",
    "pathfinder_inits",
    "build_payload",
    "compile_model",
    "fit_model",
]
