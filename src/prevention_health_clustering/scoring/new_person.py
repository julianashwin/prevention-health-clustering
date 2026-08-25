"""Held-out new-person scoring for the Gaussian panel models.

Two estimands, both defined by the predecessor's corrected protocol:

Same-window new-person log predictive density
    log p(y_i | D_train) = log( (1/D) * sum_d sum_k theta_dk * p(y_i | phi_dk) )
    The held-out person's complete window is integrated over class AND
    training-posterior uncertainty. No class is chosen from y_i.

Temporal (landmark) prediction
    w_idk = theta_dk p(h_i | phi_dk) / sum_{d'k'} theta_d'k' p(h_i | phi_d'k')
    log p(y_i+ | h_i, D_train) = log sum_dk w_idk p(y_i+ | phi_dk)
    Weights come from strictly pre-landmark history and are then FIXED while
    future outcomes are scored.

The normalisation is a single logsumexp over the whole draw-by-class grid.
Normalising within each draw first and averaging computes an expectation of
ratios — the defect the August 2026 audit quarantined. The wrong rule is
implemented here only inside the test suite, to prove the two differ.

Leakage guard: because weights depend only on history, perturbing every future
outcome must change the class posterior by exactly zero. ``leakage_probe``
checks that at 1e-12, as a permanent regression detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.models.registry import ModelSpec

LEAKAGE_TOLERANCE = 1e-12


@dataclass
class GaussianPosterior:
    """Training-posterior draws in array form: D draws, K classes, C channels."""

    theta: np.ndarray        # (D, K)
    coef: np.ndarray         # (D, C, K, P)
    sigma: np.ndarray        # (D, C)  (homosigma; class-specific would be (D,K,C))
    channels: tuple[str, ...]
    moments: dict[str, dict[str, float]]
    age_center: int = 55
    age_scale: float = 10.0

    @property
    def n_draws(self) -> int:
        return self.theta.shape[0]

    @property
    def n_classes(self) -> int:
        return self.theta.shape[1]


def posterior_from_fit(fit, spec: ModelSpec, moments: dict) -> GaussianPosterior:
    """Extract the parameter draws a scorer needs from a CmdStanMCMC."""
    draws = fit.draws_pd()
    d = len(draws)
    k, c, p = spec.n_classes, spec.n_channels, spec.design_width
    theta = np.column_stack([draws[f"theta[{i}]"] for i in range(1, k + 1)])
    coef = np.zeros((d, c, k, p))
    for ci in range(1, c + 1):
        for ki in range(1, k + 1):
            for pi in range(1, p + 1):
                coef[:, ci - 1, ki - 1, pi - 1] = draws[f"coef[{ci},{ki},{pi}]"]
    sigma = np.column_stack([draws[f"sigma[1,{ci}]"] for ci in range(1, c + 1)])
    return GaussianPosterior(
        theta=theta, coef=coef, sigma=sigma,
        channels=spec.channels, moments=moments,
    )


def _row_loglik(
    posterior: GaussianPosterior, frame: pd.DataFrame
) -> np.ndarray:
    """(rows, D, K) log density of each observed row under each draw and class.

    Missing channel values contribute zero, matching the fitting likelihood.
    """
    ages = frame["age"].to_numpy(dtype=float)
    a = (ages - posterior.age_center) / posterior.age_scale
    design = np.column_stack([np.ones_like(a), a, a**2])[
        :, : posterior.coef.shape[3]
    ]  # (rows, P)

    total = np.zeros((len(frame), posterior.n_draws, posterior.n_classes))
    for ci, channel in enumerate(posterior.channels):
        raw = frame[channel].to_numpy(dtype=float)
        observed = np.isfinite(raw)
        if not observed.any():
            continue
        m = posterior.moments[channel]
        z = (raw[observed] - m["mean"]) / m["sd"]
        # mu: (rows_obs, D, K) via einsum over the design columns
        mu = np.einsum("rp,dkp->rdk", design[observed], posterior.coef[:, ci])
        sd = posterior.sigma[:, ci][None, :, None]
        resid = z[:, None, None] - mu
        total[observed] += (
            -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * (resid / sd) ** 2
        )
    return total


@dataclass
class PersonScore:
    pidp: int
    log_predictive_density: float
    n_history: int
    n_scored: int
    class_posterior: np.ndarray  # (K,) marginalised over draws


def score_people(
    posterior: GaussianPosterior,
    frame: pd.DataFrame,
    *,
    landmark_age: int | None = None,
) -> pd.DataFrame:
    """Score held-out people; one row per person.

    ``landmark_age`` None gives the same-window estimand (empty history, class
    weights are the training prior). With a landmark, rows at ages <= landmark
    are history and set the weights; rows strictly above are scored.
    """
    frame = frame.sort_values(["pidp", "age"]).reset_index(drop=True)
    log_prior = np.log(posterior.theta) - np.log(posterior.n_draws)  # (D, K)
    rows_lp = _row_loglik(posterior, frame)

    records = []
    for pidp, group in frame.groupby("pidp", sort=True):
        idx = group.index.to_numpy()
        if landmark_age is None:
            history_idx = idx[:0]
            score_idx = idx
        else:
            ages = group["age"].to_numpy()
            history_idx = idx[ages <= landmark_age]
            score_idx = idx[ages > landmark_age]
            if len(history_idx) == 0:
                raise ValueError(
                    f"Person {pidp} has no history at or before age {landmark_age}."
                )
            if len(score_idx) == 0:
                continue
        # Window discipline: every history age strictly precedes every scored
        # age for this person (trivially true by construction here; kept as an
        # explicit guard against future refactors).
        if len(history_idx) and len(score_idx):
            if frame.loc[history_idx, "age"].max() >= frame.loc[score_idx, "age"].min():
                raise AssertionError("History ages overlap scored ages.")

        history_lp = rows_lp[history_idx].sum(axis=0)          # (D, K)
        unnormalised = log_prior + history_lp
        log_weights = unnormalised - logsumexp(unnormalised)   # grid-normalised
        future_lp = rows_lp[score_idx].sum(axis=0)             # (D, K)

        lpd = float(logsumexp(log_weights + future_lp))
        class_post = np.exp(log_weights).sum(axis=0)           # marginal over draws
        records.append(
            PersonScore(
                pidp=int(pidp),
                log_predictive_density=lpd,
                n_history=int(len(history_idx)),
                n_scored=int(len(score_idx)),
                class_posterior=class_post,
            )
        )

    out = pd.DataFrame(
        {
            "pidp": [r.pidp for r in records],
            "log_predictive_density": [r.log_predictive_density for r in records],
            "n_history": [r.n_history for r in records],
            "n_scored": [r.n_scored for r in records],
        }
    )
    for k in range(posterior.n_classes):
        out[f"prob_class{k + 1}"] = [r.class_posterior[k] for r in records]
    return out



def classify_people(
    posterior: GaussianPosterior, frame: pd.DataFrame
) -> pd.DataFrame:
    """Class posterior CONDITIONED on a person's full observed window.

    This is held-out classification — a descriptive quantity — not held-out
    prediction. The same-window predictive estimand (``score_people`` with no
    landmark) deliberately reports the prior as its class posterior, because
    scored outcomes must not choose a class. Conditioning on the window is
    legitimate when the quantity is described as classification; the audit
    quarantined pipelines that conditioned on outcomes and then presented the
    result as predictive skill.
    """
    frame = frame.sort_values(["pidp", "age"]).reset_index(drop=True)
    log_prior = np.log(posterior.theta) - np.log(posterior.n_draws)
    rows_lp = _row_loglik(posterior, frame)
    records = []
    for pidp, group in frame.groupby("pidp", sort=True):
        window_lp = rows_lp[group.index.to_numpy()].sum(axis=0)
        grid = log_prior + window_lp
        grid -= logsumexp(grid)
        class_post = np.exp(grid).sum(axis=0)
        records.append((int(pidp), class_post))
    out = pd.DataFrame({"pidp": [r[0] for r in records]})
    for k in range(posterior.n_classes):
        out[f"prob_class{k + 1}"] = [r[1][k] for r in records]
    out["modal_class"] = (
        out[[f"prob_class{k + 1}" for k in range(posterior.n_classes)]]
        .to_numpy().argmax(1) + 1
    )
    return out


def leakage_probe(
    posterior: GaussianPosterior,
    frame: pd.DataFrame,
    *,
    landmark_age: int,
    perturbation: float = 1000.0,
) -> float:
    """Max |change| in any class posterior when every FUTURE outcome moves.

    Must be exactly zero by construction — the weights see only history — so
    any nonzero value is a regression, not noise. Gate at 1e-12.
    """
    base = score_people(posterior, frame, landmark_age=landmark_age)
    perturbed = frame.copy()
    future = perturbed["age"] > landmark_age
    for channel in posterior.channels:
        perturbed.loc[future, channel] = perturbed.loc[future, channel] + perturbation
    shifted = score_people(posterior, perturbed, landmark_age=landmark_age)
    merged = base.merge(shifted, on="pidp", suffixes=("_a", "_b"))
    deltas = [
        (merged[f"prob_class{k + 1}_a"] - merged[f"prob_class{k + 1}_b"]).abs().max()
        for k in range(posterior.n_classes)
    ]
    return float(max(deltas))


__all__ = [
    "LEAKAGE_TOLERANCE",
    "GaussianPosterior",
    "classify_people",
    "leakage_probe",
    "posterior_from_fit",
    "score_people",
]
