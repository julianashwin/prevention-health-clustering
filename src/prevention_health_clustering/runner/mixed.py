"""Payload construction and initialisation for the mixed-health model.

The five channels are independently ragged: PCS/MCS come standardised, SRH is
kept ordinal 1..5, chronic and ADL are counts. Each channel carries its own
value and design arrays plus per-person [start, end] windows, with start = 0
meaning the person has no rows for that channel — the pattern the predecessor's
mixed_health_channels contract used, minus the tanh-capped parameterisation.

The mortality/survival channel is deliberately absent (five-channel family).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.config import (
    DEFAULT_AGE_CENTER,
    DEFAULT_AGE_SCALE,
)
from prevention_health_clustering.models.registry import ModelSpec

GAUSS_CHANNELS = ("sf12pcs_dv", "sf12mcs_dv")
SRH_COLUMN = "general_health_combined"
CHRONIC_COLUMN = "chronic_condition_count"
ADL_COLUMN = "adl_limitation_count"


@dataclass
class MixedPayload:
    data: dict
    person_ids: np.ndarray
    channel_moments: dict[str, dict[str, float]]


def _design(ages: np.ndarray, p: int, center: int, scale: float) -> np.ndarray:
    a = (ages - center) / scale
    cols = [np.ones_like(a), a]
    if p == 3:
        cols.append(a**2)
    return np.column_stack(cols)


def _ragged_block(
    frame: pd.DataFrame,
    column: str,
    person_codes: np.ndarray,
    n_person: int,
    p: int,
    center: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Values, design, and per-person windows for one channel.

    Rows are already sorted by (person, age); a channel's observed rows stay
    contiguous per person after the mask, so windows are well defined.
    """
    mask = frame[column].notna().to_numpy()
    values = frame.loc[mask, column].to_numpy(dtype=float)
    ages = frame.loc[mask, "age"].to_numpy(dtype=float)
    codes = person_codes[mask]

    start = np.zeros(n_person, dtype=int)
    end = np.zeros(n_person, dtype=int)
    if len(codes):
        first = np.searchsorted(codes, np.arange(n_person), side="left")
        last = np.searchsorted(codes, np.arange(n_person), side="right")
        has = last > first
        start[has] = first[has] + 1  # 1-based; 0 = no rows
        end[has] = last[has]
    return values, _design(ages, p, center, scale), start, end


def build_mixed_payload(
    spec: ModelSpec,
    frame: pd.DataFrame,
    *,
    age_center: int = DEFAULT_AGE_CENTER,
    age_scale: float = DEFAULT_AGE_SCALE,
    adl_weight: float = 1.0,
    emit_person_quantities: bool = False,
    grainsize: int = 0,
) -> MixedPayload:
    """Assemble the ragged five-channel Stan data block.

    ``frame`` is person-wave rows (duplicate ages allowed, matching the
    predecessor's nominal-age mixed contract) restricted to the roster and the
    age window, with columns pidp, age, the two SF-12 scores, SRH, chronic and
    ADL counts. Channel missingness is handled per channel.
    """
    required = ["pidp", "age", *GAUSS_CHANNELS, SRH_COLUMN, CHRONIC_COLUMN, ADL_COLUMN]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing columns: {', '.join(missing)}")

    frame = frame.sort_values(["pidp", "age"]).reset_index(drop=True)
    codes, person_ids = pd.factorize(frame["pidp"], sort=True)
    n_person = len(person_ids)
    p = spec.design_width

    moments: dict[str, dict[str, float]] = {}
    data: dict = {
        "N_person": int(n_person),
        "K": int(spec.n_classes),
        "P": int(p),
        "C_gauss": len(GAUSS_CHANNELS),
        "anchor_channel": 1,  # PCS; class 1 = worst physical health
    }

    # Gaussian channels: standardise, pad to a common length for the array
    # container (Stan reads only within each window).
    gauss_vals, gauss_X, gauss_start, gauss_end = [], [], [], []
    for channel in GAUSS_CHANNELS:
        values, X, start, end = _ragged_block(
            frame, channel, codes, n_person, p, age_center, age_scale
        )
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        moments[channel] = {"mean": mean, "sd": sd}
        gauss_vals.append((values - mean) / sd)
        gauss_X.append(X)
        gauss_start.append(start)
        gauss_end.append(end)
    n_max = max(len(v) for v in gauss_vals)
    data["N_gauss"] = [int(len(v)) for v in gauss_vals]
    data["y_gauss_pad"] = [
        np.pad(v, (0, n_max - len(v))).tolist() for v in gauss_vals
    ]
    data["X_gauss_pad"] = [
        np.vstack([X, np.zeros((n_max - len(X), p))]).tolist() for X in gauss_X
    ]
    data["gauss_start"] = np.vstack(gauss_start).astype(int).tolist()
    data["gauss_end"] = np.vstack(gauss_end).astype(int).tolist()

    # SRH: strict integers 1..5.
    values, X, start, end = _ragged_block(
        frame, SRH_COLUMN, codes, n_person, p, age_center, age_scale
    )
    srh = np.rint(values).astype(int)
    if ((srh < 1) | (srh > 5)).any():
        raise ValueError("SRH values outside 1..5 after rounding.")
    data.update(
        N_srh=int(len(srh)), srh_y=srh.tolist(), X_srh=X.tolist(),
        srh_start=start.tolist(), srh_end=end.tolist(),
    )

    # Chronic count.
    values, X, start, end = _ragged_block(
        frame, CHRONIC_COLUMN, codes, n_person, p, age_center, age_scale
    )
    chronic = np.rint(values).astype(int)
    data.update(
        N_chronic=int(len(chronic)), chronic_y=chronic.tolist(),
        X_chronic=X.tolist(), chronic_start=start.tolist(),
        chronic_end=end.tolist(),
        chronic_log_mean=float(np.log(max(chronic.mean(), 0.05))),
    )

    # ADL hurdle count.
    values, X, start, end = _ragged_block(
        frame, ADL_COLUMN, codes, n_person, p, age_center, age_scale
    )
    adl = np.rint(values).astype(int)
    share_pos = float((adl > 0).mean()) if len(adl) else 0.5
    share_pos = min(max(share_pos, 0.02), 0.98)
    positive_mean = float(adl[adl > 0].mean()) if (adl > 0).any() else 1.0
    data.update(
        N_adl=int(len(adl)), adl_y=adl.tolist(), X_adl=X.tolist(),
        adl_start=start.tolist(), adl_end=end.tolist(),
        adl_zero_logit=float(np.log(share_pos / (1 - share_pos))),
        adl_log_positive_mean=float(np.log(max(positive_mean, 0.1))),
    )

    data["channel_weight"] = [1.0, 1.0, 1.0, 1.0, float(adl_weight)]
    data.update(
        alpha_prior_scale=float(spec.alpha_prior_scale),
        coef_prior_scale=[float(s) for s in spec.slope_prior_scales],
        sigma_prior_location=float(spec.sigma_prior_location),
        sigma_prior_scale=float(spec.sigma_prior_scale),
        theta_prior_concentration=float(spec.theta_prior_concentration),
        emit_person_quantities=int(emit_person_quantities),
        grainsize=int(grainsize or max(1, n_person // (spec.chains * 4))),
    )
    return MixedPayload(
        data=data, person_ids=np.asarray(person_ids), channel_moments=moments
    )


def mixed_assignment_inits(
    spec: ModelSpec,
    payload: MixedPayload,
    *,
    jitter: float = 0.15,
    seed: int | None = None,
) -> list[dict]:
    """Partition-informed inits, the recipe that identifies classes 2/3.

    K-means on per-person PCS/MCS means supplies the partition; per class,
    OLS on the gaussian channels and channel means on the others supply the
    starting values, theta comes from the shares, and each chain is jittered.
    """
    rng = np.random.default_rng(seed if seed is not None else spec.seed)
    k, p = spec.n_classes, spec.design_width
    d = payload.data
    n_person = d["N_person"]

    # Per-person gaussian means over each channel's own window.
    feats = np.full((n_person, d["C_gauss"]), np.nan)
    for c in range(d["C_gauss"]):
        y = np.asarray(d["y_gauss_pad"][c])
        start = np.asarray(d["gauss_start"])[c]
        end = np.asarray(d["gauss_end"])[c]
        for i in range(n_person):
            if start[i] > 0:
                feats[i, c] = y[start[i] - 1 : end[i]].mean()
    feats = np.where(np.isnan(feats), np.nanmean(feats, axis=0), feats)

    centers = feats[rng.choice(n_person, 1)]
    while len(centers) < k:
        d2 = ((feats[:, None, :] - centers[None]) ** 2).sum(-1).min(1)
        centers = np.vstack([centers, feats[rng.choice(n_person, p=d2 / d2.sum())]])
    for _ in range(50):
        labels = ((feats[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.vstack([
            feats[labels == g].mean(0) if (labels == g).any() else centers[g]
            for g in range(k)
        ])
        if np.allclose(new, centers):
            break
        centers = new

    theta = np.array([(labels == g).mean() for g in range(k)])
    # Order by the anchor (PCS) mean, ascending: class 1 = worst.
    order = np.argsort(centers[:, 0])
    theta = theta[order]
    relabel = np.empty(k, dtype=int)
    relabel[order] = np.arange(k)
    labels = relabel[labels]

    def gauss_ols(c: int) -> tuple[np.ndarray, float]:
        y = np.asarray(d["y_gauss_pad"][c])
        X = np.asarray(d["X_gauss_pad"][c])
        start = np.asarray(d["gauss_start"])[c]
        end = np.asarray(d["gauss_end"])[c]
        coef = np.zeros((k, p))
        resid_sd = 1.0
        residuals = []
        for g in range(k):
            rows = np.concatenate([
                np.arange(start[i] - 1, end[i])
                for i in range(n_person) if labels[i] == g and start[i] > 0
            ]) if any(labels[i] == g and start[i] > 0 for i in range(n_person)) else None
            if rows is None or len(rows) < 10:
                continue
            b, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
            coef[g] = b
            residuals.append(y[rows] - X[rows] @ b)
        if residuals:
            resid_sd = max(float(np.concatenate(residuals).std()), 0.2)
        return coef, resid_sd

    coef_pcs, sd_pcs = gauss_ols(0)
    coef_mcs, sd_mcs = gauss_ols(1)

    inits = []
    for _ in range(spec.chains):
        jit = lambda shape: rng.normal(0.0, jitter, size=shape)  # noqa: E731
        anchor = np.sort(coef_pcs[:, 0] + jit(k))
        init = {
            "theta": (np.maximum(theta, 1e-3) / np.maximum(theta, 1e-3).sum()).tolist(),
            "anchor_intercept": anchor.tolist(),
            "other_intercept": (coef_mcs[:, [0]] + jit((k, 1))).tolist(),
            "slope_gauss": [
                (coef_pcs[:, 1:] + 0.3 * jit((k, p - 1))).tolist(),
                (coef_mcs[:, 1:] + 0.3 * jit((k, p - 1))).tolist(),
            ],
            "sigma_gauss": [sd_pcs, sd_mcs],
            # Ordinal/count channels start near their empirical offsets with
            # a severity gradient consistent with the anchor ordering.
            "coef_srh_raw": np.column_stack(
                [np.linspace(1.5, -1.5, k) + jit(k), np.zeros((k, p - 1)) + 0.1 * jit((k, p - 1)) if p > 1 else None]
            ).tolist() if p > 1 else np.linspace(1.5, -1.5, k)[:, None].tolist(),
            "cut_srh_gap": [2.0, 2.0, 2.0],
            "coef_chronic_raw": np.column_stack(
                [np.linspace(0.5, -0.5, k) + jit(k), np.zeros((k, p - 1))]
            ).tolist(),
            "phi_chronic": 8.0,
            "coef_adl_zero_raw": np.column_stack(
                [np.linspace(1.0, -1.0, k) + jit(k), np.zeros((k, p - 1))]
            ).tolist(),
            "coef_adl_count_raw": np.column_stack(
                [np.linspace(0.3, -0.3, k) + jit(k), np.zeros((k, p - 1))]
            ).tolist(),
            "phi_adl": 4.0,
        }
        inits.append(init)
    return inits


def write_mixed_data(payload: MixedPayload, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stan_data.json"
    path.write_text(json.dumps({k: v for k, v in payload.data.items()
                                if not k.startswith("_")}))
    return path


__all__ = [
    "ADL_COLUMN",
    "CHRONIC_COLUMN",
    "GAUSS_CHANNELS",
    "MixedPayload",
    "SRH_COLUMN",
    "build_mixed_payload",
    "mixed_assignment_inits",
    "write_mixed_data",
]
