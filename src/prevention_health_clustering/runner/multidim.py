"""Payload for the multidimensional mixture.

Rows are the shared person-wave set of a multidim contract: every Gaussian
channel observed on every row, with the chronic count (optional) and the
mortality event carried as masked channels.

Gaussian channels are chosen by name (``gauss_channels``): the paper's runs
pair a physical variant (``theta`` or ``h``) with the mental GRM
(``theta_ment_nodepr``); the archived four-channel fits paired
``theta_phys_func`` with the mental GRM and a chronic count.

Persistence is class and channel specific (rho[k, c]); the innovation and
"spike" scales are channel specific and common to the classes.

Mortality is a Gompertz-Makeham hazard, class-specific level and slope with a
common Makeham constant, integrated over each row's interval at risk
(``mort_gap``: years to the person's next observed row, one year after their
last). ``use_mortality`` False leaves its parameters zero-sized.

Windows. The Gaussian and chronic channels use the fitted window and are
scored on the held-out window (``holdout_last_k``). Mortality uses the
person's FULL window and is never held out -- a decedent's single event sits
by construction at their last observed wave, so holding out the last rows
would delete every event from the likelihood rather than test the channel.
``full_end`` carries that window into Stan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.config import DEFAULT_AGE_CENTER, DEFAULT_AGE_SCALE

GAUSS_CHANNELS = ("theta", "theta_ment_nodepr")
ARCHIVED_CHANNELS = ("theta_phys_func", "theta_ment_nodepr")


@dataclass
class MultidimPayload:
    data: dict
    person_ids: np.ndarray
    channel_moments: dict[str, dict[str, float]]
    channels: tuple[str, ...]


def build_multidim_payload(
    long: pd.DataFrame,
    *,
    gauss_channels: tuple[str, ...] = GAUSS_CHANNELS,
    n_classes: int = 3,
    ar_mode: int = 0,
    holdout_last_k: int | None = None,
    min_person_obs: int | None = None,
    use_chronic: bool = False,
    use_mortality: bool = True,
    age_center: int = DEFAULT_AGE_CENTER,
    age_scale: float = DEFAULT_AGE_SCALE,
    emit_person_quantities: bool = False,
    channel_weight: tuple[float, ...] | None = None,
    grainsize: int = 0,
    gomp_slope_prior: tuple[float, float] = (0.09, 0.03),
    makeham_prior_scale: float = 0.001,
) -> MultidimPayload:
    frame = long.sort_values(["pidp", "age"]).reset_index(drop=True)
    missing = [c for c in gauss_channels if c not in frame.columns]
    if missing:
        raise KeyError(f"contract lacks channel(s) {missing}")
    if use_chronic and "n_chronic" not in frame.columns:
        raise KeyError("use_chronic needs an n_chronic column")
    dropped = 0
    # The sample filter is applied to EVERY variant, not only the holdout
    # ones, so that baseline / AR / holdout differ by specification alone.
    if min_person_obs is not None:
        if holdout_last_k is not None and min_person_obs <= holdout_last_k:
            raise ValueError("min_person_obs must exceed holdout_last_k.")
        n = frame.groupby("pidp")["age"].transform("size")
        dropped = int((n < min_person_obs).groupby(frame["pidp"]).first().sum())
        frame = frame[n >= min_person_obs].reset_index(drop=True)
        if frame.empty:
            raise ValueError("min_person_obs leaves nobody in the sample.")

    codes, person_ids = pd.factorize(frame["pidp"], sort=True)
    frame = frame.assign(_person=codes)
    bounds = frame.groupby("_person").agg(lo=("_person", lambda s: s.index.min()),
                                          hi=("_person", lambda s: s.index.max()))
    lo = bounds["lo"].to_numpy() + 1
    hi = bounds["hi"].to_numpy() + 1

    if holdout_last_k is None:
        fit_start, fit_end = lo, hi
        hold_start = np.zeros(len(lo), dtype=int)
        hold_end = np.zeros(len(lo), dtype=int)
    else:
        held = frame.groupby("_person").cumcount(ascending=False) < holdout_last_k
        held = held.to_numpy()
        fit_start, fit_end, hold_start, hold_end = [], [], [], []
        for p in range(len(lo)):
            block = np.arange(lo[p] - 1, hi[p])
            h = held[block]
            fit_start.append(int(block[~h].min()) + 1)
            fit_end.append(int(block[~h].max()) + 1)
            hold_start.append(int(block[h].min()) + 1 if h.any() else 0)
            hold_end.append(int(block[h].max()) + 1 if h.any() else 0)
        fit_start = np.array(fit_start); fit_end = np.array(fit_end)
        hold_start = np.array(hold_start); hold_end = np.array(hold_end)

    age = frame["age"].to_numpy(dtype=float)
    a = (age - age_center) / age_scale
    design = np.column_stack([np.ones_like(a), a, a**2])

    y, moments = [], {}
    for ch in gauss_channels:
        v = frame[ch].to_numpy(dtype=float)
        m, s = float(v.mean()), float(v.std(ddof=1))
        moments[ch] = {"mean": m, "sd": s}
        y.append((v - m) / s)

    if use_chronic:
        chronic = frame["n_chronic"].to_numpy(dtype=float)
        chronic_obs = np.isfinite(chronic).astype(int)
        chronic_y = np.where(chronic_obs == 1, np.nan_to_num(chronic), 0).astype(int)
        chronic_log_mean = float(np.log(max(chronic[chronic_obs == 1].mean(), 0.05)))
    else:
        chronic_obs = np.zeros(len(frame), dtype=int)
        chronic_y = np.zeros(len(frame), dtype=int)
        chronic_log_mean = 0.0

    mort = frame["mort_event"].to_numpy(dtype=float) if "mort_event" in frame.columns \
        else np.full(len(frame), np.nan)
    mort_obs = (np.isfinite(mort) & use_mortality).astype(int)
    mort_y = np.where(mort_obs == 1, np.nan_to_num(mort), 0).astype(int)
    # years at risk from each row: to the person's next observed row, one year after the last
    next_age = np.concatenate([age[1:], [np.nan]])
    same_person = np.concatenate([codes[1:] == codes[:-1], [False]])
    mort_gap = np.where(same_person, np.maximum(next_age - age, 1.0), 1.0)
    mort_age = age - age_center
    if mort_obs.sum() > 0:
        rate = float(np.clip(mort_y[mort_obs == 1].sum() / mort_gap[mort_obs == 1].sum(), 1e-5, 0.5))
        mort_log_mean = float(np.log(rate))
    else:
        mort_log_mean = -5.0

    if channel_weight is None:
        channel_weight = (1.0,) * (len(gauss_channels) + 2)

    data = {
        "N_obs": int(len(frame)),
        "N_person": int(len(lo)),
        "K": int(n_classes),
        "C": len(gauss_channels),
        "P": 3,
        "y": [v.tolist() for v in y],
        "X": design.tolist(),
        "fit_start": fit_start.tolist(),
        "fit_end": fit_end.tolist(),
        "hold_start": hold_start.tolist(),
        "hold_end": hold_end.tolist(),
        "full_end": hi.tolist(),
        "ar_mode": int(ar_mode),
        "age_gap": (np.concatenate([[0.0], np.maximum(
            np.diff(age), 1.0)]).tolist()
            if ar_mode != 0 else []),
        "use_chronic": int(bool(use_chronic)),
        "chronic_obs": chronic_obs.tolist(),
        "chronic_y": chronic_y.tolist(),
        "use_mortality": int(bool(use_mortality)),
        "mort_obs": mort_obs.tolist(),
        "mort_y": mort_y.tolist(),
        "mort_age": mort_age.tolist(),
        "mort_gap": mort_gap.tolist(),
        "homosigma": 1,
        "anchor_channel": 1,
        "channel_n_obs": [int(len(frame))] * len(gauss_channels),
        "channel_weight": list(channel_weight),
        "alpha_prior_scale": 2.0,
        "coef_prior_scale": [1.0, 0.5],
        "sigma_prior_location": 0.8,
        "sigma_prior_scale": 0.5,
        "theta_prior_concentration": 1.5,
        "rho_prior_alpha": 2.0,
        "rho_prior_beta": 2.0,
        # Centred well below the residual scale: the default is that most
        # variation is signal and the data have to argue for noise.
        "sigma_meas_prior_location": 0.0,
        "sigma_meas_prior_scale": 0.4,
        "chronic_log_mean": chronic_log_mean,
        "mort_log_mean": mort_log_mean,
        "gomp_slope_prior_mean": float(gomp_slope_prior[0]),
        "gomp_slope_prior_sd": float(gomp_slope_prior[1]),
        "makeham_prior_scale": float(makeham_prior_scale),
        "emit_person_quantities": int(emit_person_quantities),
        "grainsize": int(grainsize) if grainsize > 0
        else max(1, len(lo) // 64),
        "_dropped_min_obs": dropped,
    }
    return MultidimPayload(data=data, person_ids=person_ids.to_numpy(),
                           channel_moments=moments, channels=tuple(gauss_channels))


def multidim_inits(payload: MultidimPayload, *, jitter: float = 0.15,
                   n_chains: int = 4, seed: int = 20260826) -> list[dict]:
    """Partition inits: k-means on per-person channel means, then per-class OLS."""
    from sklearn.cluster import KMeans

    d = payload.data
    K, C, P = d["K"], d["C"], d["P"]
    X = np.asarray(d["X"])
    Y = np.asarray(d["y"])                      # (C, N)
    person = np.zeros(d["N_obs"], dtype=int)
    for i, (s, e) in enumerate(zip(d["fit_start"], d["full_end"])):
        person[s - 1:e] = i
    means = np.column_stack([
        pd.Series(Y[c]).groupby(person).mean().to_numpy() for c in range(C)])
    labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(means)
    order = np.argsort([means[labels == k, 0].mean() for k in range(K)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[v] for v in labels])
    row_label = labels[person]

    coefs = np.zeros((C, K, P))
    for c in range(C):
        for k in range(K):
            m = row_label == k
            coefs[c, k] = np.linalg.lstsq(X[m], Y[c][m], rcond=None)[0]
    shares = np.array([(labels == k).mean() for k in range(K)])
    sigma = float(np.std(np.concatenate(
        [Y[c] - (X * coefs[c][row_label]).sum(1) for c in range(C)])))
    # The count channel needs partition-informed starts for the SAME reason
    # the Gaussian channels do. Starting every class at the same chronic
    # coefficients leaves the label symmetry unbroken on that channel, and
    # when AR(1) weakens the Gaussian separation the chains drift into
    # different modes (observed: R-hat 13.2 on coef_chronic with identical
    # starts). A per-class log-linear fit on the k-means partition breaks it.
    chronic_coefs = np.zeros((K, P))
    if d["use_chronic"] == 1:
        chronic = np.asarray(d["chronic_y"], dtype=float)
        cobs = np.asarray(d["chronic_obs"]) == 1
        for k in range(K):
            m = cobs & (row_label == k)
            if m.sum() > P + 1:
                beta = np.linalg.lstsq(X[m], np.log(chronic[m] + 0.5), rcond=None)[0]
                chronic_coefs[k] = beta
                chronic_coefs[k, 0] -= d["chronic_log_mean"]   # the model re-adds it
    # Mortality: the partition's crude yearly death rate per class, relative
    # to the pooled rate, starts the class levels apart in the right order.
    mort_start = np.zeros(K)
    if d["use_mortality"] == 1:
        my = np.asarray(d["mort_y"], dtype=float); mg = np.asarray(d["mort_gap"], dtype=float)
        mo = np.asarray(d["mort_obs"]) == 1
        pooled = my[mo].sum() / mg[mo].sum()
        for k in range(K):
            m = mo & (row_label == k)
            if m.sum() > 0 and my[m].sum() > 0:
                mort_start[k] = np.log((my[m].sum() / mg[m].sum()) / pooled)

    rng = np.random.default_rng(seed)
    inits = []
    for _ in range(n_chains):
        j = lambda scale=jitter: rng.normal(0, scale)
        anchor = np.sort([coefs[0, k, 0] + j() for k in range(K)])
        th = np.array([max(shares[k] + j(0.02), 0.05) for k in range(K)])
        init = {
            "theta": (th / th.sum()).tolist(),
            "anchor_intercept": anchor.tolist(),
            "other_intercept": [[coefs[c, k, 0] + j() for c in range(1, C)] for k in range(K)],
            "slope": [[[coefs[c, k, p] + j(0.05) for p in range(1, P)]
                       for k in range(K)] for c in range(C)],
            "sigma_raw": [[max(sigma + j(0.05), 0.1)] * C],
        }
        if d["use_chronic"] == 1:
            init["coef_chronic_raw"] = [
                [chronic_coefs[k, 0] + j(0.1)]
                + [chronic_coefs[k, q] + j(0.05) for q in range(1, P)]
                for k in range(K)]
            init["phi_chronic"] = [float(max(2.0 + j(0.5), 0.5))]
        if d["use_mortality"] == 1:
            init["log_b_raw"] = [float(mort_start[k] + j(0.3)) for k in range(K)]
            init["gomp_slope"] = [float(np.clip(d["gomp_slope_prior_mean"] + j(0.01), 0.02, 0.2))
                                  for _ in range(K)]
            init["makeham"] = [float(abs(0.5 * d["makeham_prior_scale"] + j(0.1 * d["makeham_prior_scale"])))]
        if d["ar_mode"] != 0:
            # Under ar_mode 2 the persistence posterior sits near 0.95, not
            # 0.5: once transient noise is taken out of the residual, what is
            # left is highly persistent. Starting at 0.5 leaves a long climb
            # that chains finish at different points, which shows up as a
            # large rho R-hat even though the model is identified.
            centre = 0.90 if d["ar_mode"] == 2 else 0.5
            init["rho"] = [[float(np.clip(centre + j(0.05), 0.05, 0.97))
                            for _ in range(C)] for _ in range(K)]
        if d["ar_mode"] == 2:
            # Open on the rho/sigma_meas ridge rather than at either end of
            # it, so chains do not set off in opposite directions.
            init["sigma_meas"] = [float(np.clip(0.35 + j(0.05), 0.05, 1.0))
                                  for _ in range(d["C"])]
        inits.append(init)
    return inits


def write_multidim_data(payload: MultidimPayload, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stan_data.json"
    path.write_text(json.dumps(
        {k: v for k, v in payload.data.items() if not k.startswith("_")}))
    return path


__all__ = ["ARCHIVED_CHANNELS", "GAUSS_CHANNELS", "MultidimPayload", "build_multidim_payload",
           "multidim_inits", "write_multidim_data"]
