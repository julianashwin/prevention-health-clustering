"""Partial K-means over incompletely observed age profiles.

Port of the predecessor pipeline's masked-MSE clustering
(ukhls_health_clusters.clustering.pipeline), the method behind the PCS
construction note's Figure 5: each person is a vector over ages with most
entries missing; distances average squared deviations over OBSERVED entries
only; centroids average observed member values per age. Initialisation seeds
sklearn KMeans on zero-imputed standardised features with seeds 42..42+n_init-1
and the best of n_init runs by total assigned distance wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RANDOM_STATE = 42
DEFAULT_N_INIT = 50


def _standardize_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.astype(float).copy()
    mean = out.mean(axis=0, skipna=True)
    std = out.std(axis=0, skipna=True, ddof=0)
    out = out.sub(mean, axis=1).div(std.replace(0, np.nan), axis=1)
    zero = std.index[std.isna() | (std == 0)]
    if len(zero):
        out.loc[:, zero] = out.loc[:, zero].where(df[zero].isna(), 0.0)
    return out


def _distances(X, mask, counts, centroids):
    filled = np.where(mask, X, 0.0)
    d = ((filled**2).sum(axis=1, keepdims=True)
         - 2.0 * filled @ centroids.T
         + mask.astype(float) @ (centroids.T**2)) / counts[:, None]
    return np.maximum(d, 0.0)


def _centroids(X, mask, labels, k, previous):
    cent = previous.copy()
    for c in range(k):
        rows = labels == c
        if not rows.any():
            continue
        obs = mask[rows]
        n = obs.sum(axis=0)
        s = np.where(obs, X[rows], 0.0).sum(axis=0)
        cent[c] = np.where(n > 0, s / np.maximum(n, 1), previous[c])
    return cent


def partial_kmeans(
    frame: pd.DataFrame,
    *,
    n_clusters: int,
    n_init: int = DEFAULT_N_INIT,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> np.ndarray:
    """Cluster rows of a person-by-age frame using observed entries only."""
    std = _standardize_with_missing(frame)
    mask = std.notna().to_numpy()
    counts = mask.sum(axis=1).astype(float)
    if (counts <= 0).any():
        raise ValueError("Every row needs at least one observed entry.")
    X = np.nan_to_num(std.to_numpy(float), nan=0.0)

    best_labels, best_obj = None, np.inf
    for i in range(n_init):
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE + i,
                    n_init=1, max_iter=50, tol=1e-4)
        labels = km.fit_predict(X)
        cent = _centroids(X, mask, labels, n_clusters,
                          np.zeros((n_clusters, X.shape[1])))
        prev_obj = None
        for _ in range(max_iter):
            d = _distances(X, mask, counts, cent)
            new_labels = d.argmin(axis=1)
            new_cent = _centroids(X, mask, new_labels, n_clusters, cent)
            obj = float(d[np.arange(len(new_labels)), new_labels].sum())
            done = prev_obj is not None and abs(prev_obj - obj) <= tol
            if np.array_equal(new_labels, labels) and done:
                labels, cent = new_labels, new_cent
                break
            labels, cent, prev_obj = new_labels, new_cent, obj
        d = _distances(X, mask, counts, cent)
        obj = float(d[np.arange(len(labels)), labels].sum())
        if obj < best_obj:
            best_labels, best_obj = labels.copy(), obj
    return best_labels


__all__ = ["partial_kmeans"]
