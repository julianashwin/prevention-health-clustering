"""SF-12 subscales and summary-score construction, US and UK variants.

Ports the scoring machinery recovered and developed in the PCS construction
note (julian_sandbox scripts 03-05):

* ``build_subscales``  - the eight conventional 0-100 subscales, with the
  standard SF-36 recalibration of the general-health item (5.0/4.4/3.4/2.0/1.0),
  which the note confirmed is what UKHLS applies.
* ``score_us``         - the published US-1998-normed varimax algorithm. This
  reproduces ``sf12pcs_dv``/``sf12mcs_dv`` to max abs error 0.0054 across all
  528,485 complete person-years, so every stage below is licence-free.
* ``derive_uk_structure`` - re-run the Ware procedure on a UK reference sample:
  weighted norms, weighted correlations, 2-component PCA, varimax and promax
  rotations, regression-method factor score coefficients.
* ``score_uk_variants``   - variants B (UK norms, US coefficients),
  C (UK varimax) and D (UK promax), T-anchored to mean 50 / SD 10 in the
  weighted UK reference population.
* ``score_phys_only``     - variant E: PF/RP/BP/GH only, weighted by their UK
  physical-factor loadings, no mental subscales. The measure that removes the
  orthogonality artifact (negative RE/MH weights) by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCALES: tuple[str, ...] = ("PF", "RP", "BP", "GH", "VT", "SF", "RE", "MH")
PHYS_SCALES: tuple[str, ...] = ("PF", "RP", "BP", "GH")

# Published SF-12v2 US 1998 general-population subscale means / SDs.
US_NORMS: dict[str, tuple[float, float]] = {
    "PF": (81.18122, 29.10588), "RP": (80.52856, 27.13526),
    "BP": (81.74015, 24.53019), "GH": (72.19795, 23.19041),
    "VT": (55.59090, 24.84380), "SF": (83.73973, 24.75775),
    "RE": (86.41429, 22.35543), "MH": (70.18217, 20.50597),
}

# Published SF-12v2 factor score coefficients (orthogonal / varimax, US 1998).
# The negative RE/MH weights in the PCS row are the orthogonality artifact.
US_COEF: dict[str, dict[str, float]] = {
    "PCS": {"PF": 0.42402, "RP": 0.35119, "BP": 0.31754, "GH": 0.24954,
            "VT": 0.02877, "SF": -0.00753, "RE": -0.19206, "MH": -0.22069},
    "MCS": {"PF": -0.22999, "RP": -0.12329, "BP": -0.09731, "GH": -0.01571,
            "VT": 0.23534, "SF": 0.26876, "RE": 0.43407, "MH": 0.48581},
}

# Standard SF-36 recalibration of the general-health item.
GH_RECODE: dict[int, float] = {1: 5.0, 2: 4.4, 3: 3.4, 4: 2.0, 5: 1.0}


def build_subscales(
    items: pd.DataFrame, *, gh_recalibrate: bool = True, require_complete: bool = True
) -> pd.DataFrame:
    """The eight conventional 0-100 subscales from the 12 items.

    Reverse-scored items (higher raw value = worse health): sf1, sf5, sf6a,
    sf6b — confirmed empirically in the construction note. With
    ``require_complete`` (the UKHLS convention), a person-year with ANY missing
    item gets all-NaN subscales.
    """
    d = items
    out = pd.DataFrame(index=d.index)
    out["PF"] = ((d.sf2a + d.sf2b) - 2) / 4 * 100
    out["RP"] = ((d.sf3a + d.sf3b) - 2) / 8 * 100
    out["BP"] = ((6 - d.sf5) - 1) / 4 * 100
    gh = d.sf1.map(GH_RECODE) if gh_recalibrate else (6 - d.sf1)
    out["GH"] = (gh - 1) / 4 * 100
    out["VT"] = ((6 - d.sf6b) - 1) / 4 * 100
    out["SF"] = (d.sf7 - 1) / 4 * 100
    out["RE"] = ((d.sf4a + d.sf4b) - 2) / 8 * 100
    out["MH"] = (((6 - d.sf6a) + d.sf6c) - 2) / 8 * 100
    if require_complete:
        from prevention_health_clustering.measures.items import SF12_ITEM_STEMS

        incomplete = d[list(SF12_ITEM_STEMS)].isna().any(axis=1)
        out[incomplete] = np.nan
    return out


def _z_against(sub: pd.DataFrame, norms: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.column_stack(
        [(sub[s].to_numpy(float) - norms[s][0]) / norms[s][1] for s in SCALES]
    )


def score_us(sub: pd.DataFrame) -> pd.DataFrame:
    """Variant A: US norms + US varimax coefficients. Reproduces sf12*_dv.

    One refinement beyond the construction note: UKHLS floors the published
    scores at 0. The uncensored algorithm goes slightly negative for 32
    person-years on the MCS (published value 0.0, algorithm down to -1.05);
    ``.clip(lower=0)`` reproduces the published scores to max error 0.006 on
    BOTH channels. Returned scores here are uncensored — the floor is a
    property of the published variable, not of the health construct.
    """
    z = _z_against(sub, US_NORMS)
    out = pd.DataFrame(index=sub.index)
    for comp in ("PCS", "MCS"):
        c = np.array([US_COEF[comp][s] for s in SCALES])
        out[f"{comp}_us"] = 50.0 + 10.0 * (z @ c)
    return out


# ---------------------------------------------------------------------------
# Weighted moments and rotations (the Ware procedure, re-runnable on UK data)
# ---------------------------------------------------------------------------

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    mu = weighted_mean(x, w)
    return float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))


def weighted_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = np.array([weighted_mean(X[:, j], w) for j in range(X.shape[1])])
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc / w.sum()
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def varimax(L: np.ndarray, tol: float = 1e-8, max_iter: int = 500) -> np.ndarray:
    """Kaiser-normalised varimax rotation of a loading matrix."""
    p, k = L.shape
    h = np.sqrt((L**2).sum(axis=1, keepdims=True))
    A = L / h
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lam = A @ R
        B = A.T @ (Lam**3 - Lam @ np.diag((Lam**2).sum(axis=0)) / p)
        u, s, vt = np.linalg.svd(B)
        R = u @ vt
        d = float(s.sum())
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            break
    return (A @ R) * h


def promax(L_varimax: np.ndarray, k: int = 4):
    """Promax oblique rotation. Returns (pattern, structure, factor_corr)."""
    P = np.sign(L_varimax) * np.abs(L_varimax) ** k
    Q = np.linalg.lstsq(L_varimax, P, rcond=None)[0]
    d = np.sqrt(np.diag(np.linalg.inv(Q.T @ Q)))
    Q = Q @ np.diag(d)
    pattern = L_varimax @ np.linalg.inv(Q.T)
    phi = Q.T @ Q
    structure = pattern @ phi
    return pattern, structure, phi


def factor_score_coefficients(corr: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Regression-method factor score coefficients: B = R^-1 Lambda."""
    return np.linalg.solve(corr, loadings)


@dataclass
class UKStructure:
    """Everything ``derive_uk_structure`` recovers from the reference sample.

    Arrays are ordered by SCALES; factor columns are oriented so 'phys' loads
    positively on PF and 'ment' positively on MH.
    """

    norms: dict[str, tuple[float, float]]
    corr: np.ndarray                     # (8, 8) weighted correlations
    loadings_varimax: np.ndarray         # (8, 2) columns [phys, ment]
    coef_varimax: np.ndarray             # (8, 2) factor score coefficients
    coef_promax: np.ndarray              # (8, 2) from the promax structure
    phi: float                           # promax inter-factor CORRELATION
    phi_raw: float                       # unnormalised Q'Q off-diagonal (see note)
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def phys_only_weights(self) -> np.ndarray:
        """PF/RP/BP/GH varimax physical loadings, renormalised to sum 1."""
        idx = [SCALES.index(s) for s in PHYS_SCALES]
        w = self.loadings_varimax[idx, 0]
        return w / w.sum()


def derive_uk_structure(ref_sub: pd.DataFrame, weights: pd.Series) -> UKStructure:
    """Re-run the Ware procedure on a (weighted) UK reference sample.

    The canonical reference sample is wave 1, complete SF-12, positive
    self-completion cross-sectional weight (a_indscus_xw).
    """
    X = ref_sub[list(SCALES)].to_numpy(float)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(X) or np.any(~np.isfinite(X)) or np.any(w <= 0):
        raise ValueError("Reference sample must be complete with positive weights.")

    norms = {
        s: (weighted_mean(X[:, j], w), weighted_std(X[:, j], w))
        for j, s in enumerate(SCALES)
    }
    corr = weighted_corr(X, w)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    loadings_unrot = eigvec[:, :2] * np.sqrt(eigval[:2])
    L_vmax = varimax(loadings_unrot)
    pattern, structure, phi_mat = promax(L_vmax)
    # The sandbox printed phi_mat's off-diagonal directly as the inter-factor
    # correlation, but under its normalisation diag(phi_mat) != 1: phi_mat is
    # an unnormalised inner-product matrix. The correlation is the off-diagonal
    # rescaled by the diagonal — identical to the R stats::promax convention
    # (verified numerically). Both are kept: phi is the correlation, phi_raw
    # reproduces the number the construction note reports.
    phi_raw = float(abs(phi_mat[0, 1]))
    phi_corr = float(abs(phi_mat[0, 1]) / np.sqrt(phi_mat[0, 0] * phi_mat[1, 1]))

    # Orient: the factor loading more strongly on PF is physical, positive; the
    # other is mental, positive on MH.
    pf, mh = SCALES.index("PF"), SCALES.index("MH")
    phys_col = int(np.argmax(np.abs(L_vmax[pf, :])))
    ment_col = 1 - phys_col
    for L in (L_vmax, pattern, structure):
        if L[pf, phys_col] < 0:
            L[:, phys_col] *= -1
        if L[mh, ment_col] < 0:
            L[:, ment_col] *= -1

    B_vmax = factor_score_coefficients(corr, L_vmax)
    B_prom = factor_score_coefficients(corr, structure)
    reorder = [phys_col, ment_col]
    return UKStructure(
        norms=norms,
        corr=corr,
        loadings_varimax=L_vmax[:, reorder],
        coef_varimax=B_vmax[:, reorder],
        coef_promax=B_prom[:, reorder],
        phi=phi_corr,
        phi_raw=phi_raw,
        eigenvalues=eigval,
    )


def _t_anchor(raw: np.ndarray, raw_ref: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Anchor to mean 50 / SD 10 in the weighted reference population."""
    return 50.0 + 10.0 * (raw - weighted_mean(raw_ref, w)) / weighted_std(raw_ref, w)


def score_uk_variants(
    sub: pd.DataFrame,
    structure: UKStructure,
    ref_sub: pd.DataFrame,
    ref_weights: pd.Series,
) -> pd.DataFrame:
    """Variants B-D for PCS and MCS, T-anchored in the weighted UK reference."""
    z = _z_against(sub, structure.norms)
    z_ref = _z_against(ref_sub, structure.norms)
    w = np.asarray(ref_weights, dtype=float)
    out = pd.DataFrame(index=sub.index)
    for comp, col in (("PCS", 0), ("MCS", 1)):
        variants = {
            "uk_norm": np.array([US_COEF[comp][s] for s in SCALES]),
            "uk_varimax": structure.coef_varimax[:, col],
            "uk_promax": structure.coef_promax[:, col],
        }
        for tag, c in variants.items():
            out[f"{comp}_{tag}"] = _t_anchor(z @ c, z_ref @ c, w)
    return out


def score_phys_only(
    sub: pd.DataFrame,
    structure: UKStructure,
    anchor_mask: pd.Series | None = None,
) -> pd.Series:
    """Variant E: physical subscales only, loading-weighted, no mental terms.

    Anchored to mean 50 / SD 10 in the pooled complete-case sample (the
    sandbox's convention for this variant), or in ``anchor_mask`` rows.
    """
    w = structure.phys_only_weights
    z = np.column_stack(
        [
            (sub[s].to_numpy(float) - structure.norms[s][0]) / structure.norms[s][1]
            for s in PHYS_SCALES
        ]
    )
    raw = z @ w
    anchor = raw if anchor_mask is None else raw[np.asarray(anchor_mask, bool)]
    anchor = anchor[np.isfinite(anchor)]
    score = 50.0 + 10.0 * (raw - anchor.mean()) / anchor.std()
    return pd.Series(score, index=sub.index, name="PCS_phys_only")


__all__ = [
    "GH_RECODE",
    "PHYS_SCALES",
    "SCALES",
    "UKStructure",
    "US_COEF",
    "US_NORMS",
    "build_subscales",
    "derive_uk_structure",
    "factor_score_coefficients",
    "promax",
    "score_phys_only",
    "score_uk_variants",
    "score_us",
    "varimax",
    "weighted_corr",
    "weighted_mean",
    "weighted_std",
]
