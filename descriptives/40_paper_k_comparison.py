"""K = 3, 4 and 5 compared: plug-in log-likelihood, BIC and classification certainty.

For each fitted mixture on the health contract (independent residuals and AR(1)
plus one-period noise, on h and theta), the person-level log-likelihood at the
posterior mean is recomputed with _paper_common.bayes_fit's machinery, summed,
and turned into a BIC with N = persons and p = 3K quadratic coefficients + (K-1)
shares + 1 residual scale (+ K persistences + 1 noise scale under AR(1)). This is
a plug-in BIC, not a marginal likelihood; it ranks K the way a frequentist
mixture comparison would. Fits not yet on disk are skipped.

Outputs: paper/tables/tab_k_comparison.tex, artifacts/descriptives/paper_k_comparison.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import CONTRACT, DESC, TAB, kalman_person_lp, write_table  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

FITS = {  # (variant, spec, K): directory
    **{("h", "base", 3): ARTIFACTS_DIR / "health-base" / "health-h-base",
       ("theta", "base", 3): ARTIFACTS_DIR / "health-base" / "health-theta-base",
       ("h", "ssm", 3): ARTIFACTS_DIR / "health-ssm" / "health-h-ssm",
       ("theta", "ssm", 3): ARTIFACTS_DIR / "health-ssm" / "health-theta-ssm"},
    **{(v, spec, K): ARTIFACTS_DIR / "health-k45" / f"health-{v}-{spec}-k{K}"
       for K in (4, 5) for spec in ("base", "ssm") for v in ("h", "theta")},
}
SPEC_LAB = {"base": "independent residuals", "ssm": "AR(1) plus ``spike''"}


def loglik(fit_dir: Path, v: str, K: int):
    d = json.load(open(fit_dir / "stan_data.json")); p = json.load(open(fit_dir / "run_summary.json"))["params"]
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, K + 1)])
    sigma = p["sigma[1,1]"]["mean"]
    X, Y = np.asarray(d["X"]), np.asarray(d["y"][0]); fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    if d["ar_mode"] == 2:
        rho = np.array([p[f"rho[{k}]"]["mean"] for k in range(1, K + 1)]); sm = p["sigma_meas[1]"]["mean"]
        per = kalman_person_lp(Y, X, coef, sigma, sm, rho, np.asarray(d["age_gap"], float), fs, fe)
        n_par = 3 * K + (K - 1) + 1 + K + 1
    else:
        mu = X @ coef.T
        lp = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((Y[:, None] - mu) / sigma) ** 2
        per = np.zeros((len(fs), K)); np.add.at(per, np.repeat(np.arange(len(fs)), fe - fs + 1), lp)
        n_par = 3 * K + (K - 1) + 1
    un = np.log(theta)[None, :] + per
    mx = un.max(axis=1); ll = mx + np.log(np.exp(un - mx[:, None]).sum(axis=1))
    w = np.exp(un - mx[:, None]); w /= w.sum(axis=1, keepdims=True)
    return float(ll.sum()), n_par, len(fs), float(w.max(axis=1).mean()), theta


def main() -> int:
    rows = []
    for (v, spec, K), fit_dir in FITS.items():
        if not (fit_dir / "run_summary.json").exists():
            print(f"skip {fit_dir.name}: not fitted yet"); continue
        ll, n_par, N, mmp, theta = loglik(fit_dir, v, K)
        rows.append({"variant": v, "spec": spec, "K": K, "loglik": ll, "n_par": n_par, "N": N,
                     "BIC": -2 * ll + n_par * np.log(N), "mean_max_posterior": mmp, "min_share": theta.min()})
    t = pd.DataFrame(rows).sort_values(["variant", "spec", "K"])
    t["dBIC_vs_K3"] = t["BIC"] - t.groupby(["variant", "spec"])["BIC"].transform(lambda x: x.iloc[0])
    t.to_csv(DESC / "paper_k_comparison.csv", index=False)
    print(t.round(3).to_string(index=False))
    lab = {"h": "$h$", "theta": r"$\theta$"}
    out = [[lab[r.variant], SPEC_LAB[r.spec], str(r.K), f"{r.loglik:,.0f}", f"{r.BIC:,.0f}", f"{r.dBIC_vs_K3:+,.0f}",
            f"{r.mean_max_posterior:.2f}", f"{r.min_share:.2f}"] for r in t.itertuples()]
    write_table(TAB / "tab_k_comparison.tex", ["variant", "specification", "$K$", "log-likelihood", "BIC",
                                               "$\\Delta$BIC vs $K=3$", "mean max posterior", "smallest share"], out)
    print(f"wrote tab_k_comparison.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
