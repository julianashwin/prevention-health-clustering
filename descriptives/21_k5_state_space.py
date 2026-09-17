"""Five classes on P-FUNC under the state-space specification.

Section 10 of the note reads K=4 against K=3 and argues that the fourth class
is a ceiling artefact rather than a fourth trajectory. K=5 is the follow-up
that question invites: if the extra classes are absorbing censored variation
rather than describing paths, a fifth should behave the same way or worse.

This reuses 19_k4_state_space.py wholesale -- same Kalman filter, same
likelihood at the posterior mean, same entropy and BIC -- and adds only the
K=4 to K=5 correspondence. It reports, it does not draw; nothing here is
wired into a figure until the note has somewhere to put it.

The K=5 fit was interrupted by a reboot at 80-96% of sampling and its summary
was rebuilt from the surviving draws by clustering/salvage_partial_fit.py
(2,100 draws, max structural R-hat 1.007). Read it as a converged fit on a
smaller sample, not as a fit of reduced quality.

Outputs: artifacts/descriptives/k5_fit.csv
         artifacts/descriptives/k5_correspondence.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

# 19_k4_state_space starts with a digit, so it cannot be imported by name.
_spec = importlib.util.spec_from_file_location(
    "k4_state_space", HERE / "19_k4_state_space.py")
k4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k4)

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402
from prevention_health_clustering.models.registry import get_model  # noqa: E402
from prevention_health_clustering.runner.fit import build_payload  # noqa: E402

MEASURE = "P-FUNC"
FITS = {3: ("physgrm-func-ssm", ARTIFACTS_DIR / "ssm" / "physfunc-ssm"),
        4: ("physgrm-func-ssm-k4", ARTIFACTS_DIR / "k4" / "physfunc-ssm-k4"),
        5: ("physgrm-func-ssm-k5", ARTIFACTS_DIR / "k5" / "physfunc-ssm-k5")}


def main() -> int:
    out_dir = ARTIFACTS_DIR / "descriptives"
    long = pd.read_csv(k4.CONTRACTS / k4.MEASURES[MEASURE]["contract"] / "long.csv")

    results = {}
    for K, (model, path) in FITS.items():
        spec = get_model(model)
        assert spec.n_classes == K, (model, spec.n_classes)
        pl = build_payload(spec, long)
        prm = k4.load_params(path, K)
        r = k4.analyse(MEASURE, K, pl, prm)
        results[K] = r
        print(f"K={K}: {r['n_person']:,} people, R-hat {prm['rhat']:.4f}, "
              f"LL {r['ll']:,.1f}, {r['n_par']} par, BIC {r['bic']:,.1f}, "
              f"rel. entropy {r['rel_entropy']:.3f}, "
              f"mean max post {r['mean_max_post']:.3f}", flush=True)

    # ---- class structure -------------------------------------------------
    print("\nclass structure:")
    for K, r in results.items():
        p = r["prm"]
        sig, mea = p["sigma"], p["sigma_meas"]
        share = [(sig ** 2 / (1 - x ** 2)) / (sig ** 2 / (1 - x ** 2) + mea ** 2)
                 for x in p["rho"]]
        modal = np.bincount(r["modal"], minlength=K) / r["n_person"]
        print(f"  K={K}")
        for k in range(K):
            print(f"    class {k + 1}: share {p['theta'][k]:5.1%} "
                  f"(modal {modal[k]:5.1%})  alpha {p['coef'][k][0]:+.2f}  "
                  f"slope {p['coef'][k][1]:+.3f}  rho {p['rho'][k]:.3f}  "
                  f"signal {share[k]:5.1%}")

    # ---- fit comparison --------------------------------------------------
    rows = []
    for K, r in results.items():
        rows.append({"measure": MEASURE, "K": K, "n_person": r["n_person"],
                     "ll": r["ll"], "n_par": r["n_par"], "bic": r["bic"],
                     "rel_entropy": r["rel_entropy"],
                     "mean_max_post": r["mean_max_post"],
                     "rhat": r["prm"]["rhat"]})
    fit = pd.DataFrame(rows)
    fit.to_csv(out_dir / "k5_fit.csv", index=False)
    print("\nfit comparison:")
    print(fit.round(3).to_string(index=False))
    n_person = results[5]["n_person"]
    for a, b in ((3, 4), (4, 5)):
        d_ll = results[b]["ll"] - results[a]["ll"]
        d_par = results[b]["n_par"] - results[a]["n_par"]
        print(f"  K={a} -> K={b}: log likelihood {d_ll:+,.1f} for {d_par} "
              f"parameters; BIC {results[b]['bic'] - results[a]['bic']:+,.1f} "
              f"(negative favours K={b}); the penalty each parameter must clear "
              f"is {np.log(n_person) / 2:.1f} log-likelihood units")

    # ---- which K=4 class the fifth is carved from ------------------------
    # Same contract, so person_ids are in the same order at every K.
    assert np.array_equal(results[4]["ids"], results[5]["ids"])
    ct = pd.crosstab(results[4]["modal"] + 1, results[5]["modal"] + 1,
                     rownames=["K=4 class"], colnames=["K=5 class"])
    ct.to_csv(out_dir / "k5_correspondence.csv")
    print("\ncorrespondence, modal class (rows K=4, columns K=5):")
    print(ct.to_string())
    print("\n  row shares:")
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())
    for k5 in ct.columns:
        col = ct[k5]
        src = col.idxmax()
        print(f"  K=5 class {k5} ({col.sum() / n_person:5.1%} of people): "
              f"{col.max() / col.sum():.0%} come from K=4 class {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
