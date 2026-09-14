"""The multidimensional model under measurement error, and with mortality.

Three multidimensional fits (physical GRM, mental GRM, chronic count; K=3)
were run after section 12 was written:

  ssm             AR(1) latent state plus one-period measurement error on the
                  two Gaussian channels
  ssm-mort        the same, with the discrete-time mortality hazard switched on
  baseline-mort   mortality on the conditionally independent baseline

BOTH STATE-SPACE FITS HAVE A MODE SPLIT. The ordered constraint is on the
physical-GRM intercept, but classes 2 and 3 are separated mainly on the
chronic channel. A chain that starts with those two labels the wrong way
round cannot swap them without crossing the constraint, so it jams against
the boundary with the two intercepts pressed together. That mode sits about
480 log-posterior units below the other, so it is decisively inferior rather
than a competing explanation. Pooling chains across the two modes produced
R-hat values up to 54; relabelling alone does not repair it, because the
trapped chain is at different parameter values, not a permutation of the
same ones.

This script therefore identifies each chain's mode from its intercept gap and
log posterior, keeps the chains in the better mode, and recomputes posterior
summaries and split R-hat on those alone. baseline-mort converged without any
split and uses all four chains. The kept-chain summaries are an honest
description of the better mode, but they rest on three and two chains
respectively, and the defect they work around is a labelling constraint that
should be moved onto the chronic channel before these fits are relied on.

Mortality enters as a per-wave logit hazard with a quadratic in age, so the
class intercept is the log-odds of dying in a wave at age 55.

Outputs: docs/measurement/figures/fig_multidim_mortality.png
         artifacts/descriptives/multidim_mode_chains.csv
         artifacts/descriptives/multidim_mortality.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import CLUSTER, INK2, SURFACE, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, ROOT_DIR

K = 3
FIG = ROOT_DIR / "docs" / "measurement" / "figures" / "fig_multidim_mortality.png"
FITS = {
    "ssm": ARTIFACTS_DIR / "multidim-ssm" / "ssm",
    "ssm-mort": ARTIFACTS_DIR / "multidim-ssm" / "ssm-mort",
    "baseline-mort": ARTIFACTS_DIR / "multidim-ssm" / "baseline-mort",
    "baseline": ARTIFACTS_DIR / "multidim" / "baseline",
    "ar1": ARTIFACTS_DIR / "multidim" / "ar1",
}
GAP_FLOOR = 0.05       # intercept gap below which a chain is jammed on the constraint


def split_rhat(x: np.ndarray) -> float:
    c, d = x.shape
    h = d // 2
    p = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
    m = p.mean(axis=1)
    w = p.var(axis=1, ddof=1).mean()
    b = h * m.var(ddof=1)
    return float(np.sqrt(((h - 1) / h * w + b / h) / w)) if w > 0 else np.inf


def load_chains(path: Path) -> list[pd.DataFrame]:
    files = sorted((path / "chains").glob("*.csv"))
    return [pd.read_csv(f, comment="#") for f in files]


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    apply_style()
    out_dir = ARTIFACTS_DIR / "descriptives"
    mode_rows, par_rows, kept_draws = [], [], {}
    for tag, path in FITS.items():
        chains = load_chains(path)
        info = []
        for i, c in enumerate(chains, start=1):
            gap = float((c["coef.1.3.1"] - c["coef.1.2.1"]).mean())
            info.append({"fit": tag, "chain": i, "draws": len(c),
                         "lp_mean": float(c["lp__"].mean()),
                         "intercept_gap_k3_k2": gap,
                         "mode": "separated" if gap > GAP_FLOOR else "jammed"})
        best_lp = max(r["lp_mean"] for r in info if r["mode"] == "separated")
        for r in info:
            r["lp_below_best"] = best_lp - r["lp_mean"]
            r["kept"] = r["mode"] == "separated"
        mode_rows += info
        keep = [c for c, r in zip(chains, info) if r["kept"]]
        kept_draws[tag] = keep
        print(f"{tag:14s} chains kept {sum(r['kept'] for r in info)}/{len(info)}  "
              + "  ".join(f"[{r['chain']}: gap {r['intercept_gap_k3_k2']:.3f}, "
                          f"lp -{r['lp_below_best']:.0f}]" for r in info))

        cols = [c for c in keep[0].columns
                if c.startswith(("theta.", "rho.", "sigma.", "sigma_meas.",
                                 "coef.", "coef_chronic.", "coef_mort.",
                                 "phi_chronic"))]
        n = min(len(c) for c in keep)
        for col in cols:
            arr = np.stack([c[col].to_numpy()[:n] for c in keep])
            par_rows.append({"fit": tag, "param": col, "mean": arr.mean(),
                             "sd": arr.std(), "q05": np.quantile(arr, .05),
                             "q95": np.quantile(arr, .95),
                             "rhat_kept": split_rhat(arr) if len(keep) > 1 else np.nan,
                             "chains_kept": len(keep)})

    modes = pd.DataFrame(mode_rows)
    pars = pd.DataFrame(par_rows)
    modes.to_csv(out_dir / "multidim_mode_chains.csv", index=False)
    pars.to_csv(out_dir / "multidim_mortality.csv", index=False)

    P = pars.set_index(["fit", "param"])
    g = lambda fit, name: float(P.loc[(fit, name), "mean"]) \
        if (fit, name) in P.index else np.nan
    print("\nworst split R-hat on kept chains, structural parameters:")
    for tag in FITS:
        sub = pars[(pars.fit == tag)]
        worst = sub.loc[sub["rhat_kept"].idxmax()]
        print(f"  {tag:14s} {worst['rhat_kept']:.4f} ({worst['param']}), "
              f"{int(worst['chains_kept'])} chains")

    print("\nclass structure (kept chains):")
    for tag in FITS:
        th = [g(tag, f"theta.{k}") for k in range(1, K + 1)]
        rho = [g(tag, f"rho.{k}") for k in range(1, K + 1)]
        a1 = [g(tag, f"coef.1.{k}.1") for k in range(1, K + 1)]
        ch = [g(tag, f"coef_chronic.{k}.1") for k in range(1, K + 1)]
        sm = [g(tag, f"sigma_meas.{c}") for c in (1, 2)]
        sg = [g(tag, f"sigma.1.{c}") for c in (1, 2)]
        print(f"  {tag:14s} theta {np.round(th, 3).tolist()}  "
              f"phys alpha {np.round(a1, 3).tolist()}  chronic {np.round(ch, 2).tolist()}")
        print(f"  {'':14s} rho {np.round(rho, 3).tolist()}  sigma {np.round(sg, 3).tolist()}  "
              f"sigma_meas {np.round(sm, 3).tolist()}")

    ages = np.array([55, 70, 85])
    print("\nper-wave death probability by class (kept chains):")
    haz = {}
    for tag in ("baseline-mort", "ssm-mort"):
        for k in range(1, K + 1):
            b = [g(tag, f"coef_mort.{k}.{p}") for p in (1, 2, 3)]
            a = (ages - 55) / 10.0
            haz[(tag, k)] = inv_logit(b[0] + b[1] * a + b[2] * a ** 2)
        print(f"  {tag}:")
        for k in range(1, K + 1):
            print(f"    class {k}: logit intercept {g(tag, f'coef_mort.{k}.1'):+.3f}, "
                  f"age slope/decade {g(tag, f'coef_mort.{k}.2'):+.3f}; "
                  + ", ".join(f"age {x}: {h:.4f}" for x, h in zip(ages, haz[(tag, k)])))
        or13 = np.exp(g(tag, "coef_mort.1.1") - g(tag, "coef_mort.3.1"))
        print(f"    odds of dying at 55, class 1 against class 3: {or13:.1f}x")

    # ---- figure: class-specific mortality hazard by age -------------------
    # Deaths before about 40 are too few to pin the age quadratic, so the
    # hazard is drawn only where it is identified.
    A = np.linspace(40, 90, 101)
    a = (A - 55) / 10.0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharey=True)
    titles = {"baseline-mort": "(a) Conditionally independent baseline",
              "ssm-mort": "(b) AR(1) state plus measurement error"}
    for ax, tag in zip(axes, ("baseline-mort", "ssm-mort")):
        for k in range(1, K + 1):
            b = [g(tag, f"coef_mort.{k}.{p}") for p in (1, 2, 3)]
            h = inv_logit(b[0] + b[1] * a + b[2] * a ** 2)
            share = g(tag, f"theta.{k}")
            ax.plot(A, h * 100, color=CLUSTER[k - 1], lw=2,
                    label=f"class {k}: {share:.0%}")
        ax.set_yscale("log")
        ax.set_xlabel("age")
        ax.set_title(titles[tag], fontsize=10)
        ax.grid(True, which="major", axis="y")
        ax.set_xlim(40, 90)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel("probability of dying in a wave, % (log scale)")
    kept = modes[modes.fit == "ssm-mort"]["kept"].sum()
    fig.suptitle("Mortality hazard by latent class in the multidimensional model",
                 fontweight="bold", y=1.0)
    fig.text(0.01, -0.02,
             "Per-wave logit hazard with a quadratic in age, by class, shown from 40 where deaths are frequent enough to identify it;\nclass 1 is worst health. Physical GRM, mental GRM and "
             "chronic count jointly define the classes. Panel (b) summarises only the chains in the better of two modes "
             f"({kept} of 4);\nthe other mode pins two class intercepts against the ordering constraint and is ~480 log-posterior units worse.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    print(f"\nwrote {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
