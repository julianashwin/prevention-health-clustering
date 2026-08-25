"""End-to-end held-out scoring: folds, the corrected estimand, leakage.

Four checks:
1. Fold construction reproduces the predecessor's frozen manifest ID from the
   rebuilt roster (rule + seed + roster identity in one hash).
2. A whole-person fold CV on synthetic data: fit on training people, score the
   held-out fifth, and require the held-out class posterior to agree with the
   simulated truth for well-separated classes.
3. The leakage probe: perturbing every future outcome by +1000 changes no
   class posterior by more than 1e-12.
4. The corrected single-normalisation estimand measurably differs from the
   audited-out rule (normalise within draw, then average) — computed here,
   in the test only, to prove the distinction is real rather than notational.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model
from prevention_health_clustering.scoring.folds import (
    FROZEN_MANIFEST_ID,
    build_person_folds,
    fold_split,
)
from prevention_health_clustering.scoring.new_person import (
    LEAKAGE_TOLERANCE,
    GaussianPosterior,
    classify_people,
    leakage_probe,
    posterior_from_fit,
    score_people,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_recovery import TRUE_THETA, TRUTH, simulate  # noqa: E402


def wrong_rule_lpd(posterior, frame, landmark_age):
    """The audited-out estimand: within-draw normalisation, then average.

    Exists only in this test, to demonstrate the corrected rule differs.
    """
    from prevention_health_clustering.scoring.new_person import _row_loglik

    frame = frame.sort_values(["pidp", "age"]).reset_index(drop=True)
    rows_lp = _row_loglik(posterior, frame)
    log_theta = np.log(posterior.theta)  # (D, K)
    out = {}
    for pidp, group in frame.groupby("pidp", sort=True):
        idx = group.index.to_numpy()
        ages = group["age"].to_numpy()
        h, s = idx[ages <= landmark_age], idx[ages > landmark_age]
        if len(h) == 0 or len(s) == 0:
            continue
        hist = rows_lp[h].sum(axis=0)
        # normalise INSIDE each draw (the defect), then average over draws
        within = log_theta + hist
        within -= logsumexp(within, axis=1, keepdims=True)
        weights = np.exp(within).mean(axis=0)  # (K,) expectation of ratios
        fut = rows_lp[s].sum(axis=0)           # (D, K)
        # score with those class weights, draws averaged per class
        lpd = float(
            logsumexp(np.log(weights)[None, :] + fut - np.log(posterior.n_draws))
        )
        out[int(pidp)] = lpd
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--sampling", type=int, default=400)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--landmark-age", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("artifacts/heldout-cv"))
    args = parser.parse_args(argv)
    checks: list[tuple[str, bool, str]] = []

    # -- 1. frozen manifest reproduction on the real roster ------------------
    roster_path = Path(
        "data/processed/contracts/pcs_lifecycle_20_89_minobs3_v1/person_roster.csv"
    )
    if roster_path.exists():
        roster = pd.read_csv(roster_path)
        manifest_id = build_person_folds(roster["pidp"])["fold_manifest_id"].iloc[0]
        checks.append(
            ("frozen fold manifest", manifest_id == FROZEN_MANIFEST_ID,
             f"{manifest_id} vs {FROZEN_MANIFEST_ID}")
        )
    else:
        checks.append(("frozen fold manifest", False, "roster missing"))

    # -- 2. synthetic fold CV ------------------------------------------------
    spec = get_model("pcs-headline")
    frame, true_classes = simulate(spec.channels, args.persons), None
    # simulate() in test_recovery returns only the frame; recover labels by re-simulating classes
    rng = np.random.default_rng(20260824)
    true_labels = pd.Series(
        rng.choice(len(TRUE_THETA), size=args.persons, p=TRUE_THETA),
        index=pd.RangeIndex(args.persons), name="true_class",
    )

    folds = build_person_folds(frame["pidp"].unique(), seed=20260723)
    train_pids, held_pids = fold_split(folds, fold=1)
    train = frame[frame["pidp"].isin(train_pids)]
    held = frame[frame["pidp"].isin(held_pids)]
    print(f"fold 1: train {train['pidp'].nunique():,} people, "
          f"held out {held['pidp'].nunique():,}")

    payload = build_payload(spec, train)
    fit = fit_model(
        spec, payload, output_dir=args.output,
        chains=args.chains, iter_warmup=args.warmup,
        iter_sampling=args.sampling, threads_per_chain=args.threads_per_chain,
    )
    rhat = float(fit.summary()["R_hat"].dropna().max())
    checks.append(("training fit converged", rhat < 1.05, f"max R-hat {rhat:.4f}"))

    posterior = posterior_from_fit(fit, spec, payload.channel_moments)

    # same-window scores for every held-out person
    scores = score_people(posterior, held)
    checks.append(
        ("scored all held-out people",
         len(scores) == held["pidp"].nunique(), f"{len(scores):,} scored")
    )
    finite = np.isfinite(scores["log_predictive_density"]).all()
    checks.append(("held-out LPD finite", bool(finite), ""))

    # The same-window estimand's DEFINING property: its class posterior is the
    # training prior, because scored outcomes must not choose a class. Pin it.
    theta_hat = posterior.theta.mean(axis=0)
    prior_dev = max(
        float((scores[f"prob_class{k+1}"] - theta_hat[k]).abs().max())
        for k in range(3)
    )
    checks.append(("unconditional posterior == prior", prior_dev < 1e-9,
                   f"max dev {prior_dev:.2e}"))

    # Classification (full-window conditioning - description, not prediction)
    # must match the ORACLE, the Bayes classifier under the true parameters.
    t_ = TRUTH[spec.channels[0]]
    oracle_coef = np.zeros((1, 1, 3, 3))
    for k in range(3):
        oracle_coef[0, 0, k] = [t_["alpha"][k], t_["beta"][k], t_["gamma"][k]]
    oracle = GaussianPosterior(
        theta=np.array([TRUE_THETA]), coef=oracle_coef,
        sigma=np.array([[t_["sigma"]]]), channels=spec.channels[:1],
        moments={spec.channels[0]: {"mean": t_["mean"], "sd": t_["sd"]}},
    )
    truth_held_frame = held
    oracle_cls = classify_people(oracle, truth_held_frame)
    fitted_cls = classify_people(posterior, truth_held_frame)
    truth_lab = true_labels.loc[oracle_cls["pidp"].to_numpy()].to_numpy()
    oracle_acc = float((oracle_cls["modal_class"].to_numpy() - 1 == truth_lab).mean())
    fitted_acc = float((fitted_cls["modal_class"].to_numpy() - 1 == truth_lab).mean())
    checks.append(
        ("classification within 0.02 of oracle",
         fitted_acc >= oracle_acc - 0.02,
         f"fitted {fitted_acc:.3f} vs oracle {oracle_acc:.3f}")
    )

    # -- 3. leakage probe ----------------------------------------------------
    probe_people = held["pidp"].unique()[:150]
    probe_frame = held[held["pidp"].isin(probe_people)]
    eligible = probe_frame.groupby("pidp")["age"].agg(["min", "max"])
    ok_ids = eligible[(eligible["min"] <= args.landmark_age)
                      & (eligible["max"] > args.landmark_age)].index
    probe_frame = probe_frame[probe_frame["pidp"].isin(ok_ids)]
    if len(ok_ids) == 0:
        checks.append(("leakage probe", False, "no eligible probe people"))
    else:
        delta = leakage_probe(
            posterior, probe_frame, landmark_age=args.landmark_age
        )
        checks.append(
            (f"leakage probe <= {LEAKAGE_TOLERANCE:g}",
             delta <= LEAKAGE_TOLERANCE, f"max delta {delta:.2e} "
             f"({len(ok_ids)} people)")
        )

    # -- 4. corrected vs audited-out rule ------------------------------------
    if len(ok_ids) > 0:
        corrected = score_people(
            posterior, probe_frame, landmark_age=args.landmark_age
        ).set_index("pidp")["log_predictive_density"]
        wrong = wrong_rule_lpd(posterior, probe_frame, args.landmark_age)
        joined = pd.DataFrame(
            {"corrected": corrected, "wrong": pd.Series(wrong)}
        ).dropna()
        gap = float((joined["corrected"] - joined["wrong"]).abs().max())
        checks.append(
            ("corrected differs from within-draw rule", gap > 1e-6,
             f"max |gap| {gap:.4f} over {len(joined)} people")
        )

    print("\n=== checks ===")
    all_ok = True
    for name, ok, note in checks:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:42s} {note}")
    print(f"\n{'HELD-OUT SCORING OK' if all_ok else 'HELD-OUT SCORING FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
