"""Matched held-out comparison: AR(1) against AR(1) plus measurement error.

The problem this fixes. The AR(1) holdout fits were run before the generated
quantities were corrected, so the held-out density Stan stored for them is
the CLASS-ONLY estimand: the recursion restarts at the first held-out row
with the stationary variance and ignores the person's own history. The
state-space holdout fits store a density that conditions on the whole fitted
history through the Kalman filter. Setting one stored number against the
other compares a forecast that uses a person's history with one that does
not, which is how a +0.24 nats-per-person "advantage" for the state-space
model came to be quoted. It is not a valid comparison.

Here every model is scored OFFLINE, through the same code path, on the same
people and rows, with the same thinned posterior draws, on matched
estimands:

  per person, joint    density of both held-out rows given the fitted rows,
                       the second held-out row conditioning on the first.
                       For AR(1) this is the corrected generated-quantities
                       estimand; for the state-space model it is exactly what
                       its stored heldout_lpd holds, so the offline value is
                       checked against the stored one.
  per observation      as in section 9's table: the forecast each held-out
                       row gets from the end of the fitted window (AR(1):
                       last fitted deviation carried forward; state-space:
                       filtered state carried forward), and the class-only
                       forecast, against LOCF and an age-quadratic.

AR(1) scoring reuses oos_assessment.py's functions unchanged -- the machinery
that reproduced Stan's own stored AR(1) densities to a correlation of
0.9999998 -- so the AR(1) rows are directly continuous with section 9.

Only P-FULL has both an AR(1) and a state-space holdout fit on the same
contract, so it is the one matched pair. P-FUNC and the original GRM have
state-space holdout fits only and are scored against the benchmarks; the PCS
and combined GRM have AR(1) holdout fits only.

    PYTHONPATH=src .venv/bin/python clustering/holdout_matched.py [--draws 400]
"""

from __future__ import annotations

import argparse
import io
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))
import oos_assessment as oos  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

HOLD_K, K = oos.HOLD_K, oos.K
DRAWS_DIR = ARTIFACTS_DIR / "holdout-draws"
OUT_DIR = ARTIFACTS_DIR / "descriptives"

# (tag, artifacts subdir, contract, channel, label, kind)
FITS = [
    ("physgrm-ar1-ho", "overnight", "physgrm_lifecycle_20_89_minobs3_v1",
     "theta_phys_full", "P-FULL", "ar1"),
    ("physfull-ssm-ho", "ssm2", "physgrm_lifecycle_20_89_minobs3_v1",
     "theta_phys_full", "P-FULL", "ssm"),
    ("physfunc-ssm-ho", "ssm2", "physfunc_lifecycle_20_89_minobs3_v1",
     "theta_phys_func", "P-FUNC", "ssm"),
    ("grm-ssm-ho", "ssm2", "grm_lifecycle_20_89_minobs3_v1",
     "grm_theta", "GRM original", "ssm"),
    ("combgrm-ar1-ho", "overnight", "combgrm_lifecycle_20_89_minobs3_v1",
     "theta_combined", "GRM combined", "ar1"),
    ("pcs-ar1-ho", "overnight", "pcs_lifecycle_20_89_minobs3_v1",
     "sf12pcs_dv", "PCS", "ar1"),
]


def extract_draws(tag: str, subdir: str, kind: str, n_keep: int) -> pd.DataFrame:
    """Structural parameter draws from the chain CSVs, thinned to n_keep.

    Holdout chains carry ~10^5 person-level columns, so only the structural
    columns are cut out, with cut rather than a CSV parser.
    """
    out = DRAWS_DIR / f"{tag}.csv"
    if out.exists():
        d = pd.read_csv(out)
        if len(d) >= n_keep:
            return d.iloc[:n_keep].reset_index(drop=True)
    want = ([f"theta.{k}" for k in range(1, K + 1)]
            + [f"coef.1.{k}.{p}" for k in range(1, K + 1) for p in (1, 2, 3)]
            + ["sigma.1.1"] + [f"rho.{k}" for k in range(1, K + 1)]
            + (["sigma_meas.1"] if kind == "ssm" else []))
    chains = sorted(p for p in (ARTIFACTS_DIR / subdir / tag / "chains").glob("*.csv"))
    frames = []
    for chain in chains:
        with open(chain) as fh:
            header = next(line for line in fh if not line.startswith("#"))
        names = header.rstrip("\n").split(",")
        fields = ",".join(str(names.index(w) + 1) for w in want)
        cmd = f"grep -v '^#' {shlex.quote(str(chain))} | cut -d, -f{fields}"
        txt = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             check=True).stdout
        frames.append(pd.read_csv(io.StringIO(txt)))
    d = pd.concat(frames, ignore_index=True)
    step = max(1, len(d) // n_keep)
    d = d.iloc[::step].reset_index(drop=True).iloc[:n_keep]
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"  extracted {len(d)} draws from {len(chains)} chains of {tag}",
          flush=True)
    return d


def layout(frame: pd.DataFrame) -> dict:
    """Index arrays shared by every model scored on one contract."""
    held = frame["is_held"].to_numpy()
    person = pd.factorize(frame["pidp"])[0]
    n_person = int(person.max() + 1)
    lens = np.bincount(person, minlength=n_person)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    T = int(lens.max())
    pos = np.arange(len(frame)) - starts[person]
    idx = np.full((n_person, T), -1, dtype=np.int64)
    idx[person, pos] = np.arange(len(frame))
    mask = idx >= 0
    safe = np.where(mask, idx, 0)
    held_idx = np.flatnonzero(held)
    held_person = person[held_idx]
    last_fit_idx = np.flatnonzero(frame["rank_desc"].to_numpy() == HOLD_K)
    anchor = last_fit_idx[held_person]
    age = frame["age"].to_numpy()
    return {"held": held, "person": person, "n_person": n_person, "T": T,
            "mask": mask, "safe": safe, "held_pad": held[safe] & mask,
            "last_fit_pos": lens - HOLD_K - 1,
            "held_idx": held_idx, "held_person": held_person,
            "anchor": anchor,
            "big_gap": np.maximum(age[held_idx] - age[anchor], 1.0)[:, None],
            "window_start": (frame["new_person"].to_numpy()
                             | (frame["rank_desc"].to_numpy() == HOLD_K - 1))}


def score_ar1(frame, L, draws):
    """oos_assessment.py's AR(1) scoring, plus the per-person joint density."""
    z, a, gap = (frame[c].to_numpy() for c in ("z", "a", "gap"))
    D, n_held = len(draws), len(L["held_idx"])
    ll_stan = np.zeros((D, L["n_person"]))
    ll_joint = np.zeros((D, L["n_person"]))
    lp_fc, lp_cls = np.zeros((D, n_held)), np.zeros((D, n_held))
    pred_fc, var_fc = np.zeros((D, n_held)), np.zeros((D, n_held))
    pred_cls = np.zeros((D, n_held))
    person, held = L["person"], L["held"]
    for d in range(D):
        r = draws.iloc[d]
        theta = np.array([r[f"theta.{k}"] for k in range(1, K + 1)])
        coef = np.array([[r[f"coef.1.{k}.{p}"] for p in (1, 2, 3)]
                         for k in range(1, K + 1)])
        sigma = np.repeat(float(r["sigma.1.1"]), K)
        rho = np.array([r[f"rho.{k}"] for k in range(1, K + 1)])
        lp_all, mu = oos.class_loglik_block(z, a, gap, L["window_start"],
                                            coef, sigma, rho)
        lp_cont, _ = oos.class_loglik_block(z, a, gap,
                                            frame["new_person"].to_numpy(),
                                            coef, sigma, rho)
        fit_lp = np.zeros((L["n_person"], K))
        np.add.at(fit_lp, person[~held], lp_all[~held])
        unnorm = np.log(theta)[None, :] + fit_lp
        base = logsumexp(unnorm, axis=1)
        w = np.exp(unnorm - base[:, None])
        h_stan = np.zeros((L["n_person"], K))
        np.add.at(h_stan, person[held], lp_all[held])
        ll_stan[d] = logsumexp(unnorm + h_stan, axis=1) - base
        h_joint = np.zeros((L["n_person"], K))
        np.add.at(h_joint, person[held], lp_cont[held])
        ll_joint[d] = logsumexp(unnorm + h_joint, axis=1) - base

        hi, hp, an, G = L["held_idx"], L["held_person"], L["anchor"], L["big_gap"]
        ww = w[hp]
        stat_sd = sigma / np.sqrt(1 - rho ** 2)
        mu_h = mu[hi]
        lp_cls[d] = logsumexp(np.log(ww) + oos.norm_lpdf(
            z[hi, None], mu_h, stat_sd[None, :]), axis=1)
        pred_cls[d] = (ww * mu_h).sum(axis=1)
        pk = mu_h + rho[None, :] ** G * (z[an, None] - mu[an])
        vk = (sigma[None, :] ** 2 * (1 - rho[None, :] ** (2 * G))
              / (1 - rho[None, :] ** 2))
        m = (ww * pk).sum(axis=1)
        pred_fc[d], var_fc[d] = m, (ww * (vk + pk ** 2)).sum(axis=1) - m ** 2
        lp_fc[d] = logsumexp(np.log(ww) + oos.norm_lpdf(
            z[hi, None], pk, np.sqrt(vk)), axis=1)
    return {"ll_stan": ll_stan, "ll_joint": ll_joint, "lp_fc": lp_fc,
            "lp_cls": lp_cls, "pred_fc": pred_fc, "var_fc": var_fc,
            "pred_cls": pred_cls}


def score_ssm(frame, L, draws):
    """State-space scoring with a Kalman filter, on the same estimands.

    One pass per draw over each person's rows in order: fitted rows build the
    class posterior, the filtered state after the last fitted row gives the
    h-step forecasts, and held-out rows are scored sequentially for the joint
    density (the estimand the stored heldout_lpd holds).
    """
    z, a = frame["z"].to_numpy(), frame["a"].to_numpy()
    gap = frame["gap"].to_numpy()
    design = np.column_stack([np.ones_like(a), a, a ** 2])
    mask, safe, held_pad = L["mask"], L["safe"], L["held_pad"]
    n, T = L["n_person"], L["T"]
    zP, gP = z[safe], gap[safe]
    lastpos = L["last_fit_pos"]
    D, n_held = len(draws), len(L["held_idx"])
    ll_joint = np.zeros((D, n))
    lp_fc, lp_cls = np.zeros((D, n_held)), np.zeros((D, n_held))
    pred_fc, var_fc = np.zeros((D, n_held)), np.zeros((D, n_held))
    pred_cls = np.zeros((D, n_held))
    for d in range(D):
        r = draws.iloc[d]
        theta = np.array([r[f"theta.{k}"] for k in range(1, K + 1)])
        coef = np.array([[r[f"coef.1.{k}.{p}"] for p in (1, 2, 3)]
                         for k in range(1, K + 1)])
        sigma = float(r["sigma.1.1"])
        sm2 = float(r["sigma_meas.1"]) ** 2
        rho = np.array([r[f"rho.{k}"] for k in range(1, K + 1)])
        v_stat = sigma ** 2 / (1 - rho ** 2)
        mu_rows = design @ coef.T
        muP = mu_rows[safe]
        st = np.zeros((n, K))
        Pv = np.tile(v_stat, (n, 1))
        fit_lp, hold_lp = np.zeros((n, K)), np.zeros((n, K))
        st_end, P_end = np.zeros((n, K)), np.tile(v_stat, (n, 1))
        for t in range(T):
            m = mask[:, t][:, None]
            if t > 0:
                g = gP[:, t][:, None]
                rg = rho[None, :] ** g
                a_pred = rg * st
                P_pred = rg ** 2 * Pv + v_stat[None, :] * np.maximum(
                    1 - rho[None, :] ** (2 * g), 1e-9)
            else:
                a_pred, P_pred = st, Pv
            F = P_pred + sm2
            v = zP[:, t][:, None] - muP[:, t, :] - a_pred
            lp = -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F)
            h = held_pad[:, t][:, None]
            fit_lp += np.where(m & ~h, lp, 0.0)
            hold_lp += np.where(m & h, lp, 0.0)
            Kg = P_pred / F
            st = np.where(m, a_pred + Kg * v, st)
            Pv = np.where(m, P_pred - Kg * P_pred, Pv)
            at_end = (lastpos == t)[:, None]
            st_end = np.where(at_end, st, st_end)
            P_end = np.where(at_end, Pv, P_end)
        unnorm = np.log(theta)[None, :] + fit_lp
        base = logsumexp(unnorm, axis=1)
        w = np.exp(unnorm - base[:, None])
        ll_joint[d] = logsumexp(unnorm + hold_lp, axis=1) - base

        hi, hp, G = L["held_idx"], L["held_person"], L["big_gap"]
        ww = w[hp]
        mu_h = mu_rows[hi]
        lp_cls[d] = logsumexp(np.log(ww) + oos.norm_lpdf(
            z[hi, None], mu_h, np.sqrt(v_stat + sm2)[None, :]), axis=1)
        pred_cls[d] = (ww * mu_h).sum(axis=1)
        rg = rho[None, :] ** G
        pk = mu_h + rg * st_end[hp]
        vk = rg ** 2 * P_end[hp] + v_stat[None, :] * (1 - rg ** 2) + sm2
        mm = (ww * pk).sum(axis=1)
        pred_fc[d], var_fc[d] = mm, (ww * (vk + pk ** 2)).sum(axis=1) - mm ** 2
        lp_fc[d] = logsumexp(np.log(ww) + oos.norm_lpdf(
            z[hi, None], pk, np.sqrt(vk)), axis=1)
    return {"ll_joint": ll_joint, "lp_fc": lp_fc, "lp_cls": lp_cls,
            "pred_fc": pred_fc, "var_fc": var_fc, "pred_cls": pred_cls}


def lpd(x):
    return logsumexp(x, axis=0) - np.log(x.shape[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    args = ap.parse_args(argv)

    rows, joint_rows, bench_done = [], [], set()
    for tag, subdir, contract, channel, label, kind in FITS:
        print(f"\n[{tag}] {label}, {kind}", flush=True)
        frame = oos.prepare(contract, channel)
        L = layout(frame)
        draws = extract_draws(tag, subdir, kind, args.draws)
        D = len(draws)
        z = frame["z"].to_numpy()
        y = z[L["held_idx"]]
        s = score_ar1(frame, L, draws) if kind == "ar1" else score_ssm(frame, L, draws)

        stored = pd.read_parquet(ARTIFACTS_DIR / subdir / tag
                                 / "heldout_lpd.parquet")
        pids = pd.factorize(frame["pidp"])[1]
        stored = stored.set_index("pidp")["lpd_heldout"].reindex(pids).to_numpy()
        joint = lpd(s["ll_joint"])
        check_vec = lpd(s["ll_stan"]) if kind == "ar1" else joint
        corr = float(np.corrcoef(check_vec, stored)[0, 1])
        what = ("class-only density (the estimand Stan stored for AR(1))"
                if kind == "ar1" else "joint filtered density (what Stan stored)")
        print(f"  offline {what} vs stored: corr {corr:.6f}, "
              f"mean offline {check_vec.mean():.4f} vs stored {stored.mean():.4f}")
        joint_rows.append({
            "fit": tag, "channel": label, "model": kind, "n_person": L["n_person"],
            "n_held_rows": len(L["held_idx"]), "draws": D,
            "joint_lpd_per_person": float(joint.mean()),
            "joint_lpd_per_obs": float(joint.mean() / HOLD_K),
            "stored_mean_lpd_per_person": float(stored.mean()),
            "stored_estimand": ("class-only" if kind == "ar1" else "joint, filtered"),
            "offline_vs_stored_corr": corr})

        name = {"ar1": ("AR(1), last deviation carried forward",
                        "AR(1), class-only"),
                "ssm": ("AR(1)+meas. error, filtered state carried forward",
                        "AR(1)+meas. error, class-only")}[kind]
        rows.append(oos.metrics(name[0], tag, label, y, s["pred_fc"].mean(0),
                                lpd(s["lp_fc"]),
                                np.sqrt(s["var_fc"].mean(0) + s["pred_fc"].var(0))))
        rows.append(oos.metrics(name[1], tag, label, y, s["pred_cls"].mean(0),
                                lpd(s["lp_cls"]),
                                np.full(len(y), np.sqrt(np.mean(
                                    (y - s["pred_cls"].mean(0)) ** 2)))))

        if contract not in bench_done:
            bench_done.add(contract)
            a_, held = frame["a"].to_numpy(), L["held"]
            X = np.column_stack([np.ones((~held).sum()), a_[~held], a_[~held] ** 2])
            beta, *_ = np.linalg.lstsq(X, z[~held], rcond=None)
            sd_age = (z[~held] - X @ beta).std(ddof=3)
            hi = L["held_idx"]
            Xh = np.column_stack([np.ones(len(hi)), a_[hi], a_[hi] ** 2])
            an = L["anchor"]
            sd_locf = (y - z[an]).std(ddof=1)
            rows.append(oos.metrics("age-quadratic", tag, label, y, Xh @ beta,
                                    oos.norm_lpdf(y, Xh @ beta, sd_age),
                                    np.full(len(y), sd_age)))
            rows.append(oos.metrics("last observation carried forward", tag,
                                    label, y, z[an],
                                    oos.norm_lpdf(y, z[an], sd_locf),
                                    np.full(len(y), sd_locf)))

    tab = pd.DataFrame(rows)
    jt = pd.DataFrame(joint_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_DIR / "holdout_matched.csv", index=False)
    jt.to_csv(OUT_DIR / "holdout_matched_joint.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== per held-out observation ===")
    print(tab[["channel", "predictor", "n_rows", "lpd_per_obs", "rmse", "r2",
               "cover50", "cover90"]].round(4).to_string(index=False))
    print("\n=== per person, joint density of both held-out rows ===")
    print(jt.round(4).to_string(index=False))

    a1 = jt[jt.fit == "physgrm-ar1-ho"].iloc[0]
    s1 = jt[jt.fit == "physfull-ssm-ho"].iloc[0]
    print("\n=== THE MATCHED PAIR: P-FULL, same people, same rows, same draws ===")
    print(f"  joint density per person: AR(1) {a1.joint_lpd_per_person:.4f}, "
          f"AR(1)+meas. error {s1.joint_lpd_per_person:.4f}, difference "
          f"{s1.joint_lpd_per_person - a1.joint_lpd_per_person:+.4f}")
    print(f"  the invalid comparison of stored numbers gave "
          f"{s1.stored_mean_lpd_per_person - a1.stored_mean_lpd_per_person:+.4f}")
    fc = tab[tab.channel == "P-FULL"].set_index("predictor")
    for nm in ("AR(1), last deviation carried forward",
               "AR(1)+meas. error, filtered state carried forward"):
        rr = fc.loc[nm]
        print(f"  {nm:52s} LPD/obs {rr.lpd_per_obs:.4f}  RMSE {rr.rmse:.4f}  "
              f"R2 {rr.r2:.4f}  cover90 {rr.cover90:.3f}")
    print(f"\nwrote {OUT_DIR / 'holdout_matched.csv'} and "
          f"{OUT_DIR / 'holdout_matched_joint.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
