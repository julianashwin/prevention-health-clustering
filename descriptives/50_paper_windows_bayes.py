"""The windows figure on the Bayesian AR(1) + "spike" fits: types over limited age windows.

The K = 3 AR(1) + "spike" mixture on theta refitted on each of the ten age windows of the
K-means windows figure (26_paper_windows.py): six sliding twenty-year windows and four
windows expanding backwards from 90 (plus the lifecycle fit itself as the 20-90 window).
Each window has its own contract (data/processed/contracts/health_window_{lo}_{hi}_v1,
people observed at least three times inside it) and its own fit in
artifacts/health-windows/theta-ssm-{lo}-{hi} (clustering/runs/windows_queue.py). Classes
are ordered worst to best by the class path's mean over the window's own ages.

Panels as in the K-means figure: (a) sliding and (b) backward windows' class paths over
their own age range, the lifecycle classes in grey behind; (c) agreement of each window's
modal class with the lifecycle modal class on the people both contain (adjusted Rand index);
(d), (e) the worst class's posterior share of the person-waves observed at each age, window
by window; (f) the worst class's share of each window's people. Panel (c) also carries a
posterior-based agreement: over pairs of people (a random 3,000 of those in both fits), the
correlation between the probability that the window fit puts the pair in the same class and
the probability that the lifecycle fit does, sum_k p_ik p_jk under each. It needs no label
correspondence and is 1 whenever the two posteriors coincide, however diffuse, so it
separates a different grouping from a blurred one.

Outputs: paper/figures/fig_windows_bayes.png, artifacts/descriptives/paper_windows_bayes.csv.

    PYTHONPATH=src .venv/bin/python descriptives/50_paper_windows_bayes.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import CONTRACT, DESC, FIG, K, MAX_AGE, MIN_AGE, bayes_fit  # noqa: E402
from _style import CLUSTER, INK2, INK3, apply_style, share_ylim  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(70, 90), (60, 90), (50, 90), (40, 90), (30, 90), (20, 90)]
WIN = ARTIFACTS_DIR / "health-windows"
V = "theta"
N_PAIR_SAMPLE = 3000


def coassignment_corr(post_a: pd.DataFrame, post_b: pd.DataFrame, n: int = N_PAIR_SAMPLE, seed: int = 0) -> float:
    """Correlation over pairs of the same-class probabilities under two posteriors (people in both)."""
    both = post_a.index.intersection(post_b.index)
    rng = np.random.default_rng(seed)
    pick = rng.choice(both, size=min(n, len(both)), replace=False)
    A = post_a.loc[pick].to_numpy(); B = post_b.loc[pick].to_numpy()
    iu = np.triu_indices(len(pick), k=1)
    return float(np.corrcoef((A @ A.T)[iu], (B @ B.T)[iu])[0, 1])


def window_fit(lo: int, hi: int) -> dict:
    """bayes_fit on the window's contract, classes re-ordered worst to best by the mean path over the window."""
    if (lo, hi) == (MIN_AGE, MAX_AGE):
        r = bayes_fit(ARTIFACTS_DIR / "health-ssm" / f"health-{V}-ssm", V, K); contract = CONTRACT
    else:
        contract = PROCESSED_DATA_DIR / "contracts" / f"health_window_{lo}_{hi}_v1"
        r = bayes_fit(WIN / f"{V}-ssm-{lo}-{hi}", V, K, contract=contract)
    ages = np.arange(lo, hi + 1); A = (ages - 55) / 10
    paths = np.vstack([r["mom"]["mean"] + r["mom"]["sd"] * (c[0] + c[1] * A + c[2] * A ** 2) for c in r["coef"]])
    order = np.argsort(paths.mean(axis=1))                      # worst (lowest theta) first
    cols = [f"class{k + 1}" for k in range(K)]
    post = r["post"][cols].to_numpy()[:, order]
    comp = r["comp"][cols].to_numpy()[:, order]
    summ = json.load(open((ARTIFACTS_DIR / "health-ssm" / f"health-{V}-ssm" if (lo, hi) == (MIN_AGE, MAX_AGE) else WIN / f"{V}-ssm-{lo}-{hi}") / "run_summary.json"))
    return {"lo": lo, "hi": hi, "ages": ages, "paths": paths[order], "share": r["theta"][order], "rho": r["rho"][order],
            "post": pd.DataFrame(post, columns=cols, index=r["post"]["pidp"].to_numpy()),
            "comp": pd.DataFrame(comp, columns=cols, index=r["comp"].index), "rhat": summ["max_structural_rhat"],
            "n": len(r["post"]), "wall": summ.get("wall_hours")}


def main() -> int:
    apply_style()
    full = window_fit(MIN_AGE, MAX_AGE)
    full_modal = full["post"].to_numpy().argmax(axis=1); full_modal = pd.Series(full_modal, index=full["post"].index)
    fits = {w: (full if w == (MIN_AGE, MAX_AGE) else window_fit(*w)) for w in dict.fromkeys(SLIDING + BACKWARD)}
    for w, f in fits.items():
        print(f"{w[0]}-{w[1]}: {f['n']:,} people, rhat {f['rhat']:.3f}, shares {np.round(f['share'], 3)}, rho {np.round(f['rho'], 2)}, wall {f['wall']}")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.9), gridspec_kw={"wspace": 0.25, "hspace": 0.45}, squeeze=False)
    fig.suptitle("$\\theta$, K = 3 AR(1) + \"spike\" classes refitted on age windows", x=0.125, ha="left", fontsize=11, fontweight="bold", y=0.96)
    cmap = plt.get_cmap("viridis")
    rows = []
    for col, (wins, name) in enumerate([(SLIDING, "sliding twenty-year windows"), (BACKWARD, "windows expanding backwards from 90")]):
        ax = axes[0, col]
        for k in range(K):
            ax.plot(full["ages"], full["paths"][k], color=INK3, lw=3.0, alpha=0.35, zorder=1)
        for lo, hi in wins:
            if (lo, hi) == (MIN_AGE, MAX_AGE):
                continue
            f = fits[(lo, hi)]
            for k in range(K):
                ax.plot(f["ages"], f["paths"][k], color=CLUSTER[k], lw=1.0 + 2.5 * f["share"][k], zorder=2)
                ax.plot([lo], [f["paths"][k][0]], color=CLUSTER[k], marker="|", ms=7)
            modal = pd.Series(f["post"].to_numpy().argmax(axis=1), index=f["post"].index)
            both = modal.index.intersection(full_modal.index)
            rows.append({"kind": "sliding" if col == 0 else "backward", "lo": lo, "hi": hi, "people": f["n"], "shared": len(both),
                         "ari_vs_full": adjusted_rand_score(modal.loc[both], full_modal.loc[both]),
                         "coassign_corr": coassignment_corr(f["post"], full["post"]),
                         "same_vs_full": float((modal.loc[both] == full_modal.loc[both]).mean()),
                         "worst_share": float(f["share"][0]), "worst_modal_share": float((modal == 0).mean()), "rhat": f["rhat"],
                         "rho_worst": f["rho"][0], "rho_mid": f["rho"][1], "rho_best": f["rho"][2]})
        ax.set_title(f"({'ab'[col]}) {'sliding twenty-year windows' if col == 0 else 'windows expanding backwards from 90'}", loc="left", fontsize=9.5)
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age")
    r = pd.DataFrame(rows)
    ax = axes[0, 2]
    for kind, mk, c in [("sliding", "o", CLUSTER[1]), ("backward", "s", CLUSTER[0])]:
        q = r[r["kind"] == kind]
        x = q["lo"] + (q["hi"] - q["lo"]) / 2 if kind == "sliding" else q["lo"]
        ax.plot(x, q["ari_vs_full"], marker=mk, color=c, lw=1.4, label=f"{kind}: ARI of the modal classes" + (" (x = midpoint)" if kind == "sliding" else " (x = start age)"))
        ax.plot(x, q["coassign_corr"], marker=mk, color=c, lw=1.4, ls="--", mfc="white", label=f"{kind}: posterior co-assignment correlation")
    ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    ax.set_title("(c) agreement with the lifecycle classes", loc="left", fontsize=9.5); ax.legend(loc="lower left", fontsize=6.8); ax.set_xlabel("age")
    full_c = full["comp"]["class1"].rolling(5, center=True, min_periods=1).mean()
    for col, wins in enumerate([SLIDING, BACKWARD]):
        ax = axes[1, col]
        ax.plot(full_c.index, full_c, color=INK3, lw=3.0, alpha=0.35, zorder=1, label="lifecycle typology")
        wl = [w for w in wins if w != (MIN_AGE, MAX_AGE)]
        for i_w, (lo, hi) in enumerate(wl):
            q = fits[(lo, hi)]["comp"]["class1"].rolling(5, center=True, min_periods=1).mean()
            ax.plot(q.index, q, color=cmap(0.85 * i_w / max(len(wl) - 1, 1)), lw=1.4, zorder=2, label=f"{lo}-{hi}")
        ax.set_title(f"({'de'[col]}) worst class's share by age, {'sliding' if col == 0 else 'backward'}", loc="left", fontsize=9.5)
        ax.set_ylim(0, 0.6); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_xlabel("age"); ax.legend(fontsize=6.8, ncols=2, loc="upper left")
    ax = axes[1, 2]
    for kind, mk, c in [("sliding", "o", CLUSTER[1]), ("backward", "s", CLUSTER[0])]:
        q = r[r["kind"] == kind]
        x = q["lo"] + (q["hi"] - q["lo"]) / 2 if kind == "sliding" else q["lo"]
        ax.plot(x, q["worst_share"], marker=mk, color=c, lw=1.4, label=f"{kind} windows" + (" (x = midpoint)" if kind == "sliding" else " (x = start age)"))
    ax.axhline(full["share"][0], color=INK3, lw=1.4, ls="--", label="lifecycle typology")
    ax.set_ylim(0, 0.6); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    ax.set_title("(f) worst class's share per window", loc="left", fontsize=9.5); ax.legend(loc="upper left", fontsize=7.5); ax.set_xlabel("age")
    share_ylim([axes[0, 0], axes[0, 1]])
    footer = ("Each window refits the K = 3 AR(1) + \"spike\" mixture on theta on the people observed at least three times inside it (its own contract and fit). Coloured lines: the window's class paths at the posterior "
              "mean over its own age range (tick at the start; line width proportional to the class share); grey: the lifecycle 20-90 fit. Panel (c), solid: adjusted Rand index between a window's modal class and the "
              "lifecycle modal class, on the people both contain; dashed: over pairs of people (a random 3,000), the correlation between the probability that the window fit puts the pair in the same class "
              "and the probability that the lifecycle fit does, which is 1 whenever the two posteriors coincide. Panels (d), (e): the worst class's posterior share of the person-waves observed at each age (five-year rolling mean), one line per window, "
              "against the lifecycle fit in grey; (f) the worst class's share of each window's people. Classes ordered worst to best by the class path's mean over the window.")
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(footer, 200)), fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_windows_bayes.png", bbox_inches="tight")
    r.to_csv(DESC / "paper_windows_bayes.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(f"wrote fig_windows_bayes.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
