"""The framework note's empirical figures, on the paper's measure, from this repo.

Ports redo_concepts/emp_27_v6_illustrations.R and emp_26_local_rows.R. All
moment-based panels use the identification panel (one row per person-age,
people with at least four observed ages). h is the primary variant, theta is
drawn alongside; the convexity figure adds the deficit index and the SF-6D.

Outputs, paper/figures/:
  fig_fw_rows.png       rows of the covariance matrix leaving the diagonal at
                        five-year base ages, base ages 35 and 70 highlighted
  fig_fw_heatmap.png    the covariance surface in five-year blocks
  fig_fw_local.png      the local triple (V_H, C_Hd, V_d): the denoising solve
                        on four ages four years apart at three life stages
  fig_fw_case2.png      Case 2 on three ages three years apart, the six moments
                        and their fitted decomposition, admissible windows
  fig_fw_convexity.png  in-patient admission against health in equal-width
                        bins of each measure's own units, with the quadratic
  fig_fw_localCHd.png   the local covariance C_Hd(a) from the bundle fit of
                        each row (spike + quadratic, no decay), with a
                        person-bootstrap band, and the spike level by age
artifacts/descriptives/paper_fw_*.csv hold the numbers. BOOT sets the number
of bootstrap replicates (default 200).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, LABEL, load_measure  # noqa: E402
from _paper_moments import (  # noqa: E402
    AGES, MIN_AGE, admissible, cell_names, corr_of, denoising_solve, fit_case, fit_rows_bundle, fit_rows_linear,
    ident_panel, layers, pairs_of, pooled_moments, row_cells, wide,
)
from _style import BLUE, GREEN, INK2, INK3, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402

TWO = [("h", "$h$"), ("theta", r"$\theta$")]
LAYER_COL = {"VH": BLUE, "CHd": ORANGE, "Vd": GREEN, "noise": "#b8b8b8"}
LAYER_LAB = {"VH": "$V_H$", "CHd": "$(s+t)\\,C_{Hd}$", "Vd": "$st\\,V_d$", "noise": "noise"}
STAGES = {"ages 25-47": (25, 35), "ages 45-67": (45, 55), "ages 60-82": (60, 70)}
WIN2 = {"ages 45-61": (45, 55), "ages 60-76": (60, 70), "ages 30-76": (30, 70), "ages 25-41": (25, 35)}
K_ROW, BOOT = 9, int(os.environ.get("BOOT", "200"))


def rows_by_bin(W, lag_max=9, nmin=100):
    A = W.shape[1] - lag_max
    recs = []
    for s in range(A):
        for k in range(lag_max + 1):
            x, y = W[:, s], W[:, s + k]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() >= nmin:
                recs.append({"age": s + MIN_AGE, "lag": k, "cov": np.cov(x[ok], y[ok], ddof=1)[0, 1], "n": ok.sum()})
    r = pd.DataFrame(recs)
    r["start"] = 5 * (r["age"] // 5)
    g = r.groupby(["start", "lag"]).apply(lambda q: np.average(q["cov"], weights=q["n"]), include_groups=False)
    return g.rename("cov").reset_index().assign(later=lambda x: x["start"] + x["lag"])


def surface_blocks(W, nmin=100):
    n_age = W.shape[1]
    recs = []
    for s in range(n_age):
        for t in range(s, n_age):
            x, y = W[:, s], W[:, t]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() >= nmin:
                recs.append({"s": s + MIN_AGE, "t": t + MIN_AGE, "cov": np.cov(x[ok], y[ok], ddof=1)[0, 1], "n": ok.sum()})
    r = pd.DataFrame(recs)
    r["bs"], r["bt"] = 5 * (r["s"] // 5), 5 * (r["t"] // 5)
    g = r.groupby(["bs", "bt"]).apply(lambda q: np.average(q["cov"], weights=q["n"]), include_groups=False)
    return g.rename("cov").reset_index()


def crossing_age(s, x, min_tail=3):
    ok = ~np.isnan(x)
    s, x = np.asarray(s)[ok], np.asarray(x)[ok]
    if len(x) < 8:
        return np.nan
    xs = pd.Series(x).rolling(5, center=True, min_periods=1).mean().to_numpy()
    neg = xs < 0
    tail = np.cumprod(neg[::-1])[::-1]        # 1 where negative to the end
    idx = np.flatnonzero(tail == 1)
    if len(idx) == 0 or idx[0] == 0 or len(x) - idx[0] < min_tail:
        return np.nan
    i = idx[0]
    return s[i - 1] + (0 - xs[i - 1]) * (s[i] - s[i - 1]) / (xs[i] - xs[i - 1])


def main() -> int:
    apply_style()
    d = ident_panel()
    Ws = {v: wide(d, v) for v in ("h", "theta", "fi10")}
    print(f"identification panel: {len(d):,} person-ages, {d['pidp'].nunique():,} people")

    # ---- 1. rows, base ages 35 and 70 highlighted -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (v, lab) in zip(axes, TWO):
        rows = rows_by_bin(Ws[v])
        rows.assign(variant=v).to_csv(DESC / f"paper_fw_rows_{v}.csv", index=False)
        for st, q in rows.groupby("start"):
            hl = {35: (BLUE, "base age 35"), 70: (VERM, "base age 70")}.get(st)
            ax.plot(q["later"], q["cov"], color=hl[0] if hl else "#c8c8c8", lw=1.8 if hl else 0.8,
                    marker="o", ms=2.6 if hl else 1.4, label=hl[1] if hl else None, zorder=3 if hl else 1)
        ax.set_title(f"{lab}: rows leaving the diagonal at five-year base ages", loc="left", fontsize=9.5)
        ax.set_xlabel("later age $t$"); ax.set_ylabel(r"$\mathrm{Cov}(h_s, h_t)$"); ax.grid(True, axis="y")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_fw_rows.png"); plt.close(fig)

    # ---- 2. heatmap -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (v, lab) in zip(axes, TWO):
        sb = surface_blocks(Ws[v])
        sb.assign(variant=v).to_csv(DESC / f"paper_fw_surface_{v}.csv", index=False)
        piv = sb.pivot(index="bs", columns="bt", values="cov")
        blocks = piv.index.to_numpy()
        M = np.full((len(blocks), len(blocks)), np.nan)
        for a, bs in enumerate(blocks):
            for b, bt in enumerate(piv.columns):
                if bt in blocks and bt >= bs:
                    M[a, list(blocks).index(bt)] = piv.loc[bs, bt]
        im = ax.imshow(M, cmap="Blues", origin="upper", vmin=0)
        for a in range(len(blocks)):
            ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), 1, 1, fill=False, ec="black", lw=1.2))
        ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks, fontsize=7.5)
        ax.set_yticks(range(len(blocks))); ax.set_yticklabels(blocks, fontsize=7.5)
        ax.set_xlabel("later age $t$ (five-year blocks)"); ax.set_ylabel("earlier age $s$")
        ax.set_title(f"{lab}: the covariance surface", loc="left", fontsize=9.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG / "fig_fw_heatmap.png"); plt.close(fig)

    # ---- 3. the local triple: denoising solve on four ages four years apart --
    offs = np.array([0, 4, 8, 12]); i4, j4 = pairs_of(offs)
    loc_rows = []
    for v, lab in [("h", "$h$"), ("theta", r"$\theta$"), ("fi10", "deficit index")]:
        for stage, (lo, hi) in STAGES.items():
            m, n = pooled_moments(Ws[v], offs, lo, hi, "balanced")
            b = denoising_solve(m, i4, j4)
            loc_rows.append({"variant": v, "stage": stage, "n": n, "VH": b[0], "CHd": b[1], "Vd": b[2],
                             "corr": corr_of(b), "admissible": admissible(b)})
    loc = pd.DataFrame(loc_rows)
    loc.to_csv(DESC / "paper_fw_local.csv", index=False)
    print("\nlocal triple (denoising solve, four ages four years apart):\n", loc.round(5).to_string(index=False))
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    for r, (v, lab) in enumerate([("h", "$h$"), ("theta", r"$\theta$"), ("fi10", "deficit index")]):
        q = loc[loc["variant"] == v].set_index("stage")
        for c, (key, title) in enumerate([("VH", "$V_H$"), ("CHd", "$C_{Hd}$"), ("Vd", "$V_d$")]):
            ax = axes[r, c]
            ax.bar(range(3), q[key], color=LAYER_COL[key], width=0.65)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(3)); ax.set_xticklabels(list(STAGES), fontsize=8)
            ax.set_title(f"{lab}: {title}", loc="left", fontsize=9.5); ax.grid(True, axis="y")
    fig.text(0.01, -0.01, "Four ages four years apart pooled over base ages within each stage (balanced people); the "
             "deterministic block solved from the six off-diagonal cells alone, so any noise variance profile is allowed.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_fw_local.png"); plt.close(fig)

    # ---- 4. Case 2 on three ages three years apart ------------------------------
    offs3 = np.array([0, 3, 6]); i3, j3 = pairs_of(offs3); names3 = cell_names(i3, j3, 3)
    c2_rows, fits = [], {}
    for v, lab in TWO:
        for win, (lo, hi) in WIN2.items():
            m, n = pooled_moments(Ws[v], offs3, lo, hi, "balanced")
            best, ok = fit_case(m, "case2", i3, j3, rho_grid=np.arange(0.30, 0.975, 0.01))
            fits[(v, win)] = (m, best)
            c2_rows.append({"variant": v, "window": win, "n": n, "rho": best["rho"], "VH": best["b"][0],
                            "CHd": best["b"][1], "Vd": best["b"][2], "s2": best["b"][3], "corr": corr_of(best["b"]),
                            "admissible": admissible(best["b"]), "max_pct_err": 100 * np.abs(best["fit"] / m - 1).max()})
    c2 = pd.DataFrame(c2_rows)
    c2.to_csv(DESC / "paper_fw_case2.csv", index=False)
    print("\nCase 2, three ages three years apart:\n", c2.round(4).to_string(index=False))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), gridspec_kw={"hspace": 0.6, "wspace": 0.25})
    legend_done = False
    for r, (v, lab) in enumerate(TWO):
        for c, win in enumerate(WIN2):
            ax = axes[r, c]; m, best = fits[(v, win)]
            L = layers("case2", best["b"], best["X"], i3, j3)
            if not admissible(best["b"]):
                ax.text(0.5, 0.5, "not admissible\n(deterministic block\nnot a covariance matrix)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, color=INK2)
                ax.set_title(f"{lab}, {win}", loc="left", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                continue
            x = np.arange(len(i3)); bottom = np.zeros(len(i3))
            for key in ("VH", "CHd", "Vd", "noise"):
                ax.bar(x, L[key], bottom=bottom, color=LAYER_COL[key], width=0.72,
                       label=LAYER_LAB[key] if not legend_done else None)
                bottom += L[key]
            ax.hlines(m, x - 0.36, x + 0.36, color="black", lw=1.4)
            ax.set_xticks(x); ax.set_xticklabels(names3, fontsize=7.5)
            ax.set_title(f"{lab}, {win}\nrho = {best['rho']:.2f}, Corr(H,d) = {corr_of(best['b']):+.2f}",
                         loc="left", fontsize=8.5); ax.grid(True, axis="y")
            if not legend_done:
                ax.legend(loc="upper left", fontsize=7.5, ncols=2); legend_done = True
    fig.text(0.01, -0.01, "Three ages three years apart pooled over base ages within each window (balanced people); rho "
             "profiled on 0.30-0.97, the rest by least squares. Black ticks are the observed moments.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_fw_case2.png"); plt.close(fig)

    # ---- 5. convexity: in-patient admission in equal-width bins -----------------
    hm = load_measure()
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", "hosp"])
    sf = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "sf12_measures.parquet", columns=["pidp", "wave", "sf6d_utility"])
    u = hm.merge(cost, on=["pidp", "wave"]).merge(sf, on=["pidp", "wave"], how="left")
    u["inpat"] = np.where(u["hosp"] == 1, 1.0, np.where(u["hosp"] == 2, 0.0, np.nan))
    u = u.dropna(subset=["inpat"])
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    conv_rows = []
    for ax, (v, lab) in zip(axes, [("h", "$h$"), ("theta", r"$\theta$"), ("fi10", "deficit index"), ("sf6d_utility", "SF-6D utility")]):
        q = u.dropna(subset=[v])
        x, y = q[v].to_numpy(float), q["inpat"].to_numpy(float)
        X = np.column_stack([np.ones(len(x)), x, x ** 2])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res ** 2).sum() / (len(x) - 3))
        edges = np.linspace(x.min(), x.max(), 11)
        bn = pd.DataFrame({"x": x, "y": y, "bin": np.clip(np.digitize(x, edges) - 1, 0, 9)}).groupby("bin").agg(
            x=("x", "mean"), p=("y", "mean"), n=("y", "size"))
        grid = np.linspace(np.quantile(x, 0.002), x.max(), 80)
        ax.plot(grid, b[0] + b[1] * grid + b[2] * grid ** 2, color=ORANGE, lw=1.8)
        ax.scatter(bn["x"], bn["p"], s=6 + 60 * bn["n"] / bn["n"].max(), color=BLUE, alpha=0.85, zorder=3)
        ax.set_title(f"{lab}: quadratic t = {b[2] / se[2]:.1f}", loc="left", fontsize=9.5)
        ax.set_xlabel(f"{lab} (equal-width bins; point size = bin count)"); ax.grid(True, axis="y")
        conv_rows.append({"measure": v, "n": len(x), "quad": b[2], "quad_t": b[2] / se[2]})
    axes[0].set_ylabel("P(in-patient admission)")
    fig.tight_layout(); fig.savefig(FIG / "fig_fw_convexity.png"); plt.close(fig)
    pd.DataFrame(conv_rows).to_csv(DESC / "paper_fw_convexity.csv", index=False)
    print("\nconvexity:\n", pd.DataFrame(conv_rows).round(4).to_string(index=False))

    # ---- 6. the local covariance C_Hd(a): bundle fit with a person bootstrap ---
    rng = np.random.default_rng(20260916)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), gridspec_kw={"hspace": 0.4, "wspace": 0.25})
    out_rows = []
    for r, (v, lab) in enumerate(TWO):
        W = Ws[v]
        cells = row_cells(W, K_ROW)
        s_grid = AGES[: cells.shape[0]]
        pb = fit_rows_bundle(cells, K_ROW)
        pl = fit_rows_linear(cells, K_ROW)
        keep = pb["CHd"].notna().to_numpy()
        boot = np.full((BOOT, cells.shape[0]), np.nan); cross_b = np.full(BOOT, np.nan)
        for bi in range(BOOT):
            idx = rng.integers(0, W.shape[0], W.shape[0])
            cb = row_cells(W[idx], K_ROW)
            fb = fit_rows_bundle(cb, K_ROW)
            boot[bi] = fb["CHd"].to_numpy()
            cross_b[bi] = crossing_age(s_grid, boot[bi])
        lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
        cross = crossing_age(s_grid, pb["CHd"].to_numpy())
        cci = np.nanpercentile(cross_b, [2.5, 97.5]) if np.isfinite(cross_b).any() else (np.nan, np.nan)
        print(f"\n{v}: local C_Hd(a) crossing age {cross:.1f} [{cci[0]:.1f}, {cci[1]:.1f}], "
              f"{np.isfinite(cross_b).mean():.0%} of replicates cross; {BOOT} replicates")
        for k, s in enumerate(s_grid):
            out_rows.append({"variant": v, "anchor": s, "CHd_bundle": pb["CHd"][k], "ci_lo": lo[k], "ci_hi": hi[k],
                             "spike": pb["spike"][k], "level": pb["L"][k], "curv": pb["curv"][k],
                             "CHd_linear": pl["CHd"][k], "rho_linear": pl["rho"][k], "crossing": cross,
                             "crossing_lo": cci[0], "crossing_hi": cci[1]})
        ax = axes[r, 0]
        ax.fill_between(s_grid[keep], lo[keep], hi[keep], color=ORANGE, alpha=0.2, lw=0)
        ax.plot(s_grid[keep], pl["CHd"][keep], color=INK3, lw=1.0, ls="--", label="line + decay (row average)")
        ax.plot(s_grid[keep], pb["CHd"][keep], color=ORANGE, lw=2.0, label="bundle fit (spike + quadratic)")
        ax.axhline(0, color=INK2, lw=0.8)
        if np.isfinite(cross):
            ax.axvline(cross, color=ORANGE, lw=0.8, ls=":")
            ax.text(cross + 0.5, ax.get_ylim()[0] * 0.95 if ax.get_ylim()[0] < 0 else 0,
                    f"crossing {cross:.0f} [{cci[0]:.0f}, {cci[1]:.0f}]", fontsize=7.5, color=INK2)
        ax.set_title(f"{lab}: local $C_{{Hd}}(a)$, rows of {K_ROW} years", loc="left", fontsize=9.5)
        ax.set_xlabel("base age $a$"); ax.legend(loc="lower left", fontsize=7.5); ax.grid(True, axis="y")
        ax = axes[r, 1]
        ax.plot(s_grid[keep], pb["spike"][keep], color=INK3, lw=1.8)
        ax.set_ylim(bottom=0); ax.set_title(f"{lab}: the one-period spike by base age", loc="left", fontsize=9.5)
        ax.set_xlabel("base age $a$"); ax.grid(True, axis="y")
    fig.text(0.01, -0.01, f"Each row Cov(h_a, h_(a+k)), k = 0..{K_ROW}, over the people observed at both ages, fitted as "
             "level + k C_Hd(a) + k^2 curvature + one-period spike at k = 0 (no decay column: the rho -> 1 corner).\n"
             f"Band: 2.5-97.5 percentiles over {BOOT} person-bootstrap replicates. Crossing: the earliest base age from "
             "which the five-year-smoothed profile stays negative.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_fw_localCHd.png"); plt.close(fig)
    pd.DataFrame(out_rows).to_csv(DESC / "paper_fw_localCHd.csv", index=False)
    print(f"\nwrote six framework figures to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
