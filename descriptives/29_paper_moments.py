"""Exhibit 2 and the identification moments behind it, on the paper's measure.

Four ages two years apart, pooled over base ages within four bands, on the
identification panel (people with at least four observed ages). Cases 1 to 4
on balanced and on pairwise moments, rho profiled, at the best fit and
the best admissible fit, for theta, h and the deficit index.

Outputs, paper/figures/:
  fig_exhibit2_main.png       Cases 1 and 4 on h and theta, balanced moments (main text)
  fig_exhibit2_corr.png       Corr(H, d) by band under each case, three
                              variants x balanced / pairwise
  fig_exhibit2_{h,theta,fi10}.png
                              the barcharts: observed moments and fitted layers
                              by cell, rows Case 2 / 3 / 4, columns the bands,
                              balanced moments; the same on pairwise moments as
                              fig_exhibit2_{h,theta,fi10}_pairwise.png
artifacts/descriptives/:
  paper_ident_corr_band.csv, paper_ident_decomposition.csv, paper_ident_sample.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, LABEL, VARIANTS  # noqa: E402
from _paper_moments import (  # noqa: E402
    BANDS, OFFS4, admissible, cell_names, corr_of, fit_case, ident_panel, layers, pairs_of, pooled_moments,
    wide,
)
from _style import BLUE, GREEN, INK2, INK3, ORANGE, PURPLE, apply_style  # noqa: E402

CASES = ["case1", "case2", "case3", "case4"]
CASE_LAB = {"case1": "Case 1: i.i.d. noise, one variance", "case2": "Case 2: AR(1), one variance", "case3": "Case 3: AR(1), variance free by age",
            "case4": "Case 4: AR(1) + one-period error"}
CASE_COL = {"case1": GREEN, "case2": ORANGE, "case3": BLUE, "case4": PURPLE}
LAYER_COL = {"VH": BLUE, "CHd": ORANGE, "Vd": GREEN, "noise": "#b8b8b8", "spike": "#6f6f6f"}
LAYER_LAB = {"VH": "$V_H$", "CHd": "$(s+t)\\,C_{Hd}$", "Vd": "$st\\,V_d$", "noise": "carried noise",
             "spike": "one-period / i.i.d. noise"}


def main() -> int:
    apply_style()
    d = ident_panel()
    i, j = pairs_of(OFFS4)
    names = cell_names(i, j, 2)
    sample = {"people": d["pidp"].nunique(), "person_ages": len(d)}
    print(f"identification panel: {sample['person_ages']:,} person-ages, {sample['people']:,} people")
    corr_rows, dec_rows = [], []
    fits = {}
    for v, lab in VARIANTS:
        W = wide(d, v)
        for how in ("balanced", "pairwise"):
            for bn, (lo, hi) in BANDS.items():
                m, n = pooled_moments(W, OFFS4, lo, hi - 6, how)
                if m is None:
                    continue
                for case in CASES:
                    best, ok = fit_case(m, case, i, j, OFFS4)
                    fits[(v, how, bn, case)] = (m, best, ok)
                    corr_rows.append({"variant": v, "moments": how, "band": bn, "case": case, "n": n,
                                      "corr": corr_of(best["b"]), "chd": best["b"][1], "rho": best["rho"],
                                      "admissible": admissible(best["b"]),
                                      "corr_admissible": corr_of(ok["b"]) if ok else np.nan,
                                      "rho_admissible": ok["rho"] if ok else np.nan})
                    L = layers(case, best["b"], best["X"], i, j)
                    for c in range(len(i)):
                        dec_rows.append({"variant": v, "moments": how, "band": bn, "case": case, "cell": names[c],
                                         "i": i[c], "j": j[c], "observed": m[c], "fitted": best["fit"][c],
                                         **{k: L[k][c] for k in L}})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(DESC / "paper_ident_corr_band.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(DESC / "paper_ident_decomposition.csv", index=False)
    pd.DataFrame([sample]).to_csv(DESC / "paper_ident_sample.csv", index=False)
    show = corr.pivot_table(index=["variant", "moments", "case"], columns="band", values="corr")
    print(show.round(2).to_string())

    # ---- figure: Corr(H, d) by band ---------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharey=True, gridspec_kw={"hspace": 0.4, "wspace": 0.15})
    bands = list(BANDS)
    for col, (v, lab) in enumerate(VARIANTS):
        for row, how in enumerate(("balanced", "pairwise")):
            ax = axes[row, col]
            for case in CASES:
                q = corr[(corr["variant"] == v) & (corr["moments"] == how) & (corr["case"] == case)].set_index("band")
                y = q["corr"].reindex(bands).clip(-1.05, 1.05).to_numpy()
                ax.plot(range(4), y, color=CASE_COL[case], lw=1.5, label=CASE_LAB[case] if (row, col) == (0, 0) else None)
                adm = q["admissible"].reindex(bands).to_numpy()
                ax.scatter(np.arange(4)[adm], y[adm], color=CASE_COL[case], s=28, zorder=3)
                ax.scatter(np.arange(4)[~adm], y[~adm], color=CASE_COL[case], s=34, marker="x", zorder=3)
                ya = q["corr_admissible"].reindex(bands).to_numpy()
                ax.scatter(np.arange(4)[~adm], ya[~adm], facecolors="none", edgecolors=CASE_COL[case], s=40, zorder=3)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(4)); ax.set_xticklabels(bands)
            ax.set_ylim(-1.1, 1.1); ax.grid(True, axis="y")
            ax.set_title(f"{lab}, {how} moments", loc="left", fontsize=9.5)
    axes[0, 0].set_ylabel(r"$\mathrm{Corr}(H, d)$"); axes[1, 0].set_ylabel(r"$\mathrm{Corr}(H, d)$")
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.text(0.01, -0.01,
             f"Identification panel: {sample['people']:,} people with at least four observed ages 20-90, one row per "
             "person-age. Four ages two years apart, pooled over base ages within each band; rho profiled (Case 1 has none), the rest by "
             "least squares.\nDots: admissible (V_H, V_d >= 0, |Corr| <= 1, noise variances >= 0) at the best-fitting "
             "rho; crosses: not admissible, drawn clipped, with the best admissible rho's value as an open circle. Balanced moments use people observed at all four ages.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_exhibit2_corr.png")

    # ---- figures: the barcharts, one per variant and moment type ---------------
    for (v, lab), how in [(vl, h_) for h_ in ("balanced", "pairwise") for vl in VARIANTS]:
        fig, axes = plt.subplots(4, 4, figsize=(13, 13), sharey="row", gridspec_kw={"hspace": 0.55, "wspace": 0.08})
        for r, case in enumerate(CASES):
            for c, bn in enumerate(bands):
                ax = axes[r, c]
                if (v, how, bn, case) not in fits:
                    continue
                m, best, ok = fits[(v, how, bn, case)]
                L = layers(case, best["b"], best["X"], i, j)
                x = np.arange(len(i)) + np.where(np.arange(len(i)) >= 4, 0.6, 0) + np.where(np.arange(len(i)) >= 7, 0.6, 0)
                pos = np.zeros(len(i)); neg = np.zeros(len(i))
                for key in ("VH", "CHd", "Vd", "noise", "spike"):
                    val = L[key]
                    up = val >= 0
                    ax.bar(x[up], val[up], bottom=pos[up], color=LAYER_COL[key], width=0.8,
                           label=LAYER_LAB[key] if (r, c) == (0, 0) else None)
                    ax.bar(x[~up], val[~up], bottom=neg[~up], color=LAYER_COL[key], width=0.8)
                    pos[up] += val[up]; neg[~up] += val[~up]
                ax.hlines(m, x - 0.42, x + 0.42, color="black", lw=1.6, label="observed" if (r, c) == (0, 0) else None)
                ax.axhline(0, color=INK3, lw=0.6)
                ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
                tag = f"rho = {best['rho']:.2f}, Corr(H,d) = {corr_of(best['b']):+.2f}" + ("" if admissible(best["b"]) else "\nnot admissible")
                ax.set_title((f"ages {bn}\n" if r == 0 else "") + tag, fontsize=7.5, loc="left")
                ax.grid(True, axis="y")
                if c == 0:
                    ax.set_ylabel(CASE_LAB[case], fontsize=8.5)
        axes[0, 0].legend(loc="upper left", fontsize=7, ncols=2)
        fig.suptitle(f"Exhibit 2 on {lab}: four ages two years apart, {how} moments, fitted layers by cell",
                     fontweight="bold", y=0.995)
        fig.text(0.01, -0.01,
                 "Cells: V(k) the variance at offset 2k years, C(k, l) the covariance between offsets. Bars stack the "
                 "fitted layers at the best-fitting rho; the black tick is the observed moment.\nCase 1 has i.i.d. noise; Case 3 one carried "
                 "variance per age; Case 4 adds a one-period error on the diagonal. "
                 + ("Balanced: people observed at all four ages." if how == "balanced" else
                    "Pairwise: each cell over the people observed at both of its ages, so cells differ in who they cover."),
                 fontsize=7.4, color=INK2, va="top")
        fig.savefig(FIG / (f"fig_exhibit2_{v}.png" if how == "balanced" else f"fig_exhibit2_{v}_pairwise.png"))
        plt.close(fig)
    # ---- the main-text figure: Case 1 and Case 4, h and theta, balanced moments -----
    rows_spec = [("h", "case1"), ("h", "case4"), ("theta", "case1"), ("theta", "case4")]
    fig, axes = plt.subplots(4, 4, figsize=(13, 12), sharey="row", gridspec_kw={"hspace": 0.6, "wspace": 0.08})
    for r, (v, case) in enumerate(rows_spec):
        lab = LABEL[v]
        for c, bn in enumerate(bands):
            ax = axes[r, c]
            m, best, ok = fits[(v, "balanced", bn, case)]
            L = layers(case, best["b"], best["X"], i, j)
            x = np.arange(len(i)) + np.where(np.arange(len(i)) >= 4, 0.6, 0) + np.where(np.arange(len(i)) >= 7, 0.6, 0)
            pos = np.zeros(len(i)); neg = np.zeros(len(i))
            for key in ("VH", "CHd", "Vd", "noise", "spike"):
                val = L[key]; up = val >= 0
                ax.bar(x[up], val[up], bottom=pos[up], color=LAYER_COL[key], width=0.8,
                       label=LAYER_LAB[key] if (r, c) == (0, 0) else None)
                ax.bar(x[~up], val[~up], bottom=neg[~up], color=LAYER_COL[key], width=0.8)
                pos[up] += val[up]; neg[~up] += val[~up]
            ax.hlines(m, x - 0.42, x + 0.42, color="black", lw=1.6, label="observed" if (r, c) == (0, 0) else None)
            ax.axhline(0, color=INK3, lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
            rho_txt = "" if case == "case1" else f"rho = {best['rho']:.2f}, "
            tag = f"{rho_txt}Corr(H,d) = {corr_of(best['b']):+.2f}" + ("" if admissible(best["b"]) else "\nnot admissible")
            ax.set_title((f"ages {bn}\n" if r == 0 else "") + tag, fontsize=7.5, loc="left")
            ax.grid(True, axis="y")
            if c == 0:
                ax.set_ylabel(f"{lab}\n{CASE_LAB[case]}", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=6, fontsize=8, bbox_to_anchor=(0.5, 0.97))
    fig.text(0.01, -0.01,
             "Balanced moments (people observed at all four ages), four ages two years apart pooled over base ages within "
             "each band. Bars stack the fitted layers; the black tick is the observed moment.\nCase 1: i.i.d. noise with one "
             "variance, read off the diagonal alone. Case 4: an AR(1) carried component plus a one-period error, rho profiled.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_exhibit2_main.png"); plt.close(fig)
    print(f"wrote fig_exhibit2_corr.png, fig_exhibit2_main.png and the barchart figures to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
