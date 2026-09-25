"""Exhibit 2 and the identification moments behind it, on the paper's measure.

Two designs, both fitted with the properly constrained estimator
(_paper_moments.fit_constrained / fit_common): V_H, V_d and every noise variance
non-negative and |Corr(H, d)| <= 1 imposed in the estimation, rho profiled over
a grid. Where the unconstrained fit is already admissible the two coincide;
where it is not, the figures say so and report what the constraint costs.

  main       four ages four years apart, pairwise moments, pooled over base ages
             within each band; a base age enters when every cell rests on at
             least 50 people observed at both of its ages. rho fitted band by
             band. This is Exhibit 2 in the main text.
  2y         four ages two years apart, balanced moments (people observed at all
             four ages), one rho shared by the four bands. The appendix version:
             on h it is admissible in every band with nothing imposed.

Outputs, paper/figures/:
  fig_exhibit2_main.png       Cases 1 and 4 on theta and h, main design (main text)
  fig_exhibit2_2y.png         Cases 1 and 4 on h and theta, 2y design (appendix)
  fig_exhibit2_corr.png       Corr(H, d) by band under every case, both designs (appendix)
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
    BANDS, admissible, cell_names, corr_of, fit_common, fit_constrained, ident_panel, layers, pairs_of,
    pooled_moments, wide,
)
from _style import BLUE, GREEN, INK2, INK3, ORANGE, PURPLE, apply_style  # noqa: E402

CASES = ["case1", "case2", "case3", "case4"]
CASE_LAB = {"case1": "Case 1: i.i.d. noise", "case2": "Case 2: AR(1), one variance",
            "case3": "Case 3: AR(1), variance by age", "case4": "Case 4: AR(1) + one-period error"}
CASE_COL = {"case1": GREEN, "case2": ORANGE, "case3": BLUE, "case4": PURPLE}
LAYER_COL = {"VH": BLUE, "CHd": ORANGE, "Vd": GREEN, "noise": "#b8b8b8", "spike": "#6f6f6f"}
LAYER_LAB = {"VH": "$V_H$", "CHd": "$(s+t)\\,C_{Hd}$", "Vd": "$st\\,V_d$", "noise": "carried noise",
             "spike": "one-period / i.i.d. noise"}
DESIGNS = {
    "main": {"offs": np.array([0, 4, 8, 12]), "how": "pairwise", "cell_min": 50, "common": False,
             "label": "four ages four years apart, pairwise moments"},
    "2y": {"offs": np.array([0, 2, 4, 6]), "how": "balanced", "cell_min": None, "common": True,
           "label": "four ages two years apart, balanced moments, one rho across bands"},
}
BAND_NAMES = list(BANDS)
WEAK_VH = 0.05    # V_H below this share of the first variance: the level, and with it Corr(H, d), is not identified


def fit_design(W: np.ndarray, spec: dict) -> dict:
    """{(band, case): (moments, n, fit)} for one variant under one design."""
    offs = spec["offs"]
    i, j = pairs_of(offs)
    span = int(offs.max())
    ms, ns = {}, {}
    for bn, (lo, hi) in BANDS.items():
        m, n = pooled_moments(W, offs, lo, hi - span, spec["how"], cell_min=spec["cell_min"])
        if m is not None:
            ms[bn], ns[bn] = m, n
    out = {}
    for case in CASES:
        if spec["common"] and case != "case1":
            for bn, f in fit_common(ms, case, i, j, offs).items():
                out[(bn, case)] = (ms[bn], ns[bn], f)
        else:
            for bn, m in ms.items():
                out[(bn, case)] = (m, ns[bn], fit_constrained(m, case, i, j, offs))
    for (bn, case), (m, n, f) in out.items():
        f["weak"] = bool(f["b"][0] < WEAK_VH * m[0])
    return out


def draw(ax, m, f, case, i, j, names, first):
    L = layers(case, f["b"], f["X"], i, j)
    x = np.arange(len(i)) + np.where(np.arange(len(i)) >= 4, 0.6, 0) + np.where(np.arange(len(i)) >= 7, 0.6, 0)
    pos, neg = np.zeros(len(i)), np.zeros(len(i))
    for key in ("VH", "CHd", "Vd", "noise", "spike"):
        val = L[key]
        up = val >= 0
        ax.bar(x[up], val[up], bottom=pos[up], color=LAYER_COL[key], width=0.8, label=LAYER_LAB[key] if first else None)
        ax.bar(x[~up], val[~up], bottom=neg[~up], color=LAYER_COL[key], width=0.8)
        pos[up] += val[up]
        neg[~up] += val[~up]
    ax.hlines(m, x - 0.42, x + 0.42, color="black", lw=1.6, label="observed" if first else None)
    ax.axhline(0, color=INK3, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6.5)
    ax.grid(True, axis="y")
    rho = "" if case == "case1" else f"rho {f['rho']:.2f}, "
    flag = " (at bound)" if f.get("binds") else (" (V_H near 0: not identified)" if f.get("weak") else "")
    tag = f"{rho}Corr(H,d) {corr_of(f['b']):+.2f}" + flag
    u = f.get("unconstrained")
    if u is not None and not admissible(u["b"]):
        c = corr_of(u["b"])
        tag += ("\nunconstr.: " + ("" if case == "case1" else f"rho {u['rho']:.2f}, ")
                + (f"Corr {c:+.2f}" if np.isfinite(c) else "variance < 0") + f"; cost {100 * f['cost']:+.0f}%")
    return tag


def barchart(path, rows, fits, spec, title, footer):
    """rows: [(variant, case)]; fits: {variant: fit_design output}."""
    i, j = pairs_of(spec["offs"])
    names = cell_names(i, j, int(spec["offs"][1]))
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.05 * len(rows)), sharey="row",
                             gridspec_kw={"hspace": 0.95, "wspace": 0.08})
    for r, (v, case) in enumerate(rows):
        for c, bn in enumerate(BAND_NAMES):
            ax = axes[r, c]
            if (bn, case) not in fits[v]:
                ax.text(0.5, 0.5, "too few people", ha="center", va="center", transform=ax.transAxes, color=INK2)
                ax.set_xticks([])
                continue
            m, n, f = fits[v][(bn, case)]
            tag = draw(ax, m, f, case, i, j, names, (r, c) == (0, 0))
            ax.set_title((f"ages {bn}\n" if r == 0 else "") + tag, fontsize=6.8, loc="left")
            if c == 0:
                ax.set_ylabel(f"{LABEL[v]}\n{CASE_LAB[case]}", fontsize=8.2)
    fig.subplots_adjust(top=1 - 0.3 / len(rows), bottom=0.05)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncols=6, fontsize=8,
               bbox_to_anchor=(0.5, 1 - 0.12 / len(rows)))   # no suptitle: the paper's caption names the figure
    fig.text(0.01, 0.0, footer, fontsize=7.4, color=INK2, va="top")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    apply_style()
    d = ident_panel()
    sample = {"people": d["pidp"].nunique(), "person_ages": len(d)}
    print(f"identification panel: {sample['person_ages']:,} person-ages, {sample['people']:,} people")
    fits = {dn: {} for dn in DESIGNS}
    corr_rows, dec_rows = [], []
    for v, _ in VARIANTS:
        W = wide(d, v)
        for dn, spec in DESIGNS.items():
            fits[dn][v] = fit_design(W, spec)
            i, j = pairs_of(spec["offs"])
            names = cell_names(i, j, int(spec["offs"][1]))
            for (bn, case), (m, n, f) in fits[dn][v].items():
                u = f.get("unconstrained")
                corr_rows.append({"variant": v, "design": dn, "band": bn, "case": case, "n": n,
                                  "corr": corr_of(f["b"]), "chd": f["b"][1], "rho": f["rho"],
                                  "binds": bool(f.get("binds")), "weak": bool(f.get("weak")), "cost": f.get("cost", np.nan),
                                  "corr_unconstrained": corr_of(u["b"]) if u else np.nan,
                                  "rho_unconstrained": u["rho"] if u else np.nan,
                                  "admissible_unconstrained": admissible(u["b"]) if u else f.get("admissible_unconstrained")})
                L = layers(case, f["b"], f["X"], i, j)
                for c in range(len(i)):
                    dec_rows.append({"variant": v, "design": dn, "band": bn, "case": case, "cell": names[c],
                                     "i": i[c], "j": j[c], "observed": m[c], "fitted": f["fit"][c],
                                     **{k: L[k][c] for k in L}})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(DESC / "paper_ident_corr_band.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(DESC / "paper_ident_decomposition.csv", index=False)
    pd.DataFrame([sample]).to_csv(DESC / "paper_ident_sample.csv", index=False)
    show = corr.assign(val=corr["corr"].round(2).astype(str) + np.where(corr["binds"], "b", "") + np.where(corr["weak"], "w", "")).pivot_table(
        index=["design", "variant", "case"], columns="band", values="val", aggfunc="first")[BAND_NAMES]
    print(show.to_string())

    main_spec, y2_spec = DESIGNS["main"], DESIGNS["2y"]
    common_foot = ("Properly constrained fit: V_H, V_d and noise variances non-negative and |Corr(H,d)| <= 1 imposed in "
                   "the estimation, rho profiled. Where the unconstrained fit was inadmissible the title gives it and the "
                   f"fit cost of the constraint\nin squared error. Where V_H falls below {WEAK_VH:.0%} of the first variance the level, and so Corr(H,d), is not identified. Bars stack the fitted layers; black ticks are the observed "
                   "moments. V(k): the variance at the k-th age; C(k,l): the covariance between ages k and l.")
    main_foot = ("Four ages four years apart, pairwise moments pooled over base ages within each band; a base age enters "
                 "when every cell rests on at least 50 people observed at both of its ages.\n" + common_foot)
    barchart(FIG / "fig_exhibit2_main.png", [("theta", "case1"), ("theta", "case4"), ("h", "case1"), ("h", "case4")],
             fits["main"], main_spec, "Exhibit 2: Cases 1 and 4 on theta and h", main_foot)
    barchart(FIG / "fig_exhibit2_2y.png", [("h", "case1"), ("h", "case4"), ("theta", "case1"), ("theta", "case4")],
             fits["2y"], y2_spec, "Exhibit 2, two-year design: Cases 1 and 4 on h and theta",
             "Four ages two years apart, balanced moments (people observed at all four ages), one rho shared by the four "
             "bands in Case 4.\n" + common_foot)

    # ---- summary: Corr(H, d) by band, every case, both designs -------------------
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey=True, gridspec_kw={"hspace": 0.45, "wspace": 0.15})
    for r, dn in enumerate(DESIGNS):
        for c, (v, lab) in enumerate(VARIANTS):
            ax = axes[r, c]
            for case in CASES:
                q = corr[(corr["variant"] == v) & (corr["design"] == dn) & (corr["case"] == case)].set_index("band")
                q = q.reindex(BAND_NAMES)
                x = np.arange(4)
                y = q["corr"].to_numpy(float)
                b = (q["binds"].fillna(False) | q["weak"].fillna(False)).to_numpy(bool)
                ax.plot(x, y, color=CASE_COL[case], lw=1.5, label=CASE_LAB[case] if (r, c) == (0, 0) else None)
                ax.scatter(x[~b], y[~b], color=CASE_COL[case], s=28, zorder=3)
                ax.scatter(x[b], y[b], facecolors="none", edgecolors=CASE_COL[case], s=40, zorder=3)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(BAND_NAMES)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, axis="y")
            ax.set_title(f"{lab}, {'4 x 4 years, pairwise' if dn == 'main' else '4 x 2 years, balanced, common rho'}",
                         loc="left", fontsize=9)
    axes[0, 0].set_ylabel(r"$\mathrm{Corr}(H, d)$")
    axes[1, 0].set_ylabel(r"$\mathrm{Corr}(H, d)$")
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.text(0.01, -0.01,
             f"Identification panel: {sample['people']:,} people with at least four observed ages 20-90. Top: the main "
             "design; bottom: the two-year design with one rho shared across bands (Cases 2-4). Properly constrained fit\n"
             "throughout; open circles mark bands where the admissibility constraint binds, so only the sign of Corr(H,d) "
             f"is identified, or where V_H is below {WEAK_VH:.0%} of the first variance, so nothing is.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_exhibit2_corr.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_exhibit2_main.png, fig_exhibit2_2y.png and fig_exhibit2_corr.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
