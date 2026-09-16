"""Figures and tables for the measures note's section on the limitation banks.

Per bank, with the GRM and GPCM paired on the h scale:
  fig_lim_exhibit1_<bank>.png   Exhibit 1: mean and variance by age, and
                                Corr(H, d) by band under both estimators
  fig_lim_bars_<bank>.png       identification barcharts: observed moments and
                                fitted layers by cell, Case 2 pairwise and
                                Case 3 balanced
  fig_lim_clusters_<bank>.png   partial K-means, K = 3: trajectories and
                                composition by age
Tables, measuring_health/tables/: lim_categories.tex, lim_fit.tex,
lim_criteria.tex, lim_weights_a.tex, lim_weights_b.tex. Weights also go to
data/processed/baseline_measures/limitation_weights.csv for the workbook.

Inputs: data_cleaning/07_build_limitation_banks.py and scripts 05-09 here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import BLUE, CLUSTER, GREEN, INK, INK2, INK3, ORANGE, PURPLE, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402

MEAS = PROCESSED_DATA_DIR / "measures"
OUT = PROCESSED_DATA_DIR / "baseline_measures"
FIG = ROOT_DIR / "measuring_health" / "figures"
TAB = ROOT_DIR / "measuring_health" / "tables"
SLUG = {"P-FUNC": "pfunc", "P-LIM1": "plim1", "P-LIM": "plim", "P-LIM4": "plim4",
        "P-LIM3": "plim3", "P-LIM3+O": "plim3o"}
SF = ["GH", "PF", "RP", "BP"]
BANKS = {"P-FUNC": ["FUNC"], "P-LIM1": ["LIM_PF"], "P-LIM": ["LIM_PF", "LIM_SC"],
         "P-LIM4": ["LIM_MOB", "LIM_DEX", "LIM_CONT", "LIM_SENS"],
         "P-LIM3": ["LIM_FL", "LIM_SELF", "LIM_SENS"],
         "P-LIM3+O": ["LIM_FL", "LIM_SELF", "LIM_SENS", "LIM_OTHER"]}
SHORT = {"GH": "GH", "PF": "PF", "RP": "RP", "BP": "BP", "FUNC": "Lim.\\ 3+",
         "LIM_PF": "Lim.\\ 0--5", "LIM_SC": "Sens.+cont.", "LIM_MOB": "Mob.+lift.",
         "LIM_DEX": "Dex.+care", "LIM_CONT": "Cont.", "LIM_SENS": "Sensory",
         "LIM_FL": "Func.\\ lim.", "LIM_SELF": "Self-care", "LIM_OTHER": "Other"}
MODEL_LS = {"grm": "-", "gpcm": "--"}
MODEL_LAB = {"grm": "graded response", "gpcm": "partial credit"}
SPEC_COL = {"case2": ORANGE, "case3": BLUE, "case4": PURPLE}
CASE_LAB = {"case2": "Case 2", "case3": "Case 3", "case4": "Case 4"}
SPEC_LAB = {"case3": "Case 3, balanced", "case2": "Case 2, pairwise", "case4": "Case 4, balanced"}
BANDS = ["25-40", "40-60", "60-75", "75-90"]
LAYER = [("VH", "$V[H]$", BLUE), ("CHd", "$(s+t)\\,C[H,d]$", ORANGE),
         ("Vd", "$s\\,t\\,V[d]$", GREEN), ("noise", "AR(1) noise", "#a9a9a9"),
         ("spike", "one-period error", "#dcdcdc")]
D17 = 1.702
CON_BANKS = ["P-LIM1", "P-LIM", "P-LIM4", "P-LIM3", "P-LIM3+O"]
CODES = {"CC": ["COND"], "CG": ["CVD", "METAB", "RESP", "MSK", "CANCER", "OTHER"]}


def exhibit1(lines, ref, prof, corr, out_name):
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.4))
    for ax, col, title in ((axs[0, 0], "mean", "(a) Mean, pooled-sd units"),
                           (axs[0, 1], "vrel", "(b) Variance, relative to 30–34")):
        if ref is not None:
            r = prof[prof["measure"] == ref[0]]
            ax.plot(r["age"], r[col], color=INK3, lw=1.0, alpha=0.7)
        for meas, _, ls in lines:
            r = prof[prof["measure"] == meas]
            ax.plot(r["age"], r[col], color=INK, lw=1.6, ls=ls)
        ax.set_title(title, loc="left")
        ax.set_xlabel("age")
        ax.grid(True, axis="y")
    x = np.arange(len(BANDS))
    for ax, how, title in ((axs[1, 0], "balanced", "(c) Corr(H, d), balanced moments"),
                           (axs[1, 1], "pairwise", "(d) Corr(H, d), pairwise moments")):
        ax.axhline(0, color=INK2, lw=0.8)
        for spec, c in SPEC_COL.items():
            for meas, _, ls in lines:
                r = corr[(corr["measure"] == meas) & (corr["spec"] == spec)
                         & (corr["moments"] == how)].set_index("band").loc[BANDS]
                y = r["corr"].to_numpy(float)
                yc = np.clip(y, -1.0, 1.0)
                ax.plot(x, yc, color=c, ls=ls, lw=1.4)
                ok = r["ok"].astype(bool).to_numpy() & ~np.isnan(y)
                bad = ~r["ok"].astype(bool).to_numpy() & ~np.isnan(y)
                ax.scatter(x[ok], yc[ok], color=c, s=20, zorder=3)
                ax.scatter(x[bad], yc[bad], color=c, marker="x", s=28, zorder=3)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xticks(x, [b.replace("-", "–") for b in BANDS])
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y")
    handles = [Line2D([], [], color=INK, ls=ls, lw=1.6, label=lab) for _, lab, ls in lines]
    if ref is not None:
        handles.append(Line2D([], [], color=INK3, lw=1.0, label=ref[1]))
    handles += [Line2D([], [], color=c, lw=1.5, label=CASE_LAB[k]) for k, c in SPEC_COL.items()]
    handles += [Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="admissible"),
                Line2D([], [], color=INK2, marker="x", ls="", ms=5, label="not admissible")]
    fig.legend(handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.07), fontsize=8)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(FIG / out_name)
    plt.close(fig)


CELL_ORDER = ["V(0)", "V(1)", "V(2)", "V(3)", "C(0,1)", "C(0,2)", "C(0,3)", "C(1,2)", "C(1,3)", "C(2,3)"]
CELL_X = np.r_[0, 1, 2, 3, 4.6, 5.6, 6.6, 8.2, 9.2, 10.2]
CELL_TICK = ["V0", "V1", "V2", "V3", "C01", "C02", "C03", "C12", "C13", "C23"]


def bars(dec, rows, out_name, height):
    fig, axs = plt.subplots(len(rows), len(BANDS), figsize=(8.2, height), sharey="row", squeeze=False)
    layers = [lay for lay in LAYER if lay[0] != "spike" or any(r[0] == "case4" for r in rows)]
    for ri, (spec, how, meas, lab) in enumerate(rows):
        for bi, band in enumerate(BANDS):
            ax = axs[ri, bi]
            r = dec[(dec["measure"] == meas) & (dec["spec"] == spec) & (dec["moments"] == how)
                    & (dec["band"] == band)]
            r = r.set_index("cell").loc[CELL_ORDER]
            pos, neg = np.zeros(len(r)), np.zeros(len(r))
            for key, _, col in layers:
                v = r[key].to_numpy()
                base = np.where(v >= 0, pos, neg)
                ax.bar(CELL_X, v, bottom=base, width=0.78, color=col, lw=0)
                pos, neg = pos + np.maximum(v, 0), neg + np.minimum(v, 0)
            ax.hlines(r["observed"], CELL_X - 0.45, CELL_X + 0.45, color=INK, lw=1.3, zorder=4)
            ax.axhline(0, color=INK2, lw=0.6)
            ok = bool(r["ok"].iloc[0])
            if ri == 0:
                ax.set_title(f"ages {band.replace('-', '–')}", loc="center")
            ax.set_xticks(CELL_X, CELL_TICK if ri == len(rows) - 1 else [], fontsize=7, rotation=90)
            ax.tick_params(axis="y", labelsize=7.5)
            if not ok:
                ax.text(0.98, 0.97, "not admissible", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5, color="#b2182b")
            if bi == 0:
                ax.set_ylabel(f"{CASE_LAB[spec]}, {how}\n{lab}", fontsize=8.5)
            ax.grid(True, axis="y")
    handles = [Patch(color=col, label=lab) for _, lab, col in layers]
    handles.append(Line2D([], [], color=INK, lw=1.3, label="observed moment"))
    pad = 0.035 * 9.8 / height
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -pad), fontsize=8.5)
    fig.tight_layout(rect=(0, pad, 1, 1), h_pad=0.8, w_pad=0.4)
    fig.savefig(FIG / out_name)
    plt.close(fig)


def clusters(panels, traj, comp, labels, out_name):
    fig = plt.figure(figsize=(8.6, 4.0))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[3.2, 0.8], hspace=0.12, wspace=0.18)
    for ci, (meas, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        axc = fig.add_subplot(gs[1, ci], sharex=ax)
        t = traj[traj["measure"] == meas]
        for cl in range(3):
            r = t[(t["cluster"] == cl) & (t["count"] >= 25)]
            ax.plot(r["age"], r["mean"], color=CLUSTER[cl], lw=1.8, label=f"cluster {cl + 1}")
        lab = labels[meas].dropna()
        shares = " / ".join(f"{100 * (lab == c).mean():.0f}%" for c in range(3))
        ax.set_title(f"{title}: shares {shares}", loc="left", fontsize=9.5)
        ax.grid(True, axis="y")
        ax.tick_params(labelbottom=False)
        if ci == 0:
            ax.set_ylabel("h")
            ax.legend(loc="lower left")
        cp = comp[comp["measure"] == meas].pivot(index="age", columns="cluster", values="share")
        cp = cp.reindex(columns=range(3), fill_value=0).fillna(0)
        axc.stackplot(cp.index, *[cp[c] for c in range(3)], colors=CLUSTER, alpha=0.9)
        axc.set_ylim(0, 1)
        axc.set_yticks([])
        axc.spines["left"].set_visible(False)
        axc.set_xlim(20, 90)
        axc.set_xlabel("age")
    fig.savefig(FIG / out_name)
    plt.close(fig)


def weights():
    codes = pd.read_parquet(MEAS / "limitation_bank_items.parquet")
    scores = pd.read_parquet(MEAS / "limitation_scores.parquet")
    items = pd.read_csv(MEAS / "limitation_items.csv")
    fw = pd.read_csv(OUT / "limitation_factor_weights.csv")
    rows, r2 = [], {}
    for bank, lims in BANKS.items():
        names = SF + lims
        X = codes[names].to_numpy(float)
        sd = X.std(axis=0)
        Z = np.column_stack([np.ones(len(X)), (X - X.mean(axis=0)) / sd])
        for m in ("grm", "gpcm"):
            it = items[(items["bank"] == bank) & (items["model"] == m)].set_index("item").loc[names]
            a = it["a"].to_numpy()
            for i, nm in enumerate(names):
                rows.append({"bank": bank, "item": nm, "method": f"{m}_a", "value": a[i]})
            if m == "grm":
                lam = a / np.sqrt(a**2 + D17**2)
                w = lam / (1 - lam**2)
                for i, nm in enumerate(names):
                    rows.append({"bank": bank, "item": nm, "method": "grm_latent", "value": w[i] / w.sum()})
                    rows.append({"bank": bank, "item": nm, "method": "grm_loading", "value": lam[i]})
            else:
                w = a * sd
                for i, nm in enumerate(names):
                    rows.append({"bank": bank, "item": nm, "method": "gpcm_exact", "value": w[i] / w.sum()})
            y = scores[f"h_{SLUG[bank]}_{m}"].to_numpy()
            b, *_ = np.linalg.lstsq(Z, y, rcond=None)
            r2[(bank, m)] = 1 - ((y - Z @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            for i, nm in enumerate(names):
                rows.append({"bank": bank, "item": nm, "method": f"{m}_projection",
                             "value": b[1 + i] / b[1:].sum()})
        for meth in ("polychoric_factor", "pearson_factor", "pearson_pc1"):
            f = fw[(fw["bank"] == bank) & (fw["method"] == meth)].set_index("item").loc[names]
            for nm in names:
                rows.append({"bank": bank, "item": nm, "method": meth, "value": f.loc[nm, "weight"]})
                if meth == "polychoric_factor":
                    rows.append({"bank": bank, "item": nm, "method": "polychoric_loading",
                                 "value": f.loc[nm, "loading"]})
    W = pd.DataFrame(rows)
    W.to_csv(OUT / "limitation_weights.csv", index=False)
    return W, r2


def tex_weights(W, r2):
    meth = [("grm_a", "GRM discrimination $a$", "{:.2f}"),
            ("gpcm_a", "GPCM slope $a$", "{:.2f}"),
            ("grm_latent", "GRM, latent: $\\lambda/\\psi$", "{:.0f}"),
            ("polychoric_factor", "Polychoric one-factor", "{:.0f}"),
            ("grm_projection", "GRM $h$ on codes", "{:.0f}"),
            ("gpcm_exact", "GPCM, exact: $a\\times$sd", "{:.0f}"),
            ("gpcm_projection", "GPCM $h$ on codes", "{:.0f}"),
            ("pearson_factor", "Pearson one-factor", "{:.0f}"),
            ("pearson_pc1", "Pearson first component", "{:.0f}")]
    for part, banks in (("a", ["P-FUNC", "P-LIM1", "P-LIM"]), ("b", ["P-LIM4", "P-LIM3", "P-LIM3+O"])):
        tex_weights_part(W, r2, meth, part, banks)


def tex_weights_part(W, r2, meth, part, banks):
    ncol = max(len(SF) + len(BANKS[b]) for b in banks)
    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}",
           f"\\begin{{tabular}}{{@{{}}l{'r' * ncol}@{{}}}}", "\\toprule"]
    for bi, bank in enumerate(banks):
        lims = BANKS[bank]
        names = SF + lims
        pad = [""] * (ncol - len(names))
        if bi:
            out.append("\\midrule")
        out.append(f"\\textbf{{{bank}}} & " + " & ".join(SHORT[n] for n in names + []) +
                   ("" if not pad else " & " + " & ".join(pad)) + " \\\\")
        out.append("\\cmidrule(l){2-" + str(len(names) + 1) + "}")
        for key, lab, fmt in meth:
            v = W[(W["bank"] == bank) & (W["method"] == key)].set_index("item")["value"]
            cells = [fmt.format(100 * v[n] if fmt == "{:.0f}" else v[n]) for n in names]
            if key == "grm_projection":
                lab += f" ($R^2$ {r2[(bank, 'grm')]:.3f})".replace("0.", ".")
            if key == "gpcm_projection":
                lab += f" ($R^2$ {r2[(bank, 'gpcm')]:.3f})".replace("0.", ".")
            out.append(lab + " & " + " & ".join(cells + pad) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / f"lim_weights_{part}.tex").write_text("\n".join(out) + "\n")


def tex_fit(agree):
    fs = pd.read_csv(MEAS / "limitation_fit_summary.csv")
    sc = pd.read_parquet(MEAS / "limitation_scores.parquet")
    out = ["\\begin{center}\\footnotesize", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
           "Bank & Testlets & Patterns & $\\log L$, GRM & $\\Delta\\log L$, GPCM & max $Q_3$, GRM & "
           "max $Q_3$, GPCM & $r(\\theta)$ \\\\", "\\midrule"]
    for bank in BANKS:
        g = fs[(fs["bank"] == bank) & (fs["model"] == "grm")].iloc[0]
        p = fs[(fs["bank"] == bank) & (fs["model"] == "gpcm")].iloc[0]
        r = sc[f"theta_{SLUG[bank]}_grm"].corr(sc[f"theta_{SLUG[bank]}_gpcm"])
        out.append(f"{bank} & {int(g['testlets'])} & {int(g['patterns']):,} & "
                   f"${g['log_likelihood']:,.0f}$ & ${p['log_likelihood'] - g['log_likelihood']:+,.0f}$ & "
                   f"${g['max_q3']:+.3f}$ & ${p['max_q3']:+.3f}$ & {r:.3f} \\\\".replace(",", "{,}"))
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "lim_fit.tex").write_text("\n".join(out) + "\n")


def tex_categories():
    cats = pd.read_csv(MEAS / "limitation_categories.csv")
    lab = {"FUNC": "Physical function, capped", "LIM_PF": "Physical function",
           "LIM_SC": "Sensory and continence", "LIM_MOB": "Mobility and lifting",
           "LIM_DEX": "Dexterity, coordination, personal care", "LIM_CONT": "Continence",
           "LIM_SENS": "Hearing and sight", "LIM_FL": "Functional limitations",
           "LIM_SELF": "Self-care", "LIM_OTHER": "Other health problem"}
    areas = {"FUNC": "1, 2, 3, 10, 11", "LIM_PF": "1, 2, 3, 10, 11", "LIM_SC": "4, 5, 6",
             "LIM_MOB": "1, 2", "LIM_DEX": "3, 10, 11", "LIM_CONT": "4", "LIM_SENS": "5, 6",
             "LIM_FL": "1, 2, 3, 10", "LIM_SELF": "4, 11", "LIM_OTHER": "12"}
    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{3.3cm}l>{\\raggedright\\arraybackslash}p{4.3cm}>{\\raggedright\\arraybackslash}p{2.9cm}@{}}", "\\toprule",
           "Testlet & Label & Areas & \\% with 0 / 1 / 2 / \\dots\\ difficulties & Banks \\\\",
           "\\midrule"]
    for g in lab:
        s = cats[cats["testlet"] == g].sort_values("difficulties")
        sh = " / ".join(f"{100 * v:.2f}" for v in s["share"])
        top = int(s["difficulties"].max())
        if g in ("FUNC", "LIM_SC"):
            sh += " (top is " + str(top) + "+)"
        banks = ", ".join(b for b, v in BANKS.items() if g in v)
        out.append(f"{SHORT[g]} & {lab[g]} & {areas[g]} & {sh} & {banks} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "lim_categories.tex").write_text("\n".join(out) + "\n")


def tex_criteria():
    py = pd.read_csv(OUT / "master_py.csv")
    rr = pd.read_csv(OUT / "master_r.csv")
    d = py.merge(rr, on="key").set_index("key")
    yn = {"Yes": "$\\surd$", "No": "$\\times$"}
    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llrrrrccrrcc@{}}", "\\toprule",
           " & & Floor & Distinct & \\multicolumn{4}{c}{$h$ scale} & \\multicolumn{4}{c}{$\\theta$ scale} \\\\",
           "\\cmidrule(lr){5-8}\\cmidrule(l){9-12}",
           "Bank & Model & 60--84 & 85--90 & Cost $t$ & Var. & C3 & C2 & "
           "Cost $t$ & Var. & C3 & C2 \\\\", "\\midrule"]
    refs = [("GRM original", "GRM h", "GRM theta"), ("phys", "PHYS-4", None)]
    for lab, kh, kt in refs:
        h = d.loc[kh]
        t = d.loc[kt] if kt else None
        out.append(f"{lab} & & {100 * h['worst_60_84']:.2f}\\% & {int(h['distinct_bottom_decile_85_90'])} & "
                   f"{h['cost_curv_capped_t']:.1f} & {h['var_growth_60_85']:.2f} & {yn[h['c3_signpath']]} & "
                   f"{yn[h['c2_signpath']]} & "
                   + (f"{t['cost_curv_capped_t']:.1f} & {t['var_growth_60_85']:.2f} & {yn[t['c3_signpath']]} & "
                      f"{yn[t['c2_signpath']]} \\\\" if t is not None else " & & & \\\\"))
    out.append("\\midrule")
    for bank in BANKS:
        for m, tag, mlab in (("grm", "", "GRM"), ("gpcm", " GPCM", "GPCM")):
            h, t = d.loc[f"{bank}{tag} h"], d.loc[f"{bank}{tag} theta"]
            out.append(f"{bank if m == 'grm' else ''} & {mlab} & {100 * h['worst_60_84']:.2f}\\% & "
                       f"{int(h['distinct_bottom_decile_85_90'])} & {h['cost_curv_capped_t']:.1f} & "
                       f"{h['var_growth_60_85']:.2f} & {yn[h['c3_signpath']]} & {yn[h['c2_signpath']]} & "
                       f"{t['cost_curv_capped_t']:.1f} & {t['var_growth_60_85']:.2f} & "
                       f"{yn[t['c3_signpath']]} & {yn[t['c2_signpath']]} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "lim_criteria.tex").write_text("\n".join(out) + "\n")


def tex_ident():
    rr = pd.read_csv(OUT / "master_r.csv").set_index("key")
    combos = [("c2b", "balanced"), ("c3", "balanced"), ("c4b", "balanced"),
              ("c2", "pairwise"), ("c3p", "pairwise"), ("c4p", "pairwise")]

    def mark(k, sp):
        r = rr.loc[k]
        if r[f"{sp}_signpath"] != "Yes":
            return "$\\times$"
        return "$\\surd$" if r[f"{sp}_admissible"] == "Yes" else "$\\surd^{\\circ}$"

    def old(k, sp):
        v = rr.loc[k, f"{sp}_corr_75-90"]
        return "---" if pd.isna(v) else f"${v:+.2f}$"

    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
           "\\begin{tabular}{@{}ll" + "cr" * 6 + "@{}}", "\\toprule",
           " & & \\multicolumn{6}{c}{Balanced moments} & \\multicolumn{6}{c}{Pairwise moments} \\\\",
           "\\cmidrule(lr){3-8}\\cmidrule(l){9-14}",
           " & & \\multicolumn{2}{c}{Case 2} & \\multicolumn{2}{c}{Case 3} & \\multicolumn{2}{c}{Case 4} & "
           "\\multicolumn{2}{c}{Case 2} & \\multicolumn{2}{c}{Case 3} & \\multicolumn{2}{c}{Case 4} \\\\",
           "Bank & Model" + " & path & 75--90" * 6 + " \\\\"]
    for scale, suffix, lab in (("h", "h", "$h$ scale"), ("theta", "theta", "$\\theta$ scale")):
        out.append("\\midrule")
        out.append(f"\\multicolumn{{14}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
        for bank in BANKS:
            for m, tag, mlab in (("grm", "", "GRM"), ("gpcm", " GPCM", "GPCM")):
                k = f"{bank}{tag} {suffix}"
                cells = []
                for sp, _ in combos:
                    cells += [mark(k, sp), old(k, sp)]
                out.append(f"{bank if m == 'grm' else ''} & {mlab} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "lim_ident.tex").write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------- condition banks
CON_LAB = {"COND": "Chronic count", "CVD": "Heart disease, stroke", "METAB": "Diabetes, blood pressure",
           "RESP": "Asthma, bronchitis, emphysema", "MSK": "Arthritis", "CANCER": "Cancer",
           "OTHER": "Thyroid, liver, epilepsy"}


PAIR_LAB = {"GH": "general health", "PF": "physical functioning", "RP": "role-physical", "BP": "bodily pain",
            "LIM_PF": "limitations", "LIM_SC": "sensory and continence", "LIM_MOB": "mobility and lifting",
            "LIM_DEX": "dexterity and care", "LIM_CONT": "continence", "LIM_SENS": "hearing and sight",
            "LIM_FL": "functional limitations", "LIM_SELF": "self-care", "LIM_OTHER": "other problem",
            "COND": "the condition count", "CVD": "heart disease", "METAB": "diabetes and blood pressure",
            "RESP": "respiratory", "MSK": "arthritis", "CANCER": "cancer", "OTHER": "other conditions"}


def weights_con():
    codes = pd.read_parquet(MEAS / "limitation_bank_items.parquet")
    scores = pd.read_parquet(MEAS / "limitation_scores.parquet")
    items = pd.read_csv(MEAS / "limitation_items.csv")
    fw = pd.read_csv(OUT / "limitation_factor_weights.csv")
    keep = codes["COND"].notna().to_numpy()
    rows, r2 = [], {}
    for bank in CON_BANKS:
        for code, cond in CODES.items():
            name = f"{bank}+{code}"
            names = SF + BANKS[bank] + cond
            X = codes.loc[keep, names].to_numpy(float)
            sd = X.std(axis=0)
            Z = np.column_stack([np.ones(len(X)), (X - X.mean(axis=0)) / sd])
            it = items[(items["bank"] == name) & (items["model"] == "grm")].set_index("item").loc[names]
            a = it["a"].to_numpy()
            lam = a / np.sqrt(a**2 + D17**2)
            w = lam / (1 - lam**2)
            y = scores.loc[keep, f"h_{SLUG[bank]}{code.lower()}_grm"].to_numpy()
            b, *_ = np.linalg.lstsq(Z, y, rcond=None)
            r2[name] = 1 - ((y - Z @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            for i, nm in enumerate(names):
                rows += [{"bank": name, "item": nm, "method": "grm_a", "value": a[i]},
                         {"bank": name, "item": nm, "method": "grm_latent", "value": w[i] / w.sum()},
                         {"bank": name, "item": nm, "method": "grm_projection", "value": b[1 + i] / b[1:].sum()}]
            for meth in ("polychoric_factor", "pearson_factor", "pearson_pc1"):
                f = fw[(fw["bank"] == name) & (fw["method"] == meth)].set_index("item").loc[names]
                rows += [{"bank": name, "item": nm, "method": meth, "value": f.loc[nm, "weight"]} for nm in names]
    return pd.DataFrame(rows), r2


def tex_con_weights(Wc, r2):
    meth = [("grm_latent", "GRM, latent"), ("polychoric_factor", "Polychoric"), ("grm_projection", "GRM $h$ on codes"),
            ("pearson_factor", "Pearson"), ("pearson_pc1", "First component")]
    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
           " & \\multicolumn{5}{c}{Share of weight on the condition testlets, \\%} & & \\\\", "\\cmidrule(lr){2-6}",
           "Bank & " + " & ".join(lab for _, lab in meth) + " & $R^2$ & Condition $a$ \\\\", "\\midrule"]
    for bank in CON_BANKS:
        for code, cond in CODES.items():
            name = f"{bank}+{code}"
            x = Wc[Wc["bank"] == name]
            shares = [100 * x[(x["method"] == k) & x["item"].isin(cond)]["value"].sum() for k, _ in meth]
            a = x[(x["method"] == "grm_a") & x["item"].isin(cond)]["value"]
            a_txt = f"{a.iloc[0]:.2f}" if code == "CC" else f"{a.min():.2f}--{a.max():.2f}"
            out.append(f"{name} & " + " & ".join(f"{v:.0f}" for v in shares) + f" & {r2[name]:.3f} & {a_txt} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "con_weights.tex").write_text("\n".join(out) + "\n")


def tex_con_categories():
    cats = pd.read_csv(MEAS / "limitation_categories.csv")
    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{5.2cm}l>{\\raggedright\\arraybackslash}p{4.6cm}@{}}", "\\toprule",
           "Testlet & Conditions & Codes & \\% with 0 / 1 / 2 / \\dots \\\\", "\\midrule"]
    conds = {"COND": "all sixteen below", "CVD": "heart failure, coronary heart disease, angina, heart attack, stroke",
             "METAB": "diabetes, high blood pressure", "RESP": "asthma, emphysema, chronic bronchitis", "MSK": "arthritis",
             "CANCER": "cancer", "OTHER": "hyper- and hypothyroidism, liver condition, epilepsy"}
    capped = {"COND": "0--5, 6+", "CVD": "0, 1, 2+", "METAB": "0, 1, 2", "RESP": "0, 1, 2+", "MSK": "0, 1",
              "CANCER": "0, 1", "OTHER": "0, 1+"}
    for g in conds:
        x = cats[cats["testlet"] == g].sort_values("difficulties")
        out.append(f"{CON_LAB[g]} & {conds[g]} & {capped[g]} & " + " / ".join(f"{100 * v:.2f}" for v in x["share"]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "con_categories.tex").write_text("\n".join(out) + "\n")


def tex_con_fit():
    fs = pd.read_csv(MEAS / "limitation_fit_summary.csv")
    q = pd.read_csv(MEAS / "limitation_q3.csv")
    out = ["\\begin{center}\\footnotesize", "\\begin{tabular}{@{}lrrrl@{}}", "\\toprule",
           "Bank & Testlets & Patterns & $\\log L$ & Largest $Q_3$ \\\\", "\\midrule"]
    for bank in CON_BANKS:
        for code in CODES:
            name = f"{bank}+{code}"
            f = fs[(fs["bank"] == name)].iloc[0]
            t = q[q["bank"] == name].sort_values("q3", ascending=False).iloc[0]
            out.append(f"{name} & {int(f['testlets'])} & {int(f['patterns']):,} & ${f['log_likelihood']:,.0f}$ & "
                       f"${t['q3']:+.3f}$, {PAIR_LAB.get(t['item_a'], t['item_a'])} with "
                       f"{PAIR_LAB.get(t['item_b'], t['item_b'])} \\\\".replace(",", "{,}").replace("{,} ", ", "))
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "con_fit.tex").write_text("\n".join(out) + "\n")


def control_stats(scores, meas, keep):
    x = scores.loc[keep, meas]
    age = scores.loc[keep, "age"]
    worst = x.min()
    o60 = x[age.between(60, 84)]
    o85 = x[age.between(85, 90)]
    z = (x - x.mean()) / x.std()
    return {"worst_60_84": float((o60 == worst).mean()), "distinct_bottom_decile_85_90": int(o85[o85 <= o85.quantile(0.10)].nunique()),
            "var_growth_60_85": float(z[age.between(85, 89)].var() / z[age.between(60, 64)].var())}


def signs_from_corr(corr, meas, spec, how):
    x = corr[(corr["measure"] == meas) & (corr["spec"] == spec) & (corr["moments"] == how)].set_index("band")
    path = bool(x.loc["25-40", "chd"] > 0 and x.loc["75-90", "chd"] < 0)
    return path, bool(x["ok"].astype(bool).all()), x.loc["75-90", "corr"]


def tex_con_criteria(corr):
    py = pd.read_csv(OUT / "master_py.csv")
    rr = pd.read_csv(OUT / "master_r.csv")
    d = py.merge(rr, on="key").set_index("key")
    scores = pd.read_parquet(MEAS / "limitation_scores.parquet")
    keep = scores["h_plim3cc_grm"].notna()
    yn = lambda v: "$\\surd$" if v in ("Yes", True) else "$\\times$"
    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lrrrrrccrrcc@{}}", "\\toprule",
           " & Wave-2 & Floor & Distinct & \\multicolumn{4}{c}{$h$ scale} & \\multicolumn{4}{c}{$\\theta$ scale} \\\\",
           "\\cmidrule(lr){5-8}\\cmidrule(l){9-12}",
           "Bank & kept & 60--84 & 85--90 & Cost $t$ & Var. & C3 & C2 & Cost $t$ & Var. & C3 & C2 \\\\", "\\midrule"]
    for bi, bank in enumerate(CON_BANKS):
        if bi:
            out.append("\\addlinespace")
        slug = SLUG[bank]
        h, t = d.loc[f"{bank} h"], d.loc[f"{bank} theta"]
        out.append(f"{bank} & {100 * h['share_wave2_entrants_kept']:.0f}\\% & {100 * h['worst_60_84']:.2f}\\% & "
                   f"{int(h['distinct_bottom_decile_85_90'])} & {h['cost_curv_capped_t']:.1f} & {h['var_growth_60_85']:.2f} & "
                   f"{yn(h['c3_signpath'])} & {yn(h['c2_signpath'])} & {t['cost_curv_capped_t']:.1f} & {t['var_growth_60_85']:.2f} & "
                   f"{yn(t['c3_signpath'])} & {yn(t['c2_signpath'])} \\\\")
        c = control_stats(scores, f"h_{slug}_grm", keep)
        p3, _, _ = signs_from_corr(corr, f"h_{slug}_grm_condsample", "case3", "balanced")
        p2, _, _ = signs_from_corr(corr, f"h_{slug}_grm_condsample", "case2", "pairwise")
        out.append(f"\\quad same people & 5\\% & {100 * c['worst_60_84']:.2f}\\% & {c['distinct_bottom_decile_85_90']} & "
                   f"& {c['var_growth_60_85']:.2f} & {yn(p3)} & {yn(p2)} & & & & \\\\")
        for code in CODES:
            h, t = d.loc[f"{bank}+{code} h"], d.loc[f"{bank}+{code} theta"]
            out.append(f"\\quad +{code} & {100 * h['share_wave2_entrants_kept']:.0f}\\% & {100 * h['worst_60_84']:.2f}\\% & "
                       f"{int(h['distinct_bottom_decile_85_90'])} & {h['cost_curv_capped_t']:.1f} & {h['var_growth_60_85']:.2f} & "
                       f"{yn(h['c3_signpath'])} & {yn(h['c2_signpath'])} & {t['cost_curv_capped_t']:.1f} & {t['var_growth_60_85']:.2f} & "
                       f"{yn(t['c3_signpath'])} & {yn(t['c2_signpath'])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "con_criteria.tex").write_text("\n".join(out) + "\n")


def tex_con_ident(corr):
    rr = pd.read_csv(OUT / "master_r.csv").set_index("key")
    combos = [("case2", "balanced", "c2b"), ("case3", "balanced", "c3"), ("case4", "balanced", "c4b"),
              ("case2", "pairwise", "c2"), ("case3", "pairwise", "c3p"), ("case4", "pairwise", "c4p")]

    def cell_corr(meas, spec, how):
        path, ok, old = signs_from_corr(corr, meas, spec, how)
        m = "$\\times$" if not path else ("$\\surd$" if ok else "$\\surd^{\\circ}$")
        return [m, "---" if pd.isna(old) else f"${old:+.2f}$"]

    def cell_master(key, sp):
        r = rr.loc[key]
        m = "$\\times$" if r[f"{sp}_signpath"] != "Yes" else ("$\\surd$" if r[f"{sp}_admissible"] == "Yes" else "$\\surd^{\\circ}$")
        v = r[f"{sp}_corr_75-90"]
        return [m, "---" if pd.isna(v) else f"${v:+.2f}$"]

    out = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
           "\\begin{tabular}{@{}l" + "cr" * 6 + "@{}}", "\\toprule",
           " & \\multicolumn{6}{c}{Balanced moments} & \\multicolumn{6}{c}{Pairwise moments} \\\\",
           "\\cmidrule(lr){2-7}\\cmidrule(l){8-13}",
           " & \\multicolumn{2}{c}{Case 2} & \\multicolumn{2}{c}{Case 3} & \\multicolumn{2}{c}{Case 4} & "
           "\\multicolumn{2}{c}{Case 2} & \\multicolumn{2}{c}{Case 3} & \\multicolumn{2}{c}{Case 4} \\\\",
           "Bank" + " & path & 75--90" * 6 + " \\\\", "\\midrule", "\\multicolumn{13}{@{}l}{\\emph{$h$ scale}} \\\\"]
    for bank in CON_BANKS:
        slug = SLUG[bank]
        for meas, lab in ((f"h_{slug}_grm_condsample", f"{bank}, same people"), (f"h_{slug}cc_grm", f"\\quad +CC"),
                          (f"h_{slug}cg_grm", f"\\quad +CG")):
            cells = sum((cell_corr(meas, sp, how) for sp, how, _ in combos), [])
            out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    out += ["\\midrule", "\\multicolumn{13}{@{}l}{\\emph{$\\theta$ scale}} \\\\"]
    for bank in CON_BANKS:
        for code in CODES:
            cells = sum((cell_master(f"{bank}+{code} theta", sp) for _, _, sp in combos), [])
            out.append(f"{bank}+{code} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (TAB / "con_ident.tex").write_text("\n".join(out) + "\n")


def main() -> int:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    prof = pd.read_csv(OUT / "limitation_profiles.csv")
    corr = pd.read_csv(OUT / "limitation_corr_band.csv")
    dec = pd.read_csv(OUT / "limitation_decomposition.csv")
    traj = pd.read_csv(OUT / "limitation_cluster_trajectories.csv")
    comp = pd.read_csv(OUT / "limitation_cluster_composition.csv")
    agree = pd.read_csv(OUT / "limitation_cluster_agreement.csv")
    labels = pd.read_parquet(OUT / "limitation_cluster_labels.parquet")
    for bank in BANKS:
        s = SLUG[bank]
        g, p = f"h_{s}_grm", f"h_{s}_gpcm"
        exhibit1([(g, f"{bank}, graded response", "-"), (p, f"{bank}, partial credit", "--")],
                 None if bank == "P-FUNC" else ("h_pfunc_grm", "P-FUNC, graded response"),
                 prof, corr, f"fig_lim_exhibit1_{s}.png")
        bars(dec, [("case2", "pairwise", g, MODEL_LAB["grm"]), ("case2", "pairwise", p, MODEL_LAB["gpcm"]),
                   ("case3", "balanced", g, MODEL_LAB["grm"]), ("case3", "balanced", p, MODEL_LAB["gpcm"])],
             f"fig_lim_bars_{s}.png", 9.8)
        bars(dec, [("case4", "balanced", g, MODEL_LAB["grm"]), ("case4", "balanced", p, MODEL_LAB["gpcm"])],
             f"fig_lim_bars_case4_{s}.png", 5.4)
        clusters([(g, "Graded response"), (p, "Partial credit")], traj, comp, labels, f"fig_lim_clusters_{s}.png")
    for bank in CON_BANKS:
        s = SLUG[bank]
        cc, cg = f"h_{s}cc_grm", f"h_{s}cg_grm"
        exhibit1([(cc, f"{bank}+CC, chronic count", "-"), (cg, f"{bank}+CG, chronic groups", "--")],
                 (f"h_{s}_grm_condsample", f"{bank}, same people, no conditions"), prof, corr, f"fig_con_exhibit1_{s}.png")
        bars(dec, [("case2", "pairwise", cc, "+CC, count"), ("case2", "pairwise", cg, "+CG, groups"),
                   ("case3", "balanced", cc, "+CC, count"), ("case3", "balanced", cg, "+CG, groups")],
             f"fig_con_bars_{s}.png", 9.8)
        bars(dec, [("case4", "balanced", cc, "+CC, count"), ("case4", "balanced", cg, "+CG, groups")],
             f"fig_con_bars_case4_{s}.png", 5.4)
        clusters([(cc, "Chronic count"), (cg, "Chronic groups")], traj, comp, labels, f"fig_con_clusters_{s}.png")
        print(f"figures written for {bank}")
    W, r2 = weights()
    tex_weights(W, r2)
    tex_fit(agree)
    tex_categories()
    tex_criteria()
    tex_ident()
    Wc, r2c = weights_con()
    pd.concat([W, Wc]).to_csv(OUT / "limitation_weights.csv", index=False)
    tex_con_weights(Wc, r2c)
    tex_con_categories()
    tex_con_fit()
    tex_con_criteria(corr)
    tex_con_ident(corr)
    lam = W.pivot_table(index=["bank", "item"], columns="method", values="value")
    for bank in BANKS:
        x = lam.loc[bank]
        print(f"{bank:9s} corr(GRM loading, polychoric loading) = "
              f"{np.corrcoef(x['grm_loading'], x['polychoric_loading'])[0, 1]:.3f}; "
              f"projection R2 GRM {r2[(bank, 'grm')]:.4f}, GPCM {r2[(bank, 'gpcm')]:.4f}")
    print("tables written:", sorted(p.name for p in TAB.glob("lim_*.tex")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
