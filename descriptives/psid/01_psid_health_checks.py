"""First checks of the PSID physical health measures against the UKHLS ones in the paper.

PSID side: the measures from data_cleaning/psid/02_build_psid_health.py and 03_build_psid_frailty.py,
and the panel for deaths and for who answered. UKHLS side: only aggregate files that the UKHLS scripts
already write, read and never written:
    data/processed/measures/health_measure_items.csv, health_measure_age_profile.csv, health_measure_q3.csv
    artifacts/descriptives/paper_exhibit1_moments.csv, paper_exhibit1_moments_measures.csv (the smoothed
        profiles behind Exhibit 1), ceiling_floor.csv, cost_mortality_age_tests.csv,
        cost_mortality_age_measures_tests.csv
No UKHLS microdata is read.

Age profiles are computed as Exhibit 1 computes them (_paper_common.profile_rows): the mean and variance
at each single year of age, cells under 100 person-waves dropped, then a ten-year centred rolling mean.
The mortality gradient is the one in 37_paper_cost_mortality_age.py: a logit of death before the next
wave on the measure standardised over the sample at risk, (age - 55)/10, its square and the measure's
interaction with age, read at 55. PSID waves are two years apart, so its death is within two years where
the UKHLS one is within one; under proportional hazards the slope per sd is comparable.

Outputs:
    descriptives/psid/figures/psid_fig_measure.png          information, the theta distribution, latent profile
    descriptives/psid/figures/psid_fig_moments.png          mean and variance by age on four measures
    descriptives/psid/figures/psid_fig_waves_mortality.png  by wave within age bands; two-year death against theta
    artifacts/descriptives/psid/psid_health_checks.csv      every number printed, PSID beside UKHLS

    python descriptives/psid/01_psid_health_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # descriptives/, for the shared style
from _style import BLUE, INK2, INK3, VERM, apply_style  # noqa: E402

from prevention_health_clustering.measures.grm import GRMFit, test_information  # noqa: E402

PSID, UK_MEAS, UK_ART = Path("data/processed/psid"), Path("data/processed/measures"), Path("artifacts/descriptives")
MEAS = PSID / "measures"
FIG, OUT = Path("descriptives/psid/figures"), Path("artifacts/descriptives/psid")
WINDOW, MIN_CELL = 10, 100
VARIANTS = {"theta": r"$\theta$", "h": "$h$", "frailty": "frailty index", "logfrailty": "log frailty"}
BANDS = [(25, 44, "25-44"), (45, 64, "45-64"), (65, 84, "65-84")]
BAND_COL = ["#6baed6", "#3182bd", "#08306b"]                       # the repo's blues, young to old
MORT_BANDS = [(45, 64, "45-64"), (65, 90, "65-90")]
EDGES_MORT = np.arange(-3.0, 2.0001, 0.5)
BREAKS = {2005: "conditions\nrenumbered", 2013: "ADLs skipped\nfor the healthy", 2021: "web\ninterviews"}
TH = np.linspace(-6, 6, 121)
rows: list[dict] = []


def record(section, statistic, psid, ukhls=np.nan, published=np.nan, note=""):
    rows.append({"section": section, "statistic": statistic, "psid": psid, "ukhls": ukhls,
                 "published": published, "note": note})


def smooth_profile(d: pd.DataFrame, v: str, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """The profile half of _paper_common.profile_rows."""
    prof = d.groupby("age")[v].agg(["mean", "var", "count"])
    prof = prof[prof["count"] >= min_cell]
    return prof[["mean", "var"]].rolling(WINDOW, center=True).mean().dropna()


def profile_stats(p: pd.DataFrame) -> dict[str, float]:
    m, v, last = p["mean"], p["var"], int(p.index.max())
    m, v = m.reindex(range(20, 91)), v.reindex(range(20, 91))      # an age the profile lacks gives NaN
    return {"change a year, 30-50": (m[50] - m[30]) / 20, "change a year, 50-70": (m[70] - m[50]) / 20,
            f"change a year, 70-{last}": (m[last] - m[70]) / (last - 70),
            "variance 60 / 30": v[60] / v[30], "variance 85 / 60": v[85] / v[60],
            "age of peak variance": float(v.loc[25:86].idxmax())}


def fit_from_items(df: pd.DataFrame) -> GRMFit:
    items, ncat = {}, {}
    for _, r in df.iterrows():
        k = int(r["categories"])
        items[r["item"]] = (float(r["a"]), np.array([r[f"b{i}"] for i in range(1, k)], float))
        ncat[r["item"]] = k
    return GRMFit(items=items, ncat=ncat, th=TH, log_likelihood=np.nan, mu=np.zeros(1), sigma=np.ones(1),
                  n_iter=0, n_parameters=0)


def logit(X, y, iters=60):
    """37_paper_cost_mortality_age.logit."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    return b, np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def main() -> int:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    hm = pd.read_parquet(MEAS / "psid_health_measure.parquet")
    fr = pd.read_parquet(MEAS / "psid_frailty_index.parquet")
    panel = pd.read_parquet(PSID / "psid_panel.parquet", columns=["pid", "wave", "died_by_next_wave"])
    d = (hm.merge(fr[["pid", "wave", "frailty", "log_frailty"]], on=["pid", "wave"], how="left")
         .rename(columns={"log_frailty": "logfrailty"}).merge(panel, on=["pid", "wave"], how="left"))
    print(f"PSID: {len(d):,} person-waves with theta, {d['pid'].nunique():,} people, "
          f"{d['frailty'].notna().mean():.1%} also with the frailty index")

    # ---------------- 1. the measure itself -----------------------------------
    it_p = pd.read_csv(MEAS / "psid_health_measure_items.csv")
    it_6 = pd.read_csv(MEAS / "psid_health_measure_items_2017.csv")
    it_u = pd.read_csv(UK_MEAS / "health_measure_items.csv")
    print("\nitem parameters (a, and the range of the thresholds), PSID against UKHLS:")
    for item in it_p["item"]:
        p, u = it_p.set_index("item").loc[item], it_u.set_index("item").loc[item]
        bp = p.filter(like="b").dropna().to_numpy(float)
        bu = u.filter(like="b").dropna().to_numpy(float)
        print(f"  {item:9s} a {p['a']:4.2f} vs {u['a']:4.2f}   thresholds {bp.min():5.2f} to {bp.max():5.2f} "
              f"vs {bu.min():5.2f} to {bu.max():5.2f}   categories {int(p['categories'])} vs {int(u['categories'])}")
        record("items", f"{item}: discrimination", p["a"], u["a"])
        record("items", f"{item}: lowest threshold", bp.min(), bu.min())
        record("items", f"{item}: highest threshold", bp.max(), bu.max())
    fits = {"PSID, five testlets": fit_from_items(it_p), "PSID 2017-2023, six testlets": fit_from_items(it_6),
            "UKHLS, eight testlets": fit_from_items(it_u)}
    info = {k: test_information(f, TH).sum(axis=1) for k, f in fits.items()}
    phi = norm.pdf(TH) / norm.pdf(TH).sum()
    for k, i in info.items():          # 1 - E[1 / (1 + I)]: the usual marginal-reliability approximation
        rel = 1 - (phi / (1 + i)).sum()
        print(f"  {k}: information peaks at theta {TH[i.argmax()]:+.1f} ({i.max():.1f}); "
              f"marginal reliability about {rel:.2f}")
        record("measure", f"marginal reliability (from the item parameters), {k}",
               rel if k.startswith("PSID") else np.nan, rel if k.startswith("UKHLS") else np.nan)
    ceiling = d["theta"] == d["theta"].max()
    cf = pd.read_csv(UK_ART / "ceiling_floor.csv").set_index("measure").loc["theta_phys_full"]
    print(f"ceiling: {ceiling.mean():.1%} of PSID person-waves give the best answer on every testlet (theta "
          f"{d['theta'].max():.2f}, posterior sd {d.loc[ceiling, 'theta_sd'].iloc[0]:.2f}); UKHLS {cf['ceiling_pct']:.1f}% "
          f"(posterior sd {cf['psd_at_ceiling']:.2f}); {d['theta'].nunique()} distinct PSID scores")
    record("measure", "share at the ceiling", ceiling.mean(), cf["ceiling_pct"] / 100)
    record("measure", "posterior sd at the ceiling", d.loc[ceiling, "theta_sd"].iloc[0], cf["psd_at_ceiling"])
    record("measure", "share at the floor", (d["theta"] == d["theta"].min()).mean(), cf["floor_pct"] / 100)
    record("measure", "distinct scores", d["theta"].nunique(), note="UKHLS: 28,360 (health_measure_properties.csv)")
    identity = d["theta"].var(ddof=0) + (d["theta_sd"] ** 2).mean()
    record("measure", "EAP identity, var(theta) + mean posterior variance", identity, published=1.0)
    record("measure", "variance of theta (EAP), pooled", d["theta"].var(ddof=0))
    q3p = pd.read_csv(MEAS / "psid_health_measure_q3.csv")
    q3u = pd.read_csv(UK_MEAS / "health_measure_q3.csv")
    record("measure", "largest Q3 between testlets", q3p["q3"].max(), q3u["q3"].max(), note="gate: under 0.2")
    print(f"largest Q3 {q3p['q3'].max():+.3f} (UKHLS {q3u['q3'].max():+.3f}); EAP identity {identity:.3f}")

    lat_p = pd.read_csv(MEAS / "psid_health_measure_age_profile.csv").set_index("age")
    lat_u = pd.read_csv(UK_MEAS / "health_measure_age_profile.csv").set_index("age")
    for lab, (a0, a1) in {"30-50": (30, 50), "50-70": (50, 70), "70-90": (70, 90)}.items():
        record("latent profile", f"mu_a, change a year, {lab}", (lat_p.loc[a1, "mu"] - lat_p.loc[a0, "mu"]) / (a1 - a0),
               (lat_u.loc[a1, "mu"] - lat_u.loc[a0, "mu"]) / (a1 - a0))
    for a0, a1 in ((30, 60), (60, 85)):
        record("latent profile", f"sigma_a, {a1} / {a0}", lat_p.loc[a1, "sigma"] / lat_p.loc[a0, "sigma"],
               lat_u.loc[a1, "sigma"] / lat_u.loc[a0, "sigma"])
    record("latent profile", "age of the largest sigma_a", float(lat_p["sigma"].idxmax()), float(lat_u["sigma"].idxmax()))
    print(f"latent profile: mu(85) - mu(30) {lat_p.loc[85, 'mu'] - lat_p.loc[30, 'mu']:+.2f} against UKHLS "
          f"{lat_u.loc[85, 'mu'] - lat_u.loc[30, 'mu']:+.2f}; sigma 60/30 {lat_p.loc[60, 'sigma'] / lat_p.loc[30, 'sigma']:.2f} "
          f"({lat_u.loc[60, 'sigma'] / lat_u.loc[30, 'sigma']:.2f}), 85/60 {lat_p.loc[85, 'sigma'] / lat_p.loc[60, 'sigma']:.2f} "
          f"({lat_u.loc[85, 'sigma'] / lat_u.loc[60, 'sigma']:.2f}); sigma peaks at {lat_p['sigma'].idxmax()} against "
          f"{lat_u['sigma'].idxmax()}; pooled variance of the PSID EAP {d['theta'].var(ddof=0):.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    styles = {"PSID, five testlets": (BLUE, "-"), "PSID 2017-2023, six testlets": (BLUE, ":"),
              "UKHLS, eight testlets": (VERM, "--")}
    for k, i in info.items():
        ax[0, 0].plot(TH, i, color=styles[k][0], ls=styles[k][1], lw=1.8, label=k)
    ax[0, 0].set_xlim(-4, 2.5); ax[0, 0].set_xlabel(r"$\theta$"); ax[0, 0].set_ylabel("test information")
    ax[0, 0].set_title("(a) where each measure is precise", loc="left"); ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.52, 1.0))
    ax[0, 0].grid(True, axis="y")
    ax[0, 1].hist(d["theta"], bins=np.arange(-4.0, 1.61, 0.1), color=BLUE, alpha=0.85, rwidth=0.85)
    ax[0, 1].set_yticks(ax[0, 1].get_yticks()); ax[0, 1].set_yticklabels([f"{t / 1000:.0f}k" for t in ax[0, 1].get_yticks()])
    ymax = ax[0, 1].get_ylim()[1]
    ax[0, 1].annotate(f"{ceiling.mean():.1%} give the best answer on every\ntestlet, the ceiling (UKHLS {cf['ceiling_pct']:.1f}%)",
                      xy=(d["theta"].max(), ceiling.sum()), xytext=(-3.9, 0.82 * ymax), va="center",
                      fontsize=8.5, color=INK2, arrowprops={"arrowstyle": "-", "color": INK3, "lw": 0.8,
                                                            "connectionstyle": "angle,angleA=0,angleB=90"})
    ax[0, 1].set_xlabel(r"$\theta$, PSID"); ax[0, 1].set_ylabel("person-waves")
    ax[0, 1].set_title(f"(b) PSID scores: {d['theta'].nunique()} distinct values", loc="left")
    for j, (col, lab) in enumerate((("mu", r"(c) latent mean $\mu_a$"), ("sigma", r"(d) latent sd $\sigma_a$"))):
        a = ax[1, j]
        a.plot(lat_p.index, lat_p[col], color=BLUE, lw=1.8, label="PSID")
        a.plot(lat_u.index, lat_u[col], color=VERM, lw=1.8, ls="--", label="UKHLS")
        a.set_title(lab + ", multigroup by single year of age", loc="left"); a.set_xlabel("age")
        a.grid(True, axis="y"); a.set_xlim(20, 90)
        if j == 0:
            a.legend(loc="lower left")
    fig.text(0.01, -0.01, "PSID: Reference Persons and Spouses, 2003-2023, ages 20-90, five testlets (general health, "
             "work limitation, mobility, personal care, the nine conditions); the dotted line adds the 2017-2023 walking "
             "and lifting items. UKHLS: the paper's eight testlets, read from health_measure_items.csv and "
             "health_measure_age_profile.csv. Each latent scale is N(0, 1) in its own pooled sample.",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "psid_fig_measure.png")
    plt.close(fig)

    # ---------------- 2. moments by age ----------------------------------------
    uk = pd.concat([pd.read_csv(UK_ART / "paper_exhibit1_moments.csv"),
                    pd.read_csv(UK_ART / "paper_exhibit1_moments_measures.csv")])
    uk = uk[uk["kind"] == "profile"].assign(age=lambda x: x["age"].astype(int))
    published = {"theta": {"change a year, 30-50": -0.014, "change a year, 50-70": -0.020,
                           "change a year, 70-86": -0.038, "variance 60 / 30": 1.62, "variance 85 / 60": 0.78,
                           "age of peak variance": 62},
                 "logfrailty": {"variance 60 / 30": 1.70, "variance 85 / 60": 0.70}}
    prof = {}
    print("\nmoments by age (ten-year centred rolling means), PSID against UKHLS:")
    for v in VARIANTS:
        sp = smooth_profile(d[["age", v]].dropna(), v)
        su = uk[uk["variant"] == v].set_index("age")[["mean", "var"]]
        prof[v] = (sp, su)
        stp, stu = profile_stats(sp), profile_stats(su)
        for k in stp:
            record(f"moments: {v}", k, stp[k], stu.get(k, np.nan), published.get(v, {}).get(k, np.nan))
        print(f"  {v:10s} " + "   ".join(f"{k.replace('change a year, ', 'd/yr ')} {stp[k]:+.4f} ({stu.get(k, np.nan):+.4f})"
                                        if "change" in k else f"{k} {stp[k]:.2f} ({stu.get(k, np.nan):.2f})"
                                        for k in stp))
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.6), gridspec_kw={"hspace": 0.38, "wspace": 0.28})
    for j, (v, lab) in enumerate(VARIANTS.items()):
        sp, su = prof[v]
        ax[0, j].plot(sp.index, sp["mean"], color=BLUE, lw=1.8, label="PSID")
        ax[0, j].plot(su.index, su["mean"], color=VERM, lw=1.8, ls="--", label="UKHLS")
        ax[0, j].set_title(f"({'abcd'[j]}) {lab}: mean", loc="left")
        ax[1, j].plot(sp.index, sp["var"] / sp.loc[60, "var"], color=BLUE, lw=1.8)
        ax[1, j].plot(su.index, su["var"] / su.loc[60, "var"], color=VERM, lw=1.8, ls="--")
        ax[1, j].axhline(1, color=INK3, lw=0.7, ls=":")
        ax[1, j].set_title(f"({'efgh'[j]}) variance, relative to age 60", loc="left")
        ax[1, j].set_xlabel("age")
        for a in ax[:, j]:
            a.grid(True, axis="y"); a.set_xlim(20, 90)
    ax[0, 0].legend(loc="lower left")
    fig.text(0.01, -0.01, "Mean and variance at each single year of age, cells under 100 person-waves dropped, then "
             "ten-year centred rolling means, as in Exhibit 1. The variance row divides each profile by its value at 60, so "
             "the shapes can be compared. The variance at 60, PSID against UKHLS: " + "; ".join(
                 f"{VARIANTS[v].replace('$', '').replace(chr(92), '')} {prof[v][0].loc[60, 'var']:.3g} and "
                 f"{prof[v][1].loc[60, 'var']:.3g}" for v in VARIANTS) + ". PSID: 2003-2023, ages 20-90; frailty "
             "is the 28-deficit index of Hosseini, Kopecky and Zhao (2022) and log frailty log(frailty + 1/28). UKHLS: the "
             "paper's profiles (paper_exhibit1_moments.csv), frailty on 31 deficits and log(frailty + 1/31).",
             fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "psid_fig_moments.png")
    plt.close(fig)

    # ---------------- 3. rank agreement -----------------------------------------
    both = d.dropna(subset=["frailty"])
    r = spearmanr(both["frailty"], both["theta"]).statistic        # the same for h and for log frailty: ranks
    record("rank agreement", "Spearman, frailty index with theta (magnitude; the same for h and log frailty)",
           abs(r), published=0.92)
    print(f"\nSpearman, frailty with theta {r:+.3f} (the same for h and log frailty, which keep the ranks); "
          f"UKHLS 0.92 in magnitude")
    for lo, hi, name in BANDS:
        bb = both[both["age"].between(lo, hi)]
        rb = spearmanr(bb["frailty"], bb["theta"]).statistic
        record("rank agreement", f"Spearman, frailty index with theta, ages {name} (magnitude)", abs(rb))
        print(f"  ages {name}: {rb:+.3f}")
    s6 = pd.read_parquet(MEAS / "psid_health_measure_2017.parquet").merge(hm[["pid", "wave", "age"]], on=["pid", "wave"])
    r6p, r6s = np.corrcoef(s6["theta"], s6["theta6"])[0, 1], spearmanr(s6["theta"], s6["theta6"]).statistic
    record("six testlets, 2017-2023", "Pearson, five-testlet with six-testlet theta", r6p)
    record("six testlets, 2017-2023", "Spearman, five-testlet with six-testlet theta", r6s)
    record("six testlets, 2017-2023", "share at the ceiling, six testlets", (s6["theta6"] == s6["theta6"].max()).mean())
    record("six testlets, 2017-2023", "share at the ceiling, five testlets, same rows", (s6["theta"] == s6["theta"].max()).mean())
    for v in ("theta", "theta6"):                                  # four waves only: cells of 50 or more
        st = profile_stats(smooth_profile(s6[["age", v]], v, min_cell=50))
        for k in ("variance 60 / 30", "variance 85 / 60", "age of peak variance"):
            record("six testlets, 2017-2023", f"{k}, {'six' if v == 'theta6' else 'five'} testlets", st[k])
    print(f"six testlets on 2017-2023: Pearson {r6p:.3f}, Spearman {r6s:.3f} with the five-testlet theta")

    # ---------------- 4. who answered -------------------------------------------
    sr = d.dropna(subset=["self_report"])
    print("\nself-reports against answers given by someone else in the family (usually the partner):")
    for lo, hi, name in BANDS:
        s = sr[sr["age"].between(lo, hi)]
        g = s.groupby("self_report")["theta"]
        top = s.assign(c=s["theta"] == d["theta"].max()).groupby("self_report")["c"].mean()
        record("who answered", f"mean theta, ages {name}, self minus other", g.mean()[1] - g.mean()[0])
        record("who answered", f"share at the ceiling, ages {name}, self minus other", top[1] - top[0])
        print(f"  {name}: mean theta self {g.mean()[1]:+.3f}, other {g.mean()[0]:+.3f}; at the ceiling "
              f"{top[1]:.1%} and {top[0]:.1%}; {(s['self_report'] == 0).mean():.0%} answered by someone else")
    r = sr["theta"] - sr.groupby("age")["theta"].transform("mean")
    switch = sr.groupby("pid")["self_report"].transform(lambda x: x.nunique() > 1)
    rr, pp = r[switch], sr.loc[switch, "self_report"]
    rd, pd_ = rr - rr.groupby(sr.loc[switch, "pid"]).transform("mean"), pp - pp.groupby(sr.loc[switch, "pid"]).transform("mean")
    fe = (rd * pd_).sum() / (pd_ ** 2).sum()
    record("who answered", "within-person difference in theta, self minus other (age-adjusted)", fe,
           note=f"{sr.loc[switch, 'pid'].nunique():,} people answered by themselves in some waves and by someone else in others")
    print(f"  within the {sr.loc[switch, 'pid'].nunique():,} people seen both ways, a self-report is "
          f"{fe:+.3f} sd of theta away from the same person's other-reported waves, after age")

    # ---------------- 5. by wave within age bands -------------------------------
    by_wave = {}
    for v in ("theta", "frailty"):
        t = (d.dropna(subset=[v]).assign(band=pd.cut(d["age"], [lo - 1 for lo, _, _ in BANDS] + [BANDS[-1][1]],
                                                     labels=[b for _, _, b in BANDS]))
             .dropna(subset=["band"]).groupby(["band", "wave"], observed=True)[v].agg(["mean", "sem"]))
        by_wave[v] = t
        for _, _, b in BANDS:
            w = t.loc[b, "mean"]
            record("by wave", f"mean {v}, ages {b}, 2023 minus 2003", w[2023] - w[2003])
            record("by wave", f"mean {v}, ages {b}, largest move between adjacent waves",
                   w.diff().abs().max(), note=f"at {int(w.diff().abs().idxmax())}")
        print(f"mean {v} by wave within age bands:\n" + t["mean"].unstack("wave").round(3).to_string())

    # ---------------- 6. mortality ------------------------------------------------
    m = d.dropna(subset=["died_by_next_wave"]).reset_index(drop=True)
    tests_u = pd.concat([pd.read_csv(UK_ART / "cost_mortality_age_tests.csv", index_col=0),
                         pd.read_csv(UK_ART / "cost_mortality_age_measures_tests.csv", index_col=0)])
    print(f"\nmortality: {len(m):,} person-waves at risk 2003-2021, {int(m['died_by_next_wave'].sum()):,} deaths "
          f"within two years")
    a = (m["age"].to_numpy() - 55) / 10
    for v in VARIANTS:
        mv = m.dropna(subset=[v])
        av = (mv["age"].to_numpy() - 55) / 10
        sgn = -1.0 if v in ("theta", "h") else 1.0
        z = ((mv[v] - mv[v].mean()) / mv[v].std()).to_numpy()
        b, V = logit(np.column_stack([np.ones(len(mv)), z, av, av ** 2, z * av]), mv["died_by_next_wave"].to_numpy(float))
        record("mortality", f"log-odds of death per sd of {v}, at 55 (sign: toward worse health)", sgn * b[1],
               tests_u.loc[v, "mort_slope"], 0.90 if v == "theta" else np.nan,
               note="PSID: death within two years; UKHLS: within one")
        record("mortality", f"interaction with (age - 55)/10, {v}", sgn * b[4], tests_u.loc[v, "mort_x_age"],
               note=f"PSID t {sgn * b[4] / np.sqrt(V[4, 4]):.1f}, UKHLS t {tests_u.loc[v, 'mort_x_age_t']:.1f}")
        print(f"  {v:10s} {sgn * b[1]:.2f} per sd at 55 (UKHLS {tests_u.loc[v, 'mort_slope']:.2f}); interaction "
              f"{sgn * b[4]:+.3f} (UKHLS {tests_u.loc[v, 'mort_x_age']:+.3f})")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"wspace": 0.27})
    for j, (v, lab) in enumerate((("theta", r"mean $\theta$"), ("frailty", "mean frailty index"))):
        t = by_wave[v]
        for (lo, hi, b), col in zip(BANDS, BAND_COL):
            w = t.loc[b]
            ax[j].errorbar(w.index, w["mean"], yerr=1.96 * w["sem"], color=col, lw=1.6, marker="o", ms=3.5,
                           elinewidth=0.8, capsize=0)
            ax[j].annotate(f"ages {b}", xy=(w.index[-1], w["mean"].iloc[-1]), xytext=(6, 0),
                           textcoords="offset points", va="center", fontsize=8, color=INK2)
        for yr, txt in BREAKS.items():
            ax[j].axvline(yr, color=INK3, lw=0.7, ls=":")
            ax[j].annotate(txt, xy=(yr, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                           textcoords="offset points", va="top", fontsize=6.8, color=INK3)
        lo_, hi_ = ax[j].get_ylim()
        ax[j].set_ylim(lo_, hi_ + 0.2 * (hi_ - lo_))                # headroom for the labels of the breaks
        ax[j].set_xlim(2002, 2027.5); ax[j].set_xticks(range(2003, 2024, 4)); ax[j].grid(True, axis="y")
        ax[j].set_title(f"({'ab'[j]}) {lab} by wave, within age bands", loc="left"); ax[j].set_xlabel("wave")
    mu, sd = m["theta"].mean(), m["theta"].std()
    for (lo, hi, name), col in zip(MORT_BANDS, BAND_COL[1:]):
        s = m[m["age"].between(lo, hi)].assign(bin=lambda x: pd.cut(x["theta"], EDGES_MORT))
        g = s.groupby("bin", observed=True).agg(x=("theta", "mean"), k=("died_by_next_wave", "sum"),
                                                n=("died_by_next_wave", "size"))
        g = g[g["k"] >= 10]
        lo_ci, hi_ci = wilson(g["k"], g["n"])
        rate = g["k"] / g["n"]
        ax[2].errorbar(g["x"], rate, yerr=[rate - lo_ci, hi_ci - rate], fmt="o", ms=4, color=col, elinewidth=0.8,
                       capsize=0, label=f"ages {name}")
        z = ((s["theta"] - mu) / sd).to_numpy()
        aa = (s["age"].to_numpy() - 55) / 10
        b, _ = logit(np.column_stack([np.ones(len(s)), z, aa, aa ** 2]), s["died_by_next_wave"].to_numpy(float))
        xs = np.linspace(g["x"].min(), g["x"].max(), 60)
        am = (s["age"].mean() - 55) / 10
        ax[2].plot(xs, 1 / (1 + np.exp(-(b[0] + b[1] * (xs - mu) / sd + b[2] * am + b[3] * am ** 2))), color=col, lw=1.4)
    ax[2].set_yscale("log"); ax[2].grid(True, axis="y"); ax[2].legend(loc="upper right")
    ax[2].set_title(r"(c) death within two years against $\theta$", loc="left")
    ax[2].set_xlabel(r"$\theta$, equal-width bins (right = healthier)"); ax[2].set_ylabel("death before the next wave (log scale)")
    fig.text(0.01, -0.03, "PSID, Reference Persons and Spouses. (a), (b): means with 95% intervals; dotted lines mark "
             "the waves where the questions or the mode changed (see PSID_CROSSWALK.md). The condition count is "
             "cumulated (once reported, always counted), so it drifts up the longer a person is in the panel. "
             "(c): 2003-2021, bins with at least 10 deaths, Wilson 95% intervals; lines are the within-band logit, "
             "linear in theta and quadratic in age, at the band's mean age.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "psid_fig_waves_mortality.png")
    plt.close(fig)

    # ---------------- 7. Hosseini, Kopecky and Zhao ------------------------------
    hk = pd.read_csv(MEAS / "psid_frailty_hkz_benchmark.csv")
    for _, r in hk.iterrows():
        record("frailty: Hosseini, Kopecky and Zhao (2022), Table 1", r["statistic"], r["psid"], published=r["hkz"],
               note="their sample: heads and spouses 25+, 2003-2017")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "psid_health_checks.csv", index=False)
    print(f"\nwrote {OUT / 'psid_health_checks.csv'} ({len(out)} rows) and three figures in {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
