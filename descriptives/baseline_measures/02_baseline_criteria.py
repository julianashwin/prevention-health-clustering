"""Criteria table for the three linear measures and their graded-response twins, both rulers."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path('/Users/julianashwin/Documents/GitHub/prevention-health-clustering')
SCR = R / 'data/processed/baseline_measures'  # intermediate data, gitignored
cb = pd.read_parquet(SCR / 'baseline_candidates.parquet')
mp = pd.read_parquet(R / 'data/processed/measures/measure_panel.parquet',
                     columns=['pidp', 'wave', 'grm_theta', 'grmh', 'theta_phys_full', 'grmh_phys_full'])
d = cb.merge(mp, on=['pidp', 'wave'], how='left').rename(columns={
    'grm_theta': 'GRM theta', 'grmh': 'GRM h', 'theta_phys_full': 'P-FULL theta', 'grmh_phys_full': 'P-FULL h'})
M = ['PHYS-4', 'PHYS+F', 'PHYS+F+C', 'GRM h', 'GRM theta', 'P-FUNC h', 'P-FUNC theta', 'P-FULL h', 'P-FULL theta']
d[['pidp', 'wave', 'age'] + M].to_parquet(SCR / 'criteria_measures.parquet', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

# ---- 1. coverage, distribution, ranking ------------------------------------------
first = d.groupby('pidp')['wave'].transform('min'); bhps = first == 2
ref = d['PHYS-4'].notna()
rows = []
for m in M:
    x = d[m]; ok = x.notna(); best, worst = x[ok].max(), x[ok].min()
    y20 = x[ok & d.age.between(20, 39)]; o60 = x[ok & d.age.between(60, 84)]; o85 = x[ok & d.age.between(85, 90)]
    both = ok & ref
    rows.append(dict(measure=m, person_years=int(ok.sum()), kept_vs_phys=round(ok.sum() / ref.sum(), 3),
        bhps_kept=round((ok & bhps).sum() / (ref & bhps).sum(), 3), distinct=int(x[ok].nunique()),
        best_20_39=round(100 * (y20 == best).mean(), 1), worst_60_84=round(100 * (o60 == worst).mean(), 2),
        bottom_decile_distinct_85_90=int(o85[o85 <= o85.quantile(0.10)].nunique()),
        rank_corr_phys=round(d.loc[both, m].rank().corr(d.loc[both, 'PHYS-4'].rank()), 3)))
print('=== 1. coverage, distribution and ranking (ages 20-90, own sample) ===')
print(pd.DataFrame(rows).to_string(index=False))

# ---- 2. age profile -----------------------------------------------------------------
rows = []
for m in M:
    x = d[m]; ok = x.notna(); z = (x - x[ok].mean()) / x[ok].std()
    b = lambda lo, hi: z[ok & d.age.between(lo, hi)]
    rows.append(dict(measure=m,
        decline_per_decade_30_50=round((b(45, 49).mean() - b(30, 34).mean()) / 1.5, 2),
        decline_per_decade_70_85=round((b(85, 89).mean() - b(70, 74).mean()) / 1.5, 2),
        var_growth_60_85=round(b(85, 89).var() / b(60, 64).var(), 2)))
print('\n=== 2. age profile, pooled-sd units ===')
print(pd.DataFrame(rows).to_string(index=False))

# ---- 3. cost convexity, common sample --------------------------------------------
cost = pd.read_parquet(R / 'data/processed/measures/cost_index.parquet', columns=['pidp', 'wave', 'hosp', 'flat_cost_total'])
c = d.merge(cost, on=['pidp', 'wave'])
c = c[c['flat_cost_total'].notna() & c['hosp'].isin([1, 2])]
c = c[c[M].notna().all(axis=1)].reset_index(drop=True)
g = pd.factorize(c['pidp'])[0]
def fit(z, y):
    X = np.column_stack([np.ones(len(z)), z, z**2]); XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y; u = y - X @ b
    S = np.zeros((g.max() + 1, 3)); np.add.at(S, g, X * u[:, None])
    V = XtXi @ (S.T @ S) @ XtXi
    return b[2], b[2] / np.sqrt(V[2, 2])
def deciles(z, y, k=10):
    bb = np.clip((pd.Series(z).rank(pct=True) * k).astype(int), 0, k - 1)
    m = pd.DataFrame({'z': z, 'y': y, 'b': bb}).groupby('b').mean()
    X = np.column_stack([np.ones(k), m.z, m.z**2]); co, *_ = np.linalg.lstsq(X, np.log(m.y.to_numpy() + 1), rcond=None)
    s = np.diff(m.y.to_numpy()) / np.diff(m.z.to_numpy())
    return co[2], m.y.iloc[0] / m.y.iloc[-1], int((np.diff(s) < 0).sum())
y = c['flat_cost_total'].to_numpy(float); yc = np.minimum(y, np.quantile(y, 0.99))
rows = []
for m in M:
    z = ((c[m] - c[m].mean()) / c[m].std()).to_numpy()
    b, t = fit(z, y); bc, tc = fit(z, yc); lq, ratio, viol = deciles(z, y)
    rows.append(dict(measure=m, coef=round(b), t_cluster=round(t, 1), coef_capped=round(bc), t_capped=round(tc, 1),
                     log_curvature=round(lq, 3), cost_ratio=round(ratio, 1), non_convex_steps=viol))
print(f'\n=== 3. cost convexity: common sample {len(c):,} person-waves, {g.max()+1:,} people, flat cost index ===')
print(pd.DataFrame(rows).to_string(index=False))
