"""Candidate baseline physical-health measures: build, then diagnose.

Candidates (all oriented higher = better health):
  PCS        sf12pcs_dv, the survey's own summary (mental items weighted negatively)
  PHYS-4     PCS_phys_only: PF, RP, BP, GH subscales, UK varimax physical loadings
  PHYS-4eq   mean of the same four 0-100 subscales, equal weights
  PHYS+F     z(PHYS-4eq) - z(FUNC count), re-standardised     (full sample)
  PHYS+F+C   z(PHYS-4eq) - z(FUNC count) - z(16-condition count) (needs inventory)
  FI-15      1 - frailty index over 15 non-diagnosis deficits  (full sample)
  FI-31      1 - frailty index over 31 deficits incl. 16 physical diagnoses
  P-FUNC h / theta   the graded-response bank used in testing_metrics
"""
import numpy as np, pandas as pd
from pathlib import Path

R = Path('/Users/julianashwin/Documents/GitHub/prevention-health-clustering')
SCR = R / 'data/processed/baseline_measures'  # intermediate data, gitignored

items = pd.read_parquet(R / 'data/interim/sf12_items_long.parquet')
dis_all = [f'disdif{i}' for i in range(1, 13)] + ['disdif96']
items = items[['pidp', 'wave', 'age', 'sf1', 'sf2a', 'sf2b', 'sf3a', 'sf3b', 'sf5', 'sf6b', 'health'] + dis_all]
mp = pd.read_parquet(R / 'data/processed/measures/measure_panel.parquet',
                     columns=['pidp', 'wave', 'sf12pcs_dv', 'PCS_phys_only', 'sub_PF', 'sub_RP',
                              'sub_BP', 'sub_GH', 'grmh_phys_func', 'theta_phys_func'])
cc = pd.read_parquet(R / 'data/processed/measures/chronic_conditions.parquet',
                     columns=['pidp', 'wave', 'inventory_seen'] + [f'ever_{i}' for i in range(1, 17)])
d = items.merge(mp, on=['pidp', 'wave'], how='left').merge(cc, on=['pidp', 'wave'], how='left')
d = d[d['age'].between(20, 90)].copy()

def valid(s, k):
    return s.where(s.isin(range(1, k + 1)))

sf1, sf2a, sf2b = valid(d.sf1, 5), valid(d.sf2a, 3), valid(d.sf2b, 3)
sf3a, sf3b, sf5 = valid(d.sf3a, 5), valid(d.sf3b, 5), valid(d.sf5, 5)
sf6b = valid(d.sf6b, 5)
health = d.health.where(d.health.isin([1, 2]))

# ---- deficits, 0 = none, 1 = full deficit -------------------------------------
DEF = pd.DataFrame(index=d.index)
DEF['gh'] = (sf1 - 1) / 4
DEF['pf_mod'] = (3 - sf2a) / 2
DEF['pf_stairs'] = (3 - sf2b) / 2
DEF['rp_less'] = (5 - sf3a) / 4
DEF['rp_kind'] = (5 - sf3b) / 4
DEF['pain'] = (sf5 - 1) / 4
DEF['lsi'] = (health == 1).astype(float).where(health.notna())
asked = d[dis_all].notna().any(axis=1)
DIS = {1: 'mobility', 2: 'lifting', 3: 'dexterity', 4: 'continence', 5: 'hearing',
       6: 'sight', 10: 'coordination', 11: 'personal_care'}
for k, nm in DIS.items():
    v = (d[f'disdif{k}'] == 1).astype(float).where((health == 1) & asked)
    DEF[nm] = v.mask(health == 2, 0.0)
SF6 = ['gh', 'pf_mod', 'pf_stairs', 'rp_less', 'rp_kind', 'pain']
NONDX = SF6 + ['lsi'] + list(DIS.values())                     # 15
for i in range(1, 17):
    DEF[f'cond{i}'] = d[f'ever_{i}'].where(d['inventory_seen'] == True)
ALLD = NONDX + [f'cond{i}' for i in range(1, 17)]              # 31

def fi(cols):
    x = DEF[cols]
    return x.mean(axis=1).where(x.notna().all(axis=1))

def z(s):
    return (s - s.mean()) / s.std()

out = d[['pidp', 'wave', 'age']].copy()
out['PCS'] = d['sf12pcs_dv']
out['PHYS-4'] = d['PCS_phys_only']
phys4eq = d[['sub_PF', 'sub_RP', 'sub_BP', 'sub_GH']].mean(axis=1, skipna=False)
out['PHYS-4eq'] = phys4eq
func5 = DEF[['mobility', 'lifting', 'dexterity', 'coordination', 'personal_care']].sum(axis=1, min_count=5)
cc16 = DEF[[f'cond{i}' for i in range(1, 17)]].sum(axis=1, min_count=16)
pf = z(phys4eq) - z(func5)
out['PHYS+F'] = z(pf)
pfc = z(phys4eq) - z(func5) - z(cc16)
out['PHYS+F+C'] = z(pfc)
out['FI-15'] = 1 - fi(NONDX)
out['FI-31'] = 1 - fi(ALLD)
out['P-FUNC h'] = d['grmh_phys_func']
out['P-FUNC theta'] = d['theta_phys_func']
CANDS = ['PCS', 'PHYS-4', 'PHYS-4eq', 'PHYS+F', 'PHYS+F+C', 'FI-15', 'FI-31', 'P-FUNC h', 'P-FUNC theta']
out.to_parquet(SCR / 'baseline_candidates.parquet', index=False)

# ---- 1. coverage and distribution ------------------------------------------------
first = d.groupby('pidp')['wave'].transform('min')
bhps = first == 2
ref_n = out['PHYS-4'].notna().sum()
rows = []
for m in CANDS:
    x = out[m]; ok = x.notna()
    xs = x[ok]; best, worst = xs.max(), xs.min()
    young = ok & out['age'].between(25, 40)
    q01 = xs.quantile(0.01)
    obs = out.loc[ok].groupby('pidp').size()
    rows.append(dict(measure=m, person_years=int(ok.sum()), persons=int(out.loc[ok, 'pidp'].nunique()),
                     vs_phys=ok.sum() / ref_n, bhps_kept=(ok & bhps).sum() / (out['PHYS-4'].notna() & bhps).sum(),
                     obs_per_person=obs.mean(), distinct=int(xs.nunique()),
                     ceiling=(xs == best).mean(), ceiling_25_40=(x[young] == best).mean(),
                     floor=(xs == worst).mean(), distinct_bottom1=int(xs[xs <= q01].nunique())))
tab1 = pd.DataFrame(rows)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
print('=== 1. coverage and distribution (ages 20-90, own sample) ===')
print(tab1.round(3).to_string(index=False))
allbest = (DEF[SF6].sum(axis=1, min_count=6) == 0)
print(f"\nall-best six physical items: {allbest.mean():.1%} of rows with the items; "
      f"of those, energy 'all of the time': {(sf6b[allbest] == 1).mean():.1%} "
      f"-> ceiling with vitality added {(allbest & (sf6b == 1)).sum() / DEF[SF6].notna().all(axis=1).sum():.1%}")

# ---- 2. age profile: mean and variance by five-year bin, pooled-sd units --------
print('\n=== 2. age profiles in pooled-sd units: mean, and variance relative to ages 30-34 ===')
bins = [(30, 34), (45, 49), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89)]
rows = []
for m in CANDS:
    x = out[m]; ok = x.notna(); zz = (x - x[ok].mean()) / x[ok].std()
    r = dict(measure=m)
    for lo, hi in bins:
        s = zz[ok & out['age'].between(lo, hi)]
        r[f'mean {lo}'] = s.mean(); r[f'var {lo}'] = s.var()
    v0 = r['var 30']
    for lo, hi in bins:
        r[f'var {lo}'] = r[f'var {lo}'] / v0
    # which single year of age has the largest (5-yr smoothed) variance
    va = pd.Series({a: zz[ok & (out['age'] == a)].var() for a in range(22, 89)}).rolling(5, center=True).mean()
    r['var peak age'] = int(va.idxmax())
    r['decline/decade 30-50'] = (r['mean 45'] - r['mean 30']) / 1.5
    r['decline/decade 70-85'] = (r['mean 85'] - r['mean 70']) / 1.5
    rows.append(r)
tab2 = pd.DataFrame(rows)
print(tab2[['measure', 'mean 30', 'mean 60', 'mean 80', 'decline/decade 30-50', 'decline/decade 70-85',
            'var 45', 'var 55', 'var 60', 'var 65', 'var 70', 'var 75', 'var 80', 'var 85', 'var peak age']].round(2).to_string(index=False))

# ---- 3. convexity against the cost index and in-patient admission (waves 7-15) --
cost = pd.read_parquet(R / 'data/processed/measures/cost_index.parquet',
                       columns=['pidp', 'wave', 'hosp', 'flat_cost_total'])
c = out.merge(cost, on=['pidp', 'wave'], how='inner')
c = c[c['flat_cost_total'].notna() & c['hosp'].isin([1, 2])].copy()
c['hosp'] = (c['hosp'] == 1).astype(float)
c = c[c[CANDS].notna().all(axis=1)]
print(f"\n=== 3. convexity, common sample: {len(c):,} person-waves, {c.pidp.nunique():,} people (waves 7-15) ===")

def quad_levels(zv, y):
    X = np.column_stack([np.ones(len(zv)), zv, zv**2])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s2 = (r**2).sum() / (len(y) - 3)
    return b[2], b[2] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])

def quad_log_deciles(zv, y, k=10):
    b = np.clip((pd.Series(zv).rank(pct=True) * k).astype(int), 0, k - 1)
    g = pd.DataFrame({'z': zv, 'y': y, 'b': b}).groupby('b').agg(z=('z', 'mean'), y=('y', 'mean'))
    lg = np.log(g['y'].to_numpy() + 1.0)
    X = np.column_stack([np.ones(k), g['z'], g['z']**2])
    co, *_ = np.linalg.lstsq(X, lg, rcond=None)
    return co[2], g['y'].iloc[0] / g['y'].iloc[-1]

def logit_quad(zv, y, iters=30):
    X = np.column_stack([np.ones(len(zv)), zv, zv**2]); b = np.zeros(3)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
        H = X.T @ (X * W[:, None]); b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
    return b[2], b[2] / se[2]

rows = []
y = c['flat_cost_total'].to_numpy(float); h = c['hosp'].to_numpy(float)
for m in CANDS:
    zv = ((c[m] - c[m].mean()) / c[m].std()).to_numpy()
    ql, tl = quad_levels(zv, y); qlog, ratio = quad_log_deciles(zv, y)
    lq, lt = logit_quad(zv, h)
    pq, pt = quad_levels(zv, h)
    rows.append(dict(measure=m, quad_levels=ql, t_levels=tl, quad_log_deciles=qlog,
                     bottom_top_ratio=ratio, prob_quad=pq, t_prob=pt, logit_quad=lq, t_logit=lt))
tab3 = pd.DataFrame(rows)
print(tab3.round(3).to_string(index=False))
print("  quad_levels > 0: convex in pounds; quad_log_deciles > 0: proportional gradient steepens;")
print("  logit_quad > 0: in-patient admission convex on the log-odds scale")

tab1.to_csv(SCR / 'bc_tab1.csv', index=False); tab2.to_csv(SCR / 'bc_tab2.csv', index=False); tab3.to_csv(SCR / 'bc_tab3.csv', index=False)
print('\nwritten', SCR / 'baseline_candidates.parquet')
