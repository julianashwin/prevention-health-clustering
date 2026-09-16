"""Every candidate measure, every quantitative criterion, one pass, common samples.

The original 26 measures plus the limitation banks of
data_cleaning/07_build_limitation_banks.py: P-LIM1, P-LIM, P-LIM4, P-LIM3 and
P-LIM3+O under the graded-response model, and all six banks (with P-FUNC)
under the generalised partial credit model, each on the h and theta scales.
The banks share P-FUNC's person-waves, so the common cost and mortality samples
are unchanged by adding them."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path('/Users/julianashwin/Documents/GitHub/prevention-health-clustering'); SCR = R / 'data/processed/baseline_measures'  # intermediate data, gitignored
base = pd.read_parquet(SCR / 'baseline_candidates.parquet')
grm = pd.read_parquet(SCR / 'criteria_measures.parquet', columns=['pidp', 'wave', 'GRM h', 'GRM theta', 'P-FULL h', 'P-FULL theta'])
fsp = pd.read_parquet(SCR / 'fs_pc_measures.parquet')
d = base.merge(grm, on=['pidp', 'wave'], how='left').merge(fsp, on=['pidp', 'wave'], how='left')
SLUG = {'P-FUNC': 'pfunc', 'P-LIM1': 'plim1', 'P-LIM': 'plim', 'P-LIM4': 'plim4', 'P-LIM3': 'plim3', 'P-LIM3+O': 'plim3o'}
LIM = {}
for bank, slug in SLUG.items():
    for model, tag in (('grm', ''), ('gpcm', ' GPCM')):
        if bank == 'P-FUNC' and model == 'grm':
            continue                                   # already in the panel as P-FUNC h / theta
        LIM[f'{bank}{tag} h'] = f'h_{slug}_{model}'
        LIM[f'{bank}{tag} theta'] = f'theta_{slug}_{model}'
for bank, slug in SLUG.items():
    if bank == 'P-FUNC':
        continue                                       # P-FUNC plus the groups is P-FULL, already in the panel
    for code in ('CC', 'CG'):
        LIM[f'{bank}+{code} h'] = f'h_{slug}{code.lower()}_grm'
        LIM[f'{bank}+{code} theta'] = f'theta_{slug}{code.lower()}_grm'
lim = pd.read_parquet(R / 'data/processed/measures/limitation_scores.parquet', columns=['pidp', 'wave', *LIM.values()])
d = d.merge(lim.rename(columns={v: k for k, v in LIM.items()}), on=['pidp', 'wave'], how='left')
# the linear twins of 12_simple_twins.py: the same testlets as a weighted sum
TWIN = {}
for bank in ('P-FUNC', 'P-LIM', 'P-LIM3', 'P-LIM3+CC'):
    col = bank.replace('+', '').replace('-', '').lower()
    TWIN[f'{bank} FS'] = f'fs_{col}'
    TWIN[f'{bank} sum'] = f'sum_{col}'
tw = pd.read_parquet(SCR / 'simple_twins.parquet', columns=['pidp', 'wave', *TWIN.values()])
d = d.merge(tw.rename(columns={v: k for k, v in TWIN.items()}), on=['pidp', 'wave'], how='left')
LIM.update(TWIN)
KEYS = ['PCS', 'PHYS-4', 'PHYS-4eq', 'GRM h', 'GRM theta', 'PC pearson SF', 'PC polychoric SF', 'FS pearson SF', 'FS polychoric SF',
        'PHYS+F', 'FI-15', 'P-FUNC h', 'P-FUNC theta', 'PC pearson SF+F', 'PC polychoric SF+F', 'FS pearson SF+F', 'FS polychoric SF+F',
        'PHYS+F+C', 'FI-31', 'P-FULL h', 'P-FULL theta', 'PC pearson SF+F+C', 'PC polychoric SF+F+C', 'FS pearson SF+F+C',
        'FS polychoric SF+F+C', 'FS polychoric within age SF+F+C'] + list(LIM)
assert all(k in d.columns for k in KEYS), [k for k in KEYS if k not in d.columns]
d[['pidp', 'wave', 'age'] + KEYS].to_parquet(SCR / 'master_measures.parquet', index=False)

first = d.groupby('pidp')['wave'].transform('min'); w2 = first == 2; ref = d['PHYS-4'].notna()
rows = []
for m in KEYS:
    x = d[m]; ok = x.notna(); xs = x[ok]; best, worst = xs.max(), xs.min(); age = d['age']
    y20 = x[ok & age.between(20, 39)]; o60 = x[ok & age.between(60, 84)]; o85 = x[ok & age.between(85, 90)]
    z = (x - xs.mean()) / xs.std(); zb = lambda lo, hi: z[ok & age.between(lo, hi)]
    va = pd.Series({a: z[ok & (age == a)].var() for a in range(22, 89)}).rolling(5, center=True).mean()
    both = ok & ref
    rows.append({'key': m, 'person_years': int(ok.sum()), 'persons': int(d.loc[ok, 'pidp'].nunique()),
        'share_of_phys_person_years': ok.sum() / ref.sum(), 'share_wave2_entrants_kept': (ok & w2).sum() / (ref & w2).sum(),
        'obs_per_person': d.loc[ok].groupby('pidp').size().mean(), 'distinct_values': int(xs.nunique()),
        'best_all': (xs == best).mean(), 'best_20_39': (y20 == best).mean(), 'worst_all': (xs == worst).mean(),
        'worst_60_84': (o60 == worst).mean(), 'distinct_bottom1': int(xs[xs <= xs.quantile(0.01)].nunique()),
        'distinct_bottom_decile_85_90': int(o85[o85 <= o85.quantile(0.10)].nunique()),
        'rank_corr_phys': d.loc[both, m].rank().corr(d.loc[both, 'PHYS-4'].rank()),
        'decline_30_50': (zb(45, 49).mean() - zb(30, 34).mean()) / 1.5, 'decline_70_85': (zb(85, 89).mean() - zb(70, 74).mean()) / 1.5,
        'var_growth_30_60': zb(60, 64).var() / zb(30, 34).var(), 'var_growth_60_85': zb(85, 89).var() / zb(60, 64).var(),
        'var_peak_age': int(va.idxmax())})
T1 = pd.DataFrame(rows)

cost = pd.read_parquet(R / 'data/processed/measures/cost_index.parquet', columns=['pidp', 'wave', 'hosp', 'flat_cost_total'])
c = d.merge(cost, on=['pidp', 'wave']); c = c[c['flat_cost_total'].notna() & c['hosp'].isin([1, 2])]
c = c[c[KEYS].notna().all(axis=1)].reset_index(drop=True); g = pd.factorize(c['pidp'])[0]
assert (len(c), g.max() + 1) == (210389, 44941), 'the limitation banks changed the common cost sample'
def fit(zv, y):
    X = np.column_stack([np.ones(len(zv)), zv, zv**2]); XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    S = np.zeros((g.max() + 1, 3)); np.add.at(S, g, X * u[:, None]); V = XtXi @ (S.T @ S) @ XtXi
    return b[2], b[2] / np.sqrt(V[2, 2])
def deciles(zv, y, k=10):
    bb = np.clip((pd.Series(zv).rank(pct=True) * k).astype(int), 0, k - 1)
    mm = pd.DataFrame({'z': zv, 'y': y, 'b': bb}).groupby('b').mean()
    k = len(mm)                    # a coarse measure can leave a decile empty; then the steps count is out of k-2
    if k < 4:
        return np.nan, np.nan, -1
    X = np.column_stack([np.ones(k), mm.z, mm.z**2]); co, *_ = np.linalg.lstsq(X, np.log(mm.y.to_numpy() + 1), rcond=None)
    s = np.diff(mm.y.to_numpy()) / np.diff(mm.z.to_numpy())
    return co[2], mm.y.iloc[0] / mm.y.iloc[-1], int((np.diff(s) < 0).sum())
y = c['flat_cost_total'].to_numpy(float); yc = np.minimum(y, np.quantile(y, 0.99)); h = (c['hosp'] == 1).to_numpy(float)
rows = []
for m in KEYS:
    zv = ((c[m] - c[m].mean()) / c[m].std()).to_numpy()
    b, t = fit(zv, y); bc, tc = fit(zv, yc); lq, ratio, viol = deciles(zv, y); hb, ht = fit(zv, h)
    rows.append({'key': m, 'cost_curv': b, 'cost_curv_t': t, 'cost_curv_capped': bc, 'cost_curv_capped_t': tc, 'cost_log_curv': lq,
                 'cost_nonconvex_steps': viol, 'cost_ratio': ratio, 'inpatient_curv': hb, 'inpatient_curv_t': ht})
T2 = pd.DataFrame(rows)
out = T1.merge(T2, on='key'); out.to_csv(SCR / 'master_py.csv', index=False)
print(f"cost common sample: {len(c):,} person-waves, {g.max()+1:,} people")
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 50)
print(out.round(3).to_string(index=False))
