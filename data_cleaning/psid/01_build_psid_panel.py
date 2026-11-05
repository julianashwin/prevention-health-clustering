"""Build a person-wave PSID panel, 1999-2023, in the shape of the UKHLS analysis panel.

One row per Reference Person (RP, formerly Head) or Spouse/Partner (SP, formerly Wife)
per biennial wave. Only these two people answer the PSID health section, so other family
members are used only for household composition (equivalised income).

Run from the repository root:
  .venv/bin/python data_cleaning/psid/01_build_psid_panel.py

Inputs
  --parquet    folder with FAM1999ER.parquet ... FAM2023ER.parquet and IND2023ER.parquet
               (a folder named IND2023ER.parquet, holding part files, also works), and
               CAH85_23.parquet, the childbirth history, for the maternity flag
               default: data/raw/psid/parquet
  --crosswalk  psid_crosswalk.csv: concept, year, person, variable  (person = RP, SP, FU,
               RESP or IND). Every PSID variable used comes from this file.
               default: data_cleaning/psid/psid_crosswalk.csv
Output
  --out        parquet file, one row per person-wave, raw PSID codes recoded to clean values
               default: data/processed/psid/psid_panel.parquet

See data_cleaning/psid/PSID_CROSSWALK.md for what each item is and how close it is to the
UKHLS item it stands in for, and PSID_CROSSWALK_DETAIL.md for wave-by-wave notes.

Codes: PSID uses 8/98/998 for DK, 9/99/999 for NA or refused and 0 for Inap. All three become
missing, except where Inap means "not asked because the answer is known" (documented below).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

WAVES = list(range(1999, 2024, 2))

# ---------------------------------------------------------------------------------------------
# Recode rules, by concept (or concept prefix). Each returns a float Series.
# ---------------------------------------------------------------------------------------------

def yes_no(s):
    """1 = Yes, 5 = No; everything else missing."""
    return s.map({1: 1.0, 5: 0.0})


def keep_range(lo, hi):
    def f(s):
        return s.where(s.between(lo, hi)).astype(float)
    return f


def srh(s):
    """1 Excellent ... 5 Poor (same order and wording as UKHLS sf1/scsf1)."""
    return s.where(s.between(1, 5)).astype(float)


def better_worse(s):
    """1 Better, 3 About the same, 5 Worse -> -1, 0, 1."""
    return s.map({1: -1.0, 3: 0.0, 5: 1.0})


def limit_amount(s):
    """1 A lot, 3 Somewhat, 5 Just a little, 7 Not at all -> 3, 2, 1, 0."""
    return s.map({1: 3.0, 3: 2.0, 5: 1.0, 7: 0.0})


def iadl_diff(s):
    """1 Yes, 5 No, 7 Does not do (follow-up asks whether because of health)."""
    return s.map({1: 1.0, 5: 0.0, 7: 7.0})


def k6_item(s):
    """1 All of the time ... 5 None of the time -> 4 ... 0 (the K6 scoring)."""
    return (5 - s.where(s.between(1, 5))).astype(float)


RULES = {
    'srh': srh,
    'srh_change2y': better_worse,
    'worklim_any': yes_no,
    'worklim_types': lambda s: s.map({1: 1.0, 5: 0.0, 7: 2.0}),  # 2 = can do no work at all
    'worklim_amount': limit_amount,
    'hosp_any': yes_no,
    'hosp_nights': keep_range(1, 365),
    'hosp_weeks': keep_range(1, 52),
    'mob_walk_climb': yes_no,
    'mob_bend_lift': yes_no,
    'smoke_now': yes_no,
    'smoke_everq': yes_no,  # asked only of people who do not smoke now (Inap = smokes now)
    'weight_lbs': keep_range(50, 500),  # top-coded at 500 to 2003, at 400 from 2005
    'weight_kg': keep_range(36, 180),
    'height_ft': keep_range(2, 7),
    'height_in': lambda s: s.where(s.between(0, 11), -1.0).astype(float),  # -1 = DK/NA (98/99)
    'height_m': keep_range(0.6, 2.1),
    'k6_scale': keep_range(0, 24),
    # 1 A lot, 2 Some, 3 A little, 5 Not at all; Inap if no distress (set to 0 below)
    'k6_interfere': lambda s: s.map({1: 3.0, 2: 2.0, 3: 1.0, 5: 0.0}),
    'resp_who': lambda s: s.astype(float),
    'age_fam': keep_range(1, 120),
    'sex_fam': lambda s: s.map({1: 1.0, 2: 2.0}),
    'fam_id': lambda s: s.astype(float),
    'state': lambda s: s.where(s.between(1, 51)).astype(float),
    'region': lambda s: s.where(s.between(1, 6)).astype(float),
    'iw_month': keep_range(1, 12), 'iw_day': keep_range(1, 31), 'iw_year': keep_range(1999, 2024),
    'fu_size': keep_range(1, 20),
    'faminc': lambda s: s.astype(float),
    'famwt': lambda s: s.astype(float),
    'educ': lambda s: s.where(s.between(0, 17)).astype(float),
    'race1': lambda s: s.where(s.between(1, 7)).astype(float),
    'hisp': lambda s: s.where(s.between(0, 7)).astype(float),  # 0 = not Hispanic (asked 2005+)
    'emp1': lambda s: s.where(s.between(1, 8)).astype(float),
    'emp2': lambda s: s.where(s.between(0, 8)).astype(float),  # 0 = no second mention
    'emp3': lambda s: s.where(s.between(0, 8)).astype(float),
    'marital': lambda s: s.where(s.between(1, 5)).astype(float),  # legal status (A3)
    'n_children': keep_range(0, 20),
    'year_retired': keep_range(1920, 2024),
    'hours_annual': keep_range(0, 8760),
    'wages': lambda s: s.astype(float),
    'degree_hs': lambda s: s.where(s.between(1, 3)).astype(float),  # 1 diploma, 2 GED, 3 neither
    'degree_college': yes_no,
    'degree_highest': lambda s: s.where(s.between(1, 6) | (s == 97)).astype(float),
    'laborinc': lambda s: s.astype(float),
    'wealth': lambda s: s.astype(float),
    'hosp_exp': lambda s: s.astype(float), 'doc_exp': lambda s: s.astype(float),
    'rx_exp': lambda s: s.astype(float), 'ins_exp': lambda s: s.astype(float),
    'health_exp': lambda s: s.astype(float),
}


def has_rule(concept):
    try:
        rule_for(concept)
        return True
    except KeyError:
        return False


def rule_for(concept):
    if concept in RULES:
        return RULES[concept]
    if concept.startswith(('adl_', )) or concept.endswith('_ever') or concept.endswith('_health'):
        return yes_no
    if concept.startswith('iadl_') and concept.endswith('_diff'):
        return iadl_diff
    if concept.endswith('_limits'):
        return limit_amount
    if concept.startswith('k6_'):
        return k6_item
    if '_type' in concept:
        return lambda s: s.where(s > 0).astype(float)  # open-ended type codes, kept raw
    if concept.endswith('_onset'):
        return lambda s: s.astype(float)  # left raw: years-ago in 1999-2003, age from 2005
    raise KeyError(f'no recode rule for {concept}')


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------

def read_parquet(folder, name, columns):
    p = pathlib.Path(folder) / f'{name}.parquet'
    return pd.read_parquet(p, columns=sorted(set(columns)))


def decode_death_year(v):
    """ER32050: exact year 1967-2023, or a range coded as two 2-digit years (e.g. 709 =
    2007-2009, 9901 = 1999-2001), 9999 = dead but year unknown, 0 = not known to be dead.
    Returns (first possible year, last possible year)."""
    v = pd.Series(v, dtype='float')
    lo = pd.Series(np.nan, index=v.index)
    hi = pd.Series(np.nan, index=v.index)
    exact = v.between(1967, 2023)
    lo[exact] = v[exact]
    hi[exact] = v[exact]
    rng = (v > 0) & ~exact & (v != 9999)
    a = (v[rng] // 100).astype(int)
    b = (v[rng] % 100).astype(int)
    to4 = lambda x: np.where(x >= 67, 1900 + x, 2000 + x)
    lo[rng] = to4(a)
    hi[rng] = to4(b)
    return lo, hi


def individual_long(parq, xw):
    ind = xw[xw.person == 'IND']
    cols = ['ER30001', 'ER30002', 'ER32000', 'ER32006', 'ER32050'] + list(ind.variable)
    df = read_parquet(parq, 'IND2023ER', cols)
    pid = df.ER30001.astype('int64') * 1000 + df.ER30002.astype('int64')
    death_lo, death_hi = decode_death_year(df.ER32050)
    fixed = pd.DataFrame({'pid': pid, 'sex': df.ER32000.map({1: 1.0, 2: 2.0}),
                          'sample_member': df.ER32006.astype(float),
                          'death_year_lo': death_lo, 'death_year_hi': death_hi,
                          'death_year_raw': df.ER32050})
    out = []
    for y in WAVES:
        m = ind[ind.year == y].set_index('concept').variable
        w = pd.DataFrame({'pid': pid, 'wave': y})
        for c, v in m.items():
            w[c] = df[v].values
        out.append(w)
    long = pd.concat(out, ignore_index=True)
    return long, fixed


def family_wide(parq, xw, y):
    fx = xw[(xw.year == y) & xw.person.isin(['RP', 'SP', 'FU', 'RESP'])]
    fx = fx[[has_rule(c) for c in fx.concept]]
    df = read_parquet(parq, f'FAM{y}ER', fx.variable)
    return df, fx


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------

def build(parq, xw):
    long, fixed = individual_long(parq, xw)
    # FU composition for equivalised income: everyone in the family at the interview
    infu = long[(long.ind_iw_num > 0) & long.ind_seq.between(1, 20)].copy()
    infu['adult14'] = infu.ind_age.between(14, 120)
    comp = (infu.groupby(['wave', 'ind_iw_num'])
            .agg(n_fu=('pid', 'size'), n14=('adult14', 'sum')).reset_index())
    comp['oecd_scale'] = 1 + 0.5 * (comp.n14 - 1).clip(lower=0) + 0.3 * (comp.n_fu - comp.n14)

    # RP and SP person-waves
    role = np.select([(long.ind_seq == 1) & (long.ind_rel == 10),
                      long.ind_seq.between(1, 20) & long.ind_rel.isin([20, 22])], ['RP', 'SP'], '')
    long['role'] = role
    people = long[long.role != ''].copy()

    rows = []
    for y in WAVES:
        fam, fx = family_wide(parq, xw, y)
        famid_var = fx[(fx.concept == 'fam_id')].variable.iloc[0]
        for r in ('RP', 'SP'):
            sub = fx[fx.person.isin([r, 'FU', 'RESP'])]
            cols = {'ind_iw_num': fam[famid_var].values}
            for _, x in sub.iterrows():
                if not has_rule(x.concept):
                    continue  # documented in the crosswalk but not cleaned into the panel
                name = ('resp__' if x.person == 'RESP' else '') + x.concept
                cols[name] = rule_for(x.concept)(fam[x.variable]).values
            d = pd.DataFrame(cols)
            d['wave'] = y
            d['role'] = r
            rows.append(d)
    fam_long = pd.concat(rows, ignore_index=True)
    panel = people.merge(fam_long, on=['wave', 'ind_iw_num', 'role'], how='left', validate='1:1')
    panel = panel.merge(fixed, on='pid', how='left')
    panel = panel.merge(comp.rename(columns={'ind_iw_num': 'ind_iw_num'}),
                        on=['wave', 'ind_iw_num'], how='left')

    # Who answered: the PSID interviews one respondent per family, who answers the health
    # section for both RP and SP. self_report = this person was the respondent.
    panel['self_report'] = panel.ind_respondent.map({1: 1.0, 5: 0.0})
    # K6 is asked of the respondent only: attach it to the respondent's own row.
    for c in [c for c in panel.columns if c.startswith('resp__')]:
        panel[c[6:]] = panel[c].where(panel.self_report == 1)
        panel.drop(columns=c, inplace=True)

    k6_items = [c for c in ('k6_sad', 'k6_nervous', 'k6_restless', 'k6_hopeless', 'k6_effort',
                            'k6_worthless') if c in panel]
    if k6_items and 'k6_interfere' in panel:
        none = (panel[k6_items] == 0).all(axis=1)
        panel.loc[none & panel.k6_interfere.isna() & (panel.self_report == 1), 'k6_interfere'] = 0.0

    # 1999 and 2001: ADLs asked only of people aged 55+ (Inap = under 55), left missing.
    # 2013: most ADL/IADL items skipped for people in excellent health with no work
    # limitation (codebook Inap note); those people are coded "no difficulty".
    skip13 = (panel.wave == 2013) & (panel.srh == 1) & (panel.worklim_any == 0)
    for c in [c for c in panel.columns if c.startswith(('adl_', 'iadl_')) and c.endswith('_diff')]:
        panel.loc[skip13 & panel[c].isna(), c] = 0.0
    # Help items: asked only if the person has difficulty; no difficulty -> no help.
    for c in [c for c in panel.columns if c.startswith('adl_') and c.endswith('_help')]:
        diff = c.replace('_help', '_diff')
        panel.loc[(panel[diff] == 0) & panel[c].isna(), c] = 0.0
    # IADL: difficulty because of health = 1 if "yes" or ("does not do" and because of health)
    for c in [c for c in panel.columns if c.startswith('iadl_') and c.endswith('_diff')]:
        h = c.replace('_diff', '_health')
        raw = panel[c]
        panel[c] = np.select([raw == 1, raw == 0, (raw == 7) & (panel.get(h) == 1),
                              (raw == 7) & (panel.get(h) == 0)], [1.0, 0.0, 1.0, 0.0], np.nan)
    # Work limitation as one ordinal: 0 none, 1 just a little / not at all, 2 somewhat,
    # 3 a lot or cannot work at all.
    wl = np.select([panel.worklim_any == 0, panel.worklim_types == 2,
                    panel.worklim_amount == 3, panel.worklim_amount == 2,
                    panel.worklim_amount.isin([0, 1])], [0, 3, 3, 2, 1], -1).astype(float)
    panel['worklim_ord'] = np.where(wl < 0, np.nan, wl)
    # Hospital nights in the previous calendar year (asked from 2003): days, or weeks x 7.
    nights = panel.hosp_nights.fillna(panel.hosp_weeks * 7)
    panel['hosp_nights_tot'] = np.where(panel.hosp_any == 0, 0.0, nights)
    # Body mass index
    kg = panel.weight_lbs * 0.45359237
    kg = kg.fillna(panel.get('weight_kg'))
    inches = panel.height_in.where(panel.height_in >= 0)  # DK/NA inches -> height missing
    m = (panel.height_ft * 12 + inches) * 0.0254
    m = m.fillna(panel.get('height_m'))
    bmi = kg / m ** 2
    panel['bmi'] = bmi.where(bmi.between(12, 80))  # outside this range: misreported height/weight
    # Smoking, for the frailty index of Hosseini, Kopecky and Zhao (2022): "smokes now" is H13;
    # "has ever smoked" is 1 for current smokers and the H16 answer for everyone else.
    if 'smoke_now' in panel:
        panel['ever_smoked'] = np.where(panel.smoke_now == 1, 1.0,
                                        np.where(panel.smoke_now == 0, panel.smoke_everq, np.nan))
    # Cancer: 1999-2003 asked "excluding skin cancer"; from 2005 skin cancer counts. Keep the
    # 1999-2003 definition: drop cases whose only mentioned type is non-melanoma skin cancer
    # (2005-2011 code 7 "skin cancer", melanoma is code 5; from 2013 codes 10 and 11).
    if 'cancer_type_m1' in panel:
        skin = np.where(panel.wave <= 2011, panel.cancer_type_m1 == 7,
                        panel.cancer_type_m1.isin([10, 11]))
        m2 = panel.get('cancer_type_m2')
        only_skin = skin & ((m2.isna()) | (m2 == 0) | np.where(panel.wave <= 2011, m2 == 7,
                                                               m2.isin([10, 11])))
        panel['cancer_ever'] = panel.cancer_ever.where(~(only_skin & (panel.cancer_ever == 1)), 0.0)
    # Ever-diagnosed conditions, three ways.
    #   *_ever      this wave's answer, as asked.
    #   *_ever_abs  absorbing, the UKHLS rule (once reported, kept). Because PSID re-asks the
    #               whole list every wave, any single "yes" is kept for good, so the count
    #               drifts up with time in the panel: at ages 50-59 it is 0.5 higher after
    #               four or more waves than at a person's first wave, while this wave's
    #               count does not move.
    #   *_ever_conf reported this wave, or in at least two waves so far. Uses no later waves,
    #               and stops single reports that are never repeated from persisting; the
    #               drift is 0.2.
    panel = panel.sort_values(['pid', 'wave'])
    for c in [c for c in panel.columns if c.endswith('_ever')]:
        panel[c + '_abs'] = panel.groupby('pid')[c].transform(lambda s: s.cummax())
        yes_so_far = panel[c].fillna(0).groupby(panel.pid).cumsum()
        panel[c + '_conf'] = np.where(panel[c] == 1, 1.0, np.where(yes_so_far >= 2, 1.0,
                                      np.where(panel[c].notna(), 0.0, np.nan)))
    phys = ['stroke', 'heart_attack', 'heart_disease', 'hypertension', 'asthma', 'lung_disease',
            'diabetes', 'arthritis', 'cancer']
    panel['cond_count_phys'] = panel[[c + '_ever_abs' for c in phys]].sum(axis=1, min_count=len(phys))
    panel['cond_count_now'] = panel[[c + '_ever' for c in phys]].sum(axis=1, min_count=len(phys))
    panel['cond_count_conf'] = panel[[c + '_ever_conf' for c in phys]].sum(axis=1, min_count=len(phys))
    # Out-of-pocket medical spending of the family, annual, excluding insurance premiums
    panel['oop_med_fu'] = panel[['hosp_exp', 'doc_exp', 'rx_exp']].sum(axis=1, min_count=3)
    # Mortality: died before the next wave = the person's last interview, and a death year no
    # later than two years on. A row followed by another interview is 0 even when the death
    # year is the next wave's year, because the person was interviewed alive that year.
    last_seen = panel.groupby('pid').wave.transform('max')
    is_last = panel.wave == last_seen
    panel['died_by_next_wave'] = ((panel.death_year_lo <= panel.wave + 2) & is_last).astype(float)
    panel.loc[panel.wave == max(WAVES), 'died_by_next_wave'] = np.nan  # no follow-up yet
    # A handful of people (10 in the 2023 release) have a death year before a later
    # interview; flag them rather than guess which record is wrong.
    panel['death_conflict'] = (panel.death_year_lo < last_seen).astype(float)
    # Equivalised family income (OECD modified), previous calendar year
    panel['faminc_eq'] = panel.faminc / panel.oecd_scale
    panel['age'] = panel.ind_age.where(panel.ind_age.between(1, 120))
    panel['faminc_rank'] = panel.groupby('wave').faminc_eq.rank(pct=True)
    # Employment status at interview (individual file, prioritised over up to 3 mentions):
    # 1 working, 2 temporarily laid off or on sick/maternity leave, 3 unemployed, 4 retired,
    # 5 permanently or temporarily disabled, 6 keeping house, 7 student, 8 other
    panel['emp_status'] = panel.ind_emp.where(panel.ind_emp.between(1, 8)).astype(float)
    # Education: years completed (individual-file version counts a GED as 12), in 4 levels:
    # 1 less than high school, 2 high school or GED, 3 some college, 4 bachelor's or more
    yrs = panel.ind_educ.where(panel.ind_educ.between(1, 17)).astype(float)
    yrs = yrs.fillna(panel.educ)
    panel['educ_years'] = yrs
    panel['educ_level'] = pd.cut(yrs, [-1, 11, 12, 15, 17], labels=[1, 2, 3, 4]).astype(float)
    # Ethnicity: race asked when someone becomes RP/SP (re-asked of all in 2005); Hispanic
    # origin asked from 2005 only, so a person who ever reports Hispanic origin counts as
    # Hispanic in every wave. In 1999-2003 "Latino" was one of the race codes (5).
    hisp_ever = panel.groupby('pid').hisp.transform(lambda h: (h > 0).max() if h.notna().any() else np.nan)
    panel['white_nonhisp'] = np.where(panel.race1.isna(), np.nan,
                                      ((panel.race1 == 1) & (hisp_ever != 1)).astype(float))
    panel = ukhls_counterparts(panel, parq)
    return panel


def any_of(panel, cols):
    """1 if any item is 1, 0 if every item is 0, missing otherwise."""
    x = panel[cols]
    return pd.Series(np.where((x == 1).any(axis=1), 1.0, np.where((x == 0).all(axis=1), 0.0, np.nan)),
                     index=panel.index)


def births_by_mother(parq):
    """Calendar years in which each woman gave birth, from the childbirth history CAH85_23
    (record type 1 = birth; parent sex 2 = female). Empty if the file is not there."""
    p = pathlib.Path(parq) / 'CAH85_23.parquet'
    if not p.exists():
        print('CAH85_23.parquet not found: maternity flag left missing')
        return None
    c = pd.read_parquet(p, columns=['CAH2', 'CAH3', 'CAH4', 'CAH5', 'CAH15'])
    c = c[(c.CAH2 == 1) & (c.CAH5 == 2) & c.CAH15.between(1900, 2024)]
    return pd.DataFrame({'pid': c.CAH3.astype('int64') * 1000 + c.CAH4.astype('int64'),
                         'birth_cal_year': c.CAH15.astype(int)}).drop_duplicates()


def ukhls_counterparts(panel, parq):
    """Variables built the way the UKHLS pipeline builds its own (src/prevention_health_clustering
    in the main repo), so the PSID exercises can reuse the same definitions."""
    # Impairment areas (UKHLS disdif, measures/health.py HEALTH_GROUPS). UKHLS counts areas of
    # difficulty; PSID asks ADL difficulty item by item, so an area is "any of its ADL items".
    #   mobility (disdif1, "moving around at home and walking"): walking, bed or chair, getting outside
    #   lifting and carrying (disdif2): trouble bending, lifting or stooping, 2017+ only
    #   personal care (disdif11, "washing, dressing"): bathing, dressing, eating, using the toilet
    # Not in PSID: dexterity (3), co-ordination (10), continence (4), hearing (5), sight (6).
    panel['area_mobility'] = any_of(panel, ['adl_walk_diff', 'adl_bed_diff', 'adl_outside_diff'])
    panel['area_lifting'] = panel.mob_bend_lift
    panel['area_personal_care'] = any_of(panel, ['adl_bath_diff', 'adl_dress_diff', 'adl_eat_diff',
                                                 'adl_toilet_diff'])
    panel['lim_fl'] = panel.area_mobility                               # 0-1, all waves
    panel['lim_fl_17'] = panel.area_mobility + panel.area_lifting       # 0-2, 2017+
    panel['lim_self'] = panel.area_personal_care                        # 0-1, all waves
    # Employment as in the paper's employment checks (UKHLS jbstat 1-2 employed, 4 retired,
    # 8 long-term sick or disabled). PSID code 2 (temporarily laid off, sick or maternity
    # leave; under 1 percent) counts as employed, because people on sick leave still hold a job.
    panel['employed'] = panel.emp_status.isin([1, 2]).astype(float).where(panel.emp_status.notna())
    panel['retired'] = (panel.emp_status == 4).astype(float).where(panel.emp_status.notna())
    panel['lt_sick_disabled'] = (panel.emp_status == 5).astype(float).where(panel.emp_status.notna())
    # Birth year (UKHLS birthy): PSID repeats it every wave; take the most common valid value,
    # falling back on wave minus age.
    by = panel.ind_birth_year.where(panel.ind_birth_year.between(1870, 2024))
    mode = by.groupby(panel.pid).agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    panel['birth_year'] = panel.pid.map(mode)
    approx = (panel.wave - panel.age).groupby(panel.pid).transform('median')
    panel['birth_year'] = panel.birth_year.fillna(approx.round())
    # Mortality as in the UKHLS multidimensional contract: died = the study knows the person
    # died; mort_event = 1 on the person's last panel row if they died, 0 on every other row.
    # UKHLS learns of a death at the next wave; PSID also traces deaths long after a family
    # stops responding, so a death more than two years after the last interview (216 people,
    # 8 percent of deaths in the 2023 release) is not an event here. died and death_year_lo
    # keep them.
    panel['died'] = panel.death_year_lo.notna().astype(float)
    last = panel.wave == panel.groupby('pid').wave.transform('max')
    panel['mort_event'] = (last & (panel.death_year_lo <= panel.wave + 2)).astype(float)
    # Education as in the UKHLS observables: the best level recorded at age 25 or over (any age
    # if never seen at 25+), fixed for the person. 1 < high school ... 4 bachelor's or more.
    best25 = panel.educ_level.where(panel.age >= 25).groupby(panel.pid).transform('max')
    panel['educ_best'] = best25.fillna(panel.groupby('pid').educ_level.transform('max'))
    # Race and ethnicity, person level, US groups in place of the UKHLS five: Hispanic (ever
    # reports Hispanic origin, or "Latino" race code 5 in 1999-2003), else the latest valid
    # race: white, Black, Asian (2005+ code 4; 1999-2003 code 4 includes Pacific Islanders),
    # other (Native American, Pacific Islander from 2005, other, multiple not separated).
    latino_old = (panel.wave <= 2003) & (panel.race1 == 5)
    hisp_any = (panel.hisp > 0) | latino_old
    hisp_p = hisp_any.groupby(panel.pid).transform('max')
    r = panel.race1.where(~latino_old)
    lastrace = r.groupby(panel.pid).transform('last')
    grp = np.select([hisp_p, lastrace == 1, lastrace == 2, lastrace == 4, lastrace.notna()],
                    ['hispanic', 'white', 'black', 'asian', 'other'], '')
    panel['race_group'] = pd.Series(grp, index=panel.index).replace('', np.nan)
    # Maternity (UKHLS hospch): the UKHLS cost index drops in-patient stays for childbirth.
    # PSID does not ask why someone was in hospital, so a woman who gave birth in the calendar
    # year the hospital question covers (the year before the interview) is flagged, and
    # hosp_any_nonmat / hosp_nights_nonmat treat her stay as maternity, as the UKHLS code does.
    births = births_by_mother(parq)
    if births is not None:
        b = births.assign(wave=births.birth_cal_year + 1, birth_prev_year=1.0)[['pid', 'wave', 'birth_prev_year']]
        panel = panel.merge(b.drop_duplicates(['pid', 'wave']), on=['pid', 'wave'], how='left')
        panel['birth_prev_year'] = panel.birth_prev_year.fillna(0.0)
    else:
        panel['birth_prev_year'] = np.nan
    mat = panel.birth_prev_year == 1
    panel['hosp_any_nonmat'] = panel.hosp_any.where(~mat, 0.0).where(panel.hosp_any.notna())
    panel['hosp_nights_nonmat'] = panel.hosp_nights_tot.where(~mat, 0.0).where(panel.hosp_nights_tot.notna())
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', default='data/raw/psid/parquet')
    ap.add_argument('--crosswalk', default='data_cleaning/psid/psid_crosswalk.csv')
    ap.add_argument('--out', default='data/processed/psid/psid_panel.parquet')
    a = ap.parse_args()
    xw = pd.read_csv(a.crosswalk, dtype={'variable': str})
    xw['year'] = pd.to_numeric(xw.year, errors='coerce')
    panel = build(a.parquet, xw)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(a.out, index=False)
    print(f'{len(panel):,} person-waves, {panel.pid.nunique():,} people -> {a.out}')


if __name__ == '__main__':
    main()
