import math, re
import numpy as np, pandas as pd
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter, column_index_from_string
from openpyxl.formatting.rule import CellIsRule
from openpyxl.comments import Comment
from openpyxl.workbook.properties import CalcProperties

R0_ = Path('/Users/julianashwin/Documents/GitHub/prevention-health-clustering'); SCR = R0_ / 'data/processed/baseline_measures'
OUT = Path('/Users/julianashwin/Documents/GitHub/AnalysisForEIT/exploratory_w_johannes/redo_concepts/baseline_measure_comparison.xlsx')
R0 = Path('/Users/julianashwin/Documents/GitHub/prevention-health-clustering')
df = pd.read_csv(SCR / 'master_py.csv').merge(pd.read_csv(SCR / 'master_r.csv'), on='key')
df = df.merge(pd.read_csv(SCR / 'cost_age_invariance.csv')[['key', 'log_20-34', 'log_75-90', 'log_ratio_old_young', 'log_interaction', 'log_interaction_t']], on='key')
W = pd.read_csv(SCR / 'fs_pc_weights.csv')

FONT = 'Arial'
def f(bold=False, size=10, color='000000', italic=False): return Font(name=FONT, bold=bold, size=size, color=color, italic=italic)
thin = Side(style='thin', color='BFBFBF'); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical='top'); TOP = Alignment(vertical='top'); CTR = Alignment(horizontal='center', vertical='top', wrap_text=True)

# ---------------------------------------------------------------------------- text
SF6 = ("sf1 general health; sf2a health limits moderate activities; sf2b health limits climbing several flights of stairs; "
       "sf3a accomplished less because of physical health; sf3b limited in kind of work because of physical health; sf5 pain interfered with normal work")
SF12 = SF6 + ("; sf4a accomplished less because of emotional problems; sf4b did work less carefully because of emotional problems; "
              "sf6a felt calm and peaceful; sf6b had a lot of energy; sf6c felt downhearted and depressed; sf7 health interfered with social activities. "
              "Uses the UKHLS derived variable sf12pcs_dv")
FUNC5 = ("health long-standing illness, which gates the impairment module; disdif1 mobility; disdif2 lifting, carrying or moving objects; "
         "disdif3 manual dexterity; disdif10 physical coordination; disdif11 personal care")
FI15 = ("health long-standing illness; disdif1 mobility; disdif2 lifting, carrying or moving objects; disdif3 manual dexterity; disdif4 continence; "
        "disdif5 hearing; disdif6 sight; disdif10 physical coordination; disdif11 personal care")
COND16 = ("Ever-diagnosed conditions, built from the inventory at first interview, hcond1 to hcond16, and later new-diagnosis reports, hcondn in waves 2 to 9 "
          "and hcondcode and hcondncode from wave 10: asthma, arthritis, congestive heart failure, coronary heart disease, angina, heart attack, stroke, "
          "emphysema, hyperthyroidism, hypothyroidism, chronic bronchitis, liver condition, cancer, diabetes, epilepsy, high blood pressure")
GROUPS = ("Ever-diagnosed conditions from the same sources, in six groups: cardiovascular, from congestive heart failure, coronary heart disease, angina, "
          "heart attack and stroke; diabetes and high blood pressure; respiratory, from asthma, emphysema and chronic bronchitis; arthritis; cancer; "
          "other, from hyperthyroidism, hypothyroidism, liver condition and epilepsy")
V = {'SF6': SF6, 'SF12': SF12, 'SF6+F': SF6 + '\n' + FUNC5, 'FI15': SF6 + '\n' + FI15, 'SF6+F+C16': SF6 + '\n' + FUNC5 + '\n' + COND16,
     'FI31': SF6 + '\n' + FI15 + '\n' + COND16, 'SF6+F+G': SF6 + '\n' + FUNC5 + '\n' + GROUPS}
V['SF6+L9'] = SF6 + '\n' + FI15 + '; disdif12 other health problem or disability'
V['FI15+C16'] = V['FI15'] + '\n' + COND16; V['FI15+G'] = V['FI15'] + '\n' + GROUPS
V['L9+C16'] = V['SF6+L9'] + '\n' + COND16; V['L9+G'] = V['SF6+L9'] + '\n' + GROUPS

T_GRM = ("Needs a measurement-model section: discriminations, thresholds, latent scoring and the choice of ruler.", "No")
T_LIN = ("A weighted sum with a printed weight table; the estimation step takes one sentence.", "Yes")
L_GRM = ("Partly", "Samejima 1969 graded-response model, used to calibrate the PROMIS item banks. Rare in economics, and these banks were built for this project.")
L_PCP = ("Yes", "First-principal-component health index, as in Poterba, Venti and Wise's HRS health index.")
L_PCY = ("Yes", "Principal components on polychoric correlations, as recommended for discrete indicators by Kolenikov and Angeles 2009.")
L_FS = ("Partly", "Factor-score weighting, as in the PCS scoring algorithm. A one-factor index on these items is not a published standard.")
T_GPCM = ("A weighted sum of the answers with estimated weights, passed through one fixed curve; still needs a short measurement-model paragraph.", "Partly")
L_GPCM = ("Partly", "Muraki 1992 generalised partial credit model, standard in educational testing; less used than the graded-response model in health measurement, and rare in economics.")

PC_P = ("The first principal component of the Pearson correlation matrix of {items}. Each item is standardised and weighted by the first eigenvector, "
        "normalised to sum to one, and the score is re-standardised. {extra}")
PC_Y = ("As the Pearson principal component, but the eigenvector comes from the polychoric correlation matrix, which treats the items as ordinal. "
        "The weights are applied to the standardised observed items. {extra}")
FS_P = ("A one-factor maximum-likelihood factor analysis of the Pearson correlations of {items}. Each standardised item is weighted by its loading divided by "
        "its unique variance, the factor-score weight, normalised to sum to one, and the score is re-standardised. {extra}")
FS_Y = ("As the Pearson factor scores, but with the one-factor model fitted to polychoric correlations and the weights applied to the standardised observed "
        "items, which over-weights skewed items. {extra}")
IT_SF = "the four SF-12 testlets used by the GRM: general health, physical functioning, role-physical and bodily pain"
IT_F = "the four SF-12 testlets and the limitation count, capped at three as in the GRM"
IT_C = "the eleven P-FULL items: the four SF-12 testlets, the capped limitation count and the six condition groups"

M = [
 # key, name, family, content, method, variables, transparency (level, note), literature (level, anchor), excludes mental
 ('PCS', 'PCS', 'Survey summary score', 'SF-12, physical and mental',
  "The survey's own SF-12 physical component summary. All twelve items form eight subscales, each standardised against US 1998 population norms and combined "
  "with the published orthogonal factor-score coefficients. The mental health and role-emotional subscales enter with negative weights, so better mental "
  "health lowers the physical score. Scaled to mean 50 and standard deviation 10 in the norm population.",
  'SF12', ('Yes', "A published score that can be cited rather than explained; the negative mental weights need a sentence."),
  ('Yes', "Ware, Kosinski and Keller 1996 SF-12 scoring. Its negative mental weights are criticised by Simon and coauthors 1998, Taft and coauthors 2001 and Farivar and coauthors 2007."), 'No'),
 ('PHYS-4', 'phys', 'Weighted subscales', 'SF-12 physical',
  "The four physical SF-12 subscales: physical functioning, role-physical, bodily pain and general health. Each is scored 0 to 100, standardised against "
  "UKHLS wave-1 UK norms and weighted by its loading on the physical factor of a UK varimax factor analysis, renormalised to sum to one; on the "
  "standardised subscales the weights are 0.29, 0.27, 0.23 and 0.22. The result is rescaled to mean 50 and standard deviation 10, and no mental item enters.",
  'SF6', ('Yes', "Four named subscales with fixed weights; one paragraph."),
  ('Partly', "The SF-12's own physical subscales, weighted by factor loadings as in the PCS logic but without the mental subscales. Not a published summary score."), 'Yes'),
 ('PHYS-4eq', 'Physical subscales, equal weights', 'Weighted subscales', 'SF-12 physical',
  "The plain average of the same four physical subscales, each on its standard 0 to 100 scale, with no norming or estimated weights. It behaves like phys "
  "on every criterion, but equal weights create ties, so it takes far fewer distinct values.",
  'SF6', ('Yes', "A plain average of four subscales."), ('Partly', "Standard SF-12 subscale scoring, averaged."), 'Yes'),
 ('GRM h', 'GRM original, TCC', 'Graded-response model, TCC scale', 'SF-12 physical',
  "A graded-response item response model on four SF-12 testlets: general health, physical functioning as the sum of the two activity items, role-physical "
  "as the sum of the two role items, and bodily pain. The model estimates a discrimination and thresholds for each testlet and a latent score θ for each "
  "answer pattern, fitted multigroup by single year of age. This version reports the test-characteristic-curve score: θ mapped to the expected share of the "
  "maximum total score, on a 0 to 1 scale.", 'SF6', ("No", T_GRM[0]), L_GRM, 'Yes'),
 ('GRM theta', 'GRM original, θ', 'Graded-response model, θ scale', 'SF-12 physical',
  "The same graded-response model reported on its latent scale: the expected a posteriori θ, on a standard-normal footing in the pooled population. It ranks "
  "people identically to the TCC score but stretches both tails.", 'SF6', ("No", T_GRM[0]), L_GRM, 'Yes'),
 ('PC pearson SF', 'Principal component, Pearson: SF-12 physical', 'Principal component', 'SF-12 physical',
  PC_P.format(items=IT_SF, extra="The weights come out almost equal."), 'SF6', ("Yes", T_LIN[0]), L_PCP, 'Yes'),
 ('PC polychoric SF', 'Principal component, polychoric: SF-12 physical', 'Principal component', 'SF-12 physical',
  PC_Y.format(extra="For these four items the weights are almost identical to the Pearson version."), 'SF6', ("Yes", T_LIN[0]), L_PCY, 'Yes'),
 ('FS pearson SF', 'Factor scores, Pearson: SF-12 physical', 'Factor scores', 'SF-12 physical',
  FS_P.format(items=IT_SF, extra="It puts more weight on physical functioning and role limitation than on general health and pain."), 'SF6', ("Yes", T_LIN[0]), L_FS, 'Yes'),
 ('FS polychoric SF', 'Factor scores, polychoric: SF-12 physical', 'Factor scores', 'SF-12 physical',
  FS_Y.format(extra=""), 'SF6', ("Yes", T_LIN[0]), L_FS, 'Yes'),
 ('PHYS+F', 'PHYS without conditions', 'Standardised-component index', 'SF-12 physical + limitations',
  "The paper structure's PHYS index without its conditions part. The equal-weighted physical subscale mean and a count of five Equality Act physical "
  "limitations are each standardised, the count is subtracted, and the result is re-standardised. The count runs from 0 to 5 uncapped and is zero for "
  "anyone without a long-standing illness, because only they are asked. Equal standardised weights make one limitation worth about 25 points of the 0 to 100 subscale mean.",
  'SF6+F', ('Yes', "Two standardised components with equal weights; simple to state, but the implied weight of one limitation is hard to defend."),
  ('Partly', "Average of standardised components, as in Kling, Liebman and Katz 2007 summary indices; this combination of health components is new."), 'Yes'),
 ('FI-15', 'Frailty index, 15 deficits', 'Frailty index', 'SF-12 physical + limitations',
  "A deficit-accumulation frailty index over 15 deficits, reported as one minus the index so that 1 means no deficits. The six SF-12 physical items are graded "
  "from 0 to 1 by response step, and long-standing illness and eight Equality Act impairments are binary. Every deficit has equal weight, and a person-wave "
  "is scored only when all 15 are observed.", 'FI15', ('Yes', "The share of 15 listed deficits a person has."),
  ('Partly', "Deficit-accumulation method of Searle and coauthors 2008; standard indices include diagnoses and usually 30 or more deficits."), 'Yes'),
 ('P-FUNC h', 'P-FUNC, TCC', 'Graded-response model, TCC scale', 'SF-12 physical + limitations',
  "The graded-response model with a fifth testlet: the count of five Equality Act physical limitations, capped at three or more and zero for anyone without "
  "a long-standing illness. Reported on the test-characteristic-curve scale, from 0 to 1.", 'SF6+F', ("No", T_GRM[0]), L_GRM, 'Yes'),
 ('P-FUNC theta', 'P-FUNC, θ', 'Graded-response model, θ scale', 'SF-12 physical + limitations',
  "P-FUNC reported on its latent θ scale, which ranks people identically to its TCC score but stretches both tails.", 'SF6+F', ("No", T_GRM[0]), L_GRM, 'Yes'),
 ('PC pearson SF+F', 'Principal component, Pearson: + limitations', 'Principal component', 'SF-12 physical + limitations',
  PC_P.format(items=IT_F, extra=""), 'SF6+F', ("Yes", T_LIN[0]), L_PCP, 'Yes'),
 ('PC polychoric SF+F', 'Principal component, polychoric: + limitations', 'Principal component', 'SF-12 physical + limitations',
  PC_Y.format(extra=""), 'SF6+F', ("Yes", T_LIN[0]), L_PCY, 'Yes'),
 ('FS pearson SF+F', 'Factor scores, Pearson: + limitations', 'Factor scores', 'SF-12 physical + limitations',
  FS_P.format(items=IT_F, extra=""), 'SF6+F', ("Yes", T_LIN[0]), L_FS, 'Yes'),
 ('FS polychoric SF+F', 'Factor scores, polychoric: + limitations', 'Factor scores', 'SF-12 physical + limitations',
  FS_Y.format(extra="The limitation count gets the largest weight of any item."), 'SF6+F', ("Yes", T_LIN[0]), L_FS, 'Yes'),
 ('PHYS+F+C', 'PHYS with conditions', 'Standardised-component index', 'SF-12 physical + limitations + conditions',
  "The paper structure's PHYS index as specified: the equal-weighted physical subscale mean, the uncapped count of five physical limitations and the count "
  "of 16 ever-diagnosed physical conditions, each standardised, combined with equal weights and re-standardised. One diagnosis is worth about 20 points of "
  "the 0 to 100 subscale mean. Conditions exist only from a person's first condition inventory, which almost no BHPS continuer receives.",
  'SF6+F+C16', ('Yes', "Three standardised components with equal weights; simple to state, but the implied weight of one diagnosis is hard to defend."),
  ('Partly', "Average of standardised components, as in Kling, Liebman and Katz 2007 summary indices; this combination of health components is new."), 'Yes'),
 ('FI-31', 'Frailty index, 31 deficits', 'Frailty index', 'SF-12 physical + limitations + conditions',
  "The 15-deficit frailty index plus 16 ever-diagnosed physical conditions as binary deficits, 31 in all, each with equal weight and reported as one minus "
  "the index. Diagnoses carry just over half of the weight. Depression and cognitive items are excluded, and the index needs the condition inventory.",
  'FI31', ('Yes', "The share of 31 listed deficits a person has."),
  ('Yes', "Rockwood deficit-accumulation index, Searle and coauthors 2008. Used for lifecycle health by Hosseini, Kopecky and Zhao 2022; a 36-item UKHLS index by Labeit and coauthors 2024."), 'Yes'),
 ('P-FULL h', 'P-FULL, TCC', 'Graded-response model, TCC scale', 'SF-12 physical + limitations + conditions',
  "P-FUNC plus six condition-group items from the ever-diagnosed inventory: cardiovascular, diabetes and high blood pressure, and respiratory counted as 0, 1, "
  "or 2 or more, and arthritis, cancer and other conditions as binary. The model estimates low discriminations for the condition groups, so they carry little "
  "weight. Reported on the test-characteristic-curve scale; needs the condition inventory.", 'SF6+F+G', ("No", T_GRM[0]), L_GRM, 'Yes'),
 ('P-FULL theta', 'P-FULL, θ', 'Graded-response model, θ scale', 'SF-12 physical + limitations + conditions',
  "P-FULL reported on its latent θ scale, which ranks people identically to its TCC score but stretches both tails.", 'SF6+F+G', ("No", T_GRM[0]), L_GRM, 'Yes'),
 ('PC pearson SF+F+C', 'Principal component, Pearson: + limitations + conditions', 'Principal component', 'SF-12 physical + limitations + conditions',
  PC_P.format(items=IT_C, extra="The condition groups receive about a third of the weight."), 'SF6+F+G', ("Yes", T_LIN[0]), L_PCP, 'Yes'),
 ('PC polychoric SF+F+C', 'Principal component, polychoric: + limitations + conditions', 'Principal component', 'SF-12 physical + limitations + conditions',
  PC_Y.format(extra="The condition groups receive about two fifths of the weight."), 'SF6+F+G', ("Yes", T_LIN[0]), L_PCY, 'Yes'),
 ('FS pearson SF+F+C', 'Factor scores, Pearson: + limitations + conditions', 'Factor scores', 'SF-12 physical + limitations + conditions',
  FS_P.format(items=IT_C, extra="The condition groups receive 15% of the weight, and physical functioning and role limitation the most."), 'SF6+F+G', ("Yes", T_LIN[0]), L_FS, 'Yes'),
 ('FS polychoric SF+F+C', 'Factor scores, polychoric: + limitations + conditions', 'Factor scores', 'SF-12 physical + limitations + conditions',
  FS_Y.format(extra="The condition groups receive 16% of the weight, and the limitation count the largest single weight."), 'SF6+F+G', ("Yes", T_LIN[0]), L_FS, 'Yes'),
 ('FS polychoric within age SF+F+C', 'Factor scores, polychoric within age: + limitations + conditions', 'Factor scores', 'SF-12 physical + limitations + conditions',
  "As the polychoric factor scores, but with the polychoric correlations estimated separately within seven age bands, 20 to 29 up to 80 to 90, and averaged "
  "with sample-size weights, so the weights reflect how the items move together among people of the same age.", 'SF6+F+G', ("Yes", T_LIN[0]), L_FS, 'Yes'),
]
# ---------------------------------------------------------------------------- the limitation banks, 15 September 2026
LIMB = [
 ('P-FUNC', 'SF6+F', 'SF-12 physical + limitations', None),
 ('P-LIM1', 'SF6+F', 'SF-12 physical + limitations',
  "P-FUNC with the limitation count uncapped: the five Equality Act physical limitations counted from 0 to 5, zero for anyone without a long-standing "
  "illness. Every count keeps its own category, since each holds at least half a percent of person-waves."),
 ('P-LIM', 'FI15', 'SF-12 physical + limitations, sensory, continence',
  "P-LIM1 plus a second count over continence, hearing and sight, scored 0, 1, or 2 or more; three such difficulties, 0.18% of person-waves, merge into two."),
 ('P-LIM4', 'FI15', 'SF-12 physical + limitations, sensory, continence',
  "The four SF-12 testlets plus four limitation testlets: mobility and lifting counted 0 to 2; dexterity, coordination and personal care 0 to 3; continence "
  "yes or no; and hearing and sight 0 to 2. The split lets the model weight the common and the more severe limitations separately."),
 ('P-LIM3', 'FI15', 'SF-12 physical + limitations, sensory, continence',
  "The four SF-12 testlets plus three limitation testlets: functional limitations, counting mobility, lifting, dexterity and coordination from 0 to 4; "
  "self-care, counting personal care and continence from 0 to 2; and sensory, counting hearing and sight from 0 to 2."),
 ('P-LIM3+O', 'SF6+L9', 'SF-12 physical + limitations, sensory, continence, other',
  "P-LIM3 plus the non-specific impairment item, other health problem or disability, as its own yes-or-no testlet. Its content is unstated and may "
  "include mental health or fatigue."),
]
GPCM_NOTE = ("The model's score depends on the answers only through a weighted sum, with one estimated weight per testlet for each category step, so it is "
             "exactly a fixed monotone curve applied to that sum. It fits these items worse than the graded-response model, because the two-item SF-12 "
             "testlets have uneven category use.")
for bank, vkey, content, text in LIMB:
    mental = 'Partly' if bank == 'P-LIM3+O' else 'Yes'
    if text is not None:
        M.append((f'{bank} h', f'{bank}, TCC', 'Graded-response model, TCC scale', content,
                  text + " Reported on the test-characteristic-curve scale, from 0 to 1.", vkey, ("No", T_GRM[0]), L_GRM, mental))
        M.append((f'{bank} theta', f'{bank}, θ', 'Graded-response model, θ scale', content,
                  f"{bank} reported on its latent θ scale, which ranks people identically to its TCC score but stretches both tails.", vkey, ("No", T_GRM[0]), L_GRM, mental))
    M.append((f'{bank} GPCM h', f'{bank}, partial credit, expected score', 'Partial credit model, expected-score scale', content,
              f"The {bank} items fitted with the generalised partial credit model instead of the graded-response model. " + GPCM_NOTE +
              " Reported on the expected-score scale, from 0 to 1.", vkey, (T_GPCM[1], T_GPCM[0]), L_GPCM, mental))
    M.append((f'{bank} GPCM theta', f'{bank}, partial credit, θ', 'Partial credit model, θ scale', content,
              f"{bank} under the partial credit model, reported on its latent θ scale, which ranks people identically to its expected score but "
              "stretches both tails.", vkey, (T_GPCM[1], T_GPCM[0]), L_GPCM, mental))
# ---------------------------------------------------------------------------- the limitation banks with conditions, 15 September 2026
CONV = {'P-LIM1': ('SF6+F+C16', 'SF6+F+G'), 'P-LIM': ('FI15+C16', 'FI15+G'), 'P-LIM4': ('FI15+C16', 'FI15+G'),
        'P-LIM3': ('FI15+C16', 'FI15+G'), 'P-LIM3+O': ('L9+C16', 'L9+G')}
CC_TEXT = ("{bank} plus the count of the sixteen ever-diagnosed physical conditions as one testlet, from 0 to 5 and 6 or more; seven or more hold "
           "under half a percent of person-waves and merge into six. Conditions exist only from a person's first condition inventory, which almost "
           "no BHPS continuer receives. Reported on the test-characteristic-curve scale, from 0 to 1.")
CG_TEXT = ("{bank} plus P-FULL's six condition-group items: cardiovascular, diabetes and high blood pressure, and respiratory counted 0, 1, or 2 or "
           "more, and arthritis, cancer and other conditions as binary. Needs the condition inventory, which almost no BHPS continuer receives. "
           "Reported on the test-characteristic-curve scale, from 0 to 1.")
for bank, (v_cc, v_cg) in CONV.items():
    mental = 'Partly' if bank == 'P-LIM3+O' else 'Yes'
    for code, vkey, text in (('CC', v_cc, CC_TEXT), ('CG', v_cg, CG_TEXT)):
        nm = f'{bank}+{code}'
        M.append((f'{nm} h', f'{nm}, TCC', 'Graded-response model, TCC scale', 'SF-12 physical + limitations + conditions',
                  text.format(bank=bank), vkey, ("No", T_GRM[0]), L_GRM, mental))
        M.append((f'{nm} theta', f'{nm}, θ', 'Graded-response model, θ scale', 'SF-12 physical + limitations + conditions',
                  f"{nm} reported on its latent θ scale, which ranks people identically to its TCC score but stretches both tails.",
                  vkey, ("No", T_GRM[0]), L_GRM, mental))
# ---------------------------------------------------------------------------- the linear twins, 16 September 2026
TWIN_ITEMS = {'P-FUNC': ('SF6+F', 'the four SF-12 testlets and the limitation count capped at three'),
              'P-LIM': ('FI15', 'the four SF-12 testlets, the limitation count 0 to 5 and the sensory and continence count'),
              'P-LIM3': ('FI15', 'the four SF-12 testlets, functional limitations, self-care and sensory'),
              'P-LIM3+CC': ('FI15+C16', 'the four SF-12 testlets, functional limitations, self-care, sensory and the condition count')}
FS_TEXT = ("A one-factor maximum-likelihood factor analysis of the Pearson correlations of {items}, the same codes the graded-response model is "
           "fitted to. Each standardised testlet is weighted by its loading over its uniqueness, the weights are normalised to sum to one, and the "
           "score is re-standardised. It is the transparent twin of {bank}: the graded-response score regressed on these codes has an R-squared of "
           "0.993, and the two rank people almost identically.")
SUM_TEXT = ("The plain sum of the same testlet codes, equal weights and no standardisation: the most transparent index these items allow. It "
            "reproduces {bank}'s criteria, but takes only a few dozen distinct values, so it is coarse.")
for bank, (vkey, items) in TWIN_ITEMS.items():
    mental = 'Yes'
    M.append((f'{bank} FS', f'{bank}, Pearson factor score', 'Factor scores', 'SF-12 physical + limitations' if '+CC' not in bank else 'SF-12 physical + limitations + conditions',
              FS_TEXT.format(items=items, bank=bank), vkey, ("Yes", T_LIN[0]), L_FS, mental))
    M.append((f'{bank} sum', f'{bank}, equal-weight sum', 'Weighted subscales', 'SF-12 physical + limitations' if '+CC' not in bank else 'SF-12 physical + limitations + conditions',
              SUM_TEXT.format(bank=bank), vkey, ("Yes", "A sum of the answers; one sentence."),
              ('Partly', "Sum scores are the standard simple scoring of an item bank, and the partial credit model makes the weighted version exact."), mental))
assert len(M) == 76 and set(k for k, *_ in M) == set(df['key'])

# ---------------------------------------------------------------------------- columns
C = []  # (id, group, header, width, number format, definition)
def col(i, g, h, w, fmt=None, d=None): C.append(dict(id=i, group=g, header=h, width=w, fmt=fmt, defn=d))
G1, G2, G3, G4, G5, G6, G7, G8, G9 = ('Measure', 'Scorecard', 'Sample', 'Continuity, ceiling and floor', 'Healthcare cost, use and mortality',
                                      'Age profile', 'Identification: Case 3, four ages two years apart, balanced moments',
                                      'Identification: Case 2, four ages two years apart, pairwise moments', 'Judgement notes')
col('name', G1, 'Measure', 30); col('family', G1, 'Family', 20); col('content', G1, 'Content', 22)
col('method', G1, 'How it is built', 72); col('variables', G1, 'Variables used', 52)
col('sc_cont', G2, 'Continuous', 11, None, 'Formula. Yes if distinct values are at or above the first threshold on the Notes sheet, Partly if at or above the second, otherwise No.')
col('sc_floor', G2, 'No binding floor', 11, None, 'Formula. Uses the share at the worst score among people aged 60 to 84 against the Notes thresholds.')
col('sc_convex', G2, 'Convex link to healthcare costs', 13, None, 'Formula. Yes if cost curvature stays significantly positive with costs capped; Partly if only the uncapped curvature is.')
col('sc_transp', G2, 'Transparent', 11, None, 'Judgement. Whether the construction fits in a couple of paragraphs; see the transparency note.')
col('sc_estab', G2, 'Established in literature', 12, None, 'Judgement. Whether the construction is a standard, citable method; see the literature anchor.')
col('sc_mental', G2, 'Excludes mental health', 11, None, 'Judgement. No for the PCS, which gives mental subscales negative weights; Partly for P-LIM3+O, whose non-specific item may carry mental health.')
col('sc_panel', G2, 'Keeps full panel', 11, None, 'Formula. Yes if the measure keeps at least the Notes share of person-years of people first interviewed at wave 2, mostly BHPS continuers.')
col('sc_ident', G2, 'Identification barchart', 12, None, 'Formula. Yes if admissible in all four bands with the sign path under both estimators; Partly if under one.')
col('sc_exh1', G2, 'Fits Exhibit 1', 11, None, 'Formula. Yes if variance growth from 60 to 85 is inside the Notes flattening band and Case 3 gives the sign path; Partly if one holds.')
col('sc_count', G2, 'Criteria met, of 9', 10, '0', 'Formula. Number of Yes in the nine scorecard columns, unweighted.')
col('person_years', G3, 'Person-years, ages 20–90', 12, '#,##0', "Person-waves aged 20 to 90 with the measure observed.")
col('persons', G3, 'People', 10, '#,##0', 'Distinct people with the measure observed at least once.')
col('share_of_phys_person_years', G3, 'Share of phys person-years', 11, '0.0%', 'Person-years relative to phys.')
col('share_wave2_entrants_kept', G3, "Share of wave-2 entrants' person-years kept", 13, '0.0%', 'Person-years of people first interviewed at wave 2, mostly BHPS continuers, relative to phys.')
col('obs_per_person', G3, 'Observations per person', 11, '0.00', 'Mean number of observed person-waves per person.')
col('distinct_values', G4, 'Distinct values', 10, '#,##0', 'Number of distinct values the measure takes.')
col('best_all', G4, 'At best score, all ages', 10, '0.0%', 'Share of person-years at the maximum value.')
col('best_20_39', G4, 'At best score, ages 20–39', 10, '0.0%', 'Share of person-years aged 20 to 39 at the maximum value.')
col('worst_all', G4, 'At worst score, all ages', 10, '0.00%', 'Share of person-years at the minimum value.')
col('worst_60_84', G4, 'At worst score, ages 60–84', 10, '0.00%', 'Share of person-years aged 60 to 84 at the minimum value.')
col('distinct_bottom1', G4, 'Distinct values in bottom 1%', 11, '#,##0', 'Distinct values among person-years at or below the 1st percentile.')
col('distinct_bottom_decile_85_90', G4, 'Distinct values in bottom decile, ages 85–90', 13, '#,##0', "Distinct values in the bottom decile of the measure among people aged 85 to 90.")
col('rank_corr_phys', G4, 'Rank correlation with phys', 11, '0.000', 'Spearman correlation with phys on person-years where both are observed.')
col('cost_curv', G5, 'Cost curvature, £ per sd²', 11, '#,##0', 'Quadratic term from regressing the flat cost index in pounds on the standardised measure and its square; positive is convex.')
col('cost_curv_t', G5, 'Cost curvature t, clustered by person', 12, '0.0', 't-statistic on the quadratic term with standard errors clustered by person.')
col('cost_curv_capped', G5, 'Cost curvature, costs capped at 99th percentile, £ per sd²', 14, '#,##0', 'As the cost curvature, with person-wave costs capped at their 99th percentile.')
col('cost_curv_capped_t', G5, 'Capped cost curvature t', 11, '0.0', 't-statistic on the capped quadratic term, clustered by person.')
col('cost_log_curv', G5, 'Log-scale cost curvature', 11, '+0.000;-0.000;0.000', 'Quadratic term from fitting log mean cost across the ten deciles of the measure; positive means the proportional gradient steepens as health worsens.')
col('cost_nonconvex_steps', G5, 'Non-convex decile steps, of 8', 11, '0', 'Number of adjacent pairs of decile-to-decile cost slopes that violate convexity; a noisy straight line gives about 4.')
col('cost_ratio', G5, 'Cost, worst vs best decile', 11, '0.0"×"', 'Mean cost in the least healthy decile divided by the healthiest decile.')
col('inpatient_curv', G5, 'In-patient admission curvature', 11, '+0.000;-0.000;0.000', 'Quadratic term from a linear probability model of any in-patient stay on the standardised measure and its square.')
col('inpatient_curv_t', G5, 'In-patient admission curvature t', 11, '0.0', 't-statistic on the in-patient quadratic term, clustered by person.')
col('mort_slope', G5, 'Mortality log-odds slope per sd', 11, '0.00', 'Logit slope of death by the next wave on the standardised measure.')
col('mort_ratio', G5, 'Mortality, worst vs best decile', 11, '0.0"×"', 'Death rate by the next wave in the least healthy decile divided by the healthiest decile. Sensitive to how ties at the top are split.')
col('decline_30_50', G6, 'Mean change per decade, ages 30–50, sd', 12, '+0.00;-0.00', 'Change in the standardised mean per decade between ages 30 to 34 and 45 to 49.')
col('decline_70_85', G6, 'Mean change per decade, ages 70–85, sd', 12, '+0.00;-0.00', 'Change in the standardised mean per decade between ages 70 to 74 and 85 to 89.')
col('var_growth_30_60', G6, 'Variance growth, 30 to 60', 10, '0.00', 'Variance at ages 60 to 64 divided by variance at ages 30 to 34.')
col('var_growth_60_85', G6, 'Variance growth, 60 to 85', 10, '0.00', 'Variance at ages 85 to 89 divided by variance at ages 60 to 64; near 1 means the variance flattens.')
for sp, g, lab in (('c3', G7, 'Case 3'), ('c2', G8, 'Case 2')):
    col(f'{sp}_admissible', g, 'Admissible in all four bands', 11, None, f'{lab}: the deterministic block is a covariance matrix and every noise variance is non-negative in all four bands.')
    col(f'{sp}_signpath', g, 'Sign path', 9, None, f'{lab}: Corr(H, d) positive in the youngest band and negative in the oldest.')
    for b in ('25-40', '40-60', '60-75', '75-90'):
        col(f'{sp}_corr_{b}', g, f'Corr(H, d), {b.replace("-", "–")}', 9, '+0.00;-0.00', f'{lab}: level-slope correlation in the band.')
G10 = 'Cost gradient by age'
col('log_20-34', G10, 'Log cost gradient per sd, ages 20–34', 11, '0.00', 'Log points of cost per standard deviation sicker, ages 20 to 34, person-clustered; from descriptives/baseline_measures/13_cost_age_invariance.py.')
col('log_75-90', G10, 'Log cost gradient per sd, ages 75–90', 11, '0.00', 'The same at ages 75 to 90.')
col('log_ratio_old_young', G10, 'Gradient ratio, 75–90 over 20–34', 11, '0.00', 'One means the proportional health-cost relationship is age-invariant; below one it weakens with age.')
col('log_interaction_t', G10, 'Health × age interaction t, log cost', 11, '0.0', 't-statistic on health interacted with age in decades, in a regression of log cost on health, health squared and age terms.')
col('transp_note', G9, 'Transparency note', 40); col('lit_anchor', G9, 'Literature anchor', 52)
L = {c['id']: get_column_letter(i + 1) for i, c in enumerate(C)}

wb = Workbook(); ws = wb.active; ws.title = 'Measures'
wsi = wb.create_sheet('Identification'); notes = wb.create_sheet('Notes'); wsw = wb.create_sheet('Weights'); wsn = wb.create_sheet('Not run')

# ---------------------------------------------------------------------------- Notes: thresholds first, so formulas can point at them
notes.column_dimensions['A'].width = 34; notes.column_dimensions['B'].width = 70; notes.column_dimensions['C'].width = 12; notes.column_dimensions['D'].width = 60
notes['A1'] = 'Baseline physical health measure: criteria comparison'; notes['A1'].font = f(True, 14)
notes['A2'] = 'UKHLS waves 1 to 15. Built 14 September 2026 from the prevention-health-clustering processed data and the redo_concepts panel; the 42 limitation-bank rows, shaded blue, added 15 September 2026.'; notes['A2'].font = f(italic=True)
notes['A4'] = 'Scorecard thresholds'; notes['A4'].font = f(True, 12)
for j, h in enumerate(['Scorecard column', 'Rule', 'Threshold', 'Why this value']): 
    c = notes.cell(5, j + 1, h); c.font = f(True); c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
TH = [('thr_cont_yes', 'Continuous', 'Yes if distinct values are at least', 1000, '0', 'Enough values to behave as continuous in regressions and mixture models.'),
      ('thr_cont_partly', 'Continuous', 'Partly if distinct values are at least', 100, '0', 'Coarse but usable.'),
      ('thr_floor_yes', 'No binding floor', 'Yes if the share at the worst score, ages 60–84, is below', 0.005, '0.0%', 'Below half a percent the bottom is not a mass point.'),
      ('thr_floor_partly', 'No binding floor', 'Partly if that share is below', 0.01, '0.0%', 'Up to one percent the floor is modest.'),
      ('thr_t', 'Convex link to healthcare costs', 't-statistic threshold for the capped and uncapped cost curvature', 2, '0.0', 'Conventional significance at about the 5% level.'),
      ('thr_panel', 'Keeps full panel', "Yes if the share of wave-2 entrants' person-years kept is at least", 0.95, '0%', 'Losing more than a twentieth of the longest panels is material for the covariance surface.'),
      ('thr_flat_lo', 'Fits Exhibit 1', 'Variance flattens if growth from 60 to 85 is at least', 0.90, '0.00', 'The narrative says variance flattens off after 60.'),
      ('thr_flat_hi', 'Fits Exhibit 1', 'and at most', 1.10, '0.00', 'Growth above this is still rising; below the lower bound it is falling.')]
REF = {}
for i, (key, sc, rule, val, fmt, why) in enumerate(TH):
    r = 6 + i
    for j, v in enumerate([sc, rule, val, why]):
        c = notes.cell(r, j + 1, v); c.font = f(color='0000FF' if j == 2 else '000000'); c.border = BORDER; c.alignment = WRAP
    notes.cell(r, 3).number_format = fmt
    REF[key] = f"Notes!$C${r}"
notes.cell(6 + len(TH), 1, 'Blue values are editable; the scorecard on the Measures sheet recalculates from them.').font = f(italic=True, color='0000FF')

# ---------------------------------------------------------------------------- Measures sheet
groups = []
for c in C:
    if not groups or groups[-1][0] != c['group']: groups.append([c['group'], c['id'], c['id']])
    else: groups[-1][2] = c['id']
GFILL = {G1: '1F3864', G2: '375623', G3: '404040', G4: '7F6000', G5: '833C0B', G6: '3A3A7A', G7: '1F4E79', G8: '1F4E79', G9: '595959', G10: '833C0B'}
for g, a, b in groups:
    ws.merge_cells(f"{L[a]}1:{L[b]}1"); c = ws[f"{L[a]}1"]; c.value = g
    c.font = f(True, 10, 'FFFFFF'); c.fill = PatternFill('solid', fgColor=GFILL[g]); c.alignment = Alignment(horizontal='left', vertical='center')
for i, cdef in enumerate(C):
    c = ws.cell(2, i + 1, cdef['header']); c.font = f(True); c.alignment = CTR if cdef['group'] != G1 else WRAP
    c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
    ws.column_dimensions[get_column_letter(i + 1)].width = cdef['width']
    if cdef['defn']: c.comment = Comment(cdef['defn'], 'Comparison'); c.comment.width = 300; c.comment.height = 110
ws.row_dimensions[1].height = 20; ws.row_dimensions[2].height = 64

BLOCK_FILL = {'SF-12, physical and mental': 'FFFFFF', 'SF-12 physical': 'F2F2F2', 'SF-12 physical + limitations': 'FFFFFF', 'SF-12 physical + limitations + conditions': 'F2F2F2'}
first_sc, last_sc = L['sc_cont'], L['sc_exh1']
for n, (key, name, family, content, method, vkey, transp, lit, mental) in enumerate(M):
    r = 3 + n; row = df.set_index('key').loc[key]
    vals = {'name': name, 'family': family, 'content': content, 'method': method, 'variables': V[vkey],
            'sc_transp': transp[0], 'sc_estab': lit[0], 'sc_mental': mental, 'transp_note': transp[1], 'lit_anchor': lit[1]}
    for cdef in C:
        i = cdef['id']
        if i in vals: v = vals[i]
        elif i.startswith('sc_'): v = None
        else:
            v = row[i]
            if isinstance(v, (np.integer,)): v = int(v)
            elif isinstance(v, (np.floating, float)): v = float(v)
        cell = ws[f"{L[i]}{r}"]
        if i == 'sc_cont': cell.value = f'=IF({L["distinct_values"]}{r}>={REF["thr_cont_yes"]},"Yes",IF({L["distinct_values"]}{r}>={REF["thr_cont_partly"]},"Partly","No"))'
        elif i == 'sc_floor': cell.value = f'=IF({L["worst_60_84"]}{r}<{REF["thr_floor_yes"]},"Yes",IF({L["worst_60_84"]}{r}<{REF["thr_floor_partly"]},"Partly","No"))'
        elif i == 'sc_convex': cell.value = f'=IF({L["cost_curv_capped_t"]}{r}>{REF["thr_t"]},"Yes",IF({L["cost_curv_t"]}{r}>{REF["thr_t"]},"Partly","No"))'
        elif i == 'sc_panel': cell.value = f'=IF({L["share_wave2_entrants_kept"]}{r}>={REF["thr_panel"]},"Yes","No")'
        elif i == 'sc_ident':
            a3, s3, a2, s2 = (f'{L[x]}{r}' for x in ('c3_admissible', 'c3_signpath', 'c2_admissible', 'c2_signpath'))
            cell.value = f'=IF(AND({a3}="Yes",{s3}="Yes",{a2}="Yes",{s2}="Yes"),"Yes",IF(OR(AND({a3}="Yes",{s3}="Yes"),AND({a2}="Yes",{s2}="Yes")),"Partly","No"))'
        elif i == 'sc_exh1':
            vg, s3 = f'{L["var_growth_60_85"]}{r}', f'{L["c3_signpath"]}{r}'
            flat = f'AND({vg}>={REF["thr_flat_lo"]},{vg}<={REF["thr_flat_hi"]})'
            cell.value = f'=IF(AND({flat},{s3}="Yes"),"Yes",IF(OR({flat},{s3}="Yes"),"Partly","No"))'
        elif i == 'sc_count': cell.value = f'=COUNTIF({first_sc}{r}:{last_sc}{r},"Yes")'
        else: cell.value = v
        cell.font = f(bold=(i in ('name', 'sc_count')), color='0000FF' if i in ('sc_transp', 'sc_estab', 'sc_mental') else '000000')
        cell.border = BORDER
        cell.alignment = WRAP if i in ('name', 'family', 'content', 'method', 'variables', 'transp_note', 'lit_anchor') else Alignment(horizontal='center', vertical='top')
        if cdef['fmt']: cell.number_format = cdef['fmt']
        if i in ('name', 'family', 'content'): cell.fill = PatternFill('solid', fgColor='E8EEF7' if n >= 26 else BLOCK_FILL[content])
    lines = max(math.ceil(len(method) / 78), sum(math.ceil(len(p) / 56) for p in V[vkey].split('\n')), math.ceil(len(lit[1]) / 56), 2)
    ws.row_dimensions[r].height = min(409, 13.2 * lines + 6)
last_row = 2 + len(M); last_col = get_column_letter(len(C))
green, amber, red = (PatternFill('solid', fgColor=x) for x in ('C6EFCE', 'FFEB9C', 'FFC7CE'))
for rng in (f"{L['sc_cont']}3:{L['sc_exh1']}{last_row}", f"{L['c3_admissible']}3:{L['c3_signpath']}{last_row}", f"{L['c2_admissible']}3:{L['c2_signpath']}{last_row}"):
    ws.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"Yes"'], fill=green, font=Font(name=FONT, color='006100')))
    ws.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"Partly"'], fill=amber, font=Font(name=FONT, color='9C5700')))
    ws.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"No"'], fill=red, font=Font(name=FONT, color='9C0006')))
ws.freeze_panes = 'B3'; ws.auto_filter.ref = f"A2:{last_col}{last_row}"

# ---------------------------------------------------------------------------- Notes: samples, definitions, caveats
r = 6 + len(TH) + 2
def section(title):
    global r
    notes.cell(r, 1, title).font = f(True, 12); r += 1
def line(a, b=''):
    global r
    ca = notes.cell(r, 1, a); cb = notes.cell(r, 2, b); ca.font = f(True); cb.font = f(); ca.alignment = WRAP; cb.alignment = WRAP
    notes.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4); notes.row_dimensions[r].height = max(15, 13.2 * math.ceil(len(b) / 150) + 4); r += 1
section('Samples')
line('Sample, distribution and age profile', "Each measure's own person-waves aged 20 to 90.")
line('Healthcare cost and in-patient use', "210,389 person-waves and 44,941 people in waves 7 to 15 where all 68 measures and the cost index are observed. Cost is the flat-rate cost index, flat_cost_total, built 4 September 2026 by data_cleaning/09_build_cost_proxy.py: in-patient spells and excess bed days, out-patient attendances and GP visits at national average unit costs, maternity excluded. The condition-weighted variant gives the same comparison.")
line('Mortality', "385,427 person-waves where all 68 measures are observed and death by the next wave is known; 1,800 deaths.")
line('Identification', "People observed in at least four waves, one row per person and age. Four ages two years apart, pooled over base ages within the bands 25 to 40, 40 to 60, 60 to 75 and 75 to 90; the AR(1) persistence is profiled over 0.05 to 0.97. Case 3 uses balanced moments, Case 2 pairwise moments.")
line('Cost gradient by age', "Added 16 September 2026. The proportional gradient is log points of cost per standard deviation sicker, fitted band by band on each measure's own cost sample with person-clustered standard errors, as in descriptives/17_cost_age_interaction.py. A ratio near one means a unit of health is worth the same proportional amount at every age; the h scales run about 0.7 and the theta scales about 1.0.")
line('Linear twins', "Added 16 September 2026. For P-FUNC, P-LIM, P-LIM3 and P-LIM3+CC the same testlet codes are also scored as a Pearson one-factor weighted sum and as a plain equal-weight sum, so the graded-response conclusions can be checked against a transparent index on identical variables.")
line('Identification sheet', "Cases 2, 3 and 4 each on balanced and on pairwise moments, for every measure, added 15 September 2026. Case 2 has one AR(1) noise variance; Case 3 frees it at each age; Case 4 is Case 2 plus a one-period error on the diagonal, profiled over 0.30 to 0.97 as in testing_metrics, and admissible only if both noise variances are non-negative. Every entry is at the best-fitting persistence; for Case 4 two more columns use the best-fitting admissible persistence in each band. The sign path reads the sign of C[H,d], so it is defined even where the correlation is not. The scorecard still uses only Case 3 on balanced and Case 2 on pairwise moments.")
r += 1; section('Column definitions')
for cdef in C:
    if cdef['defn']: line(cdef['header'], cdef['defn'])
line('Transparent, Established in literature, Excludes mental health', 'Judgement inputs in blue on the Measures sheet, explained in the transparency note and literature anchor columns. Edit them and the scorecard count updates.')
r += 1; section('Caveats')
line('Scorecard', 'A screening device with the thresholds above, not a ranking. Criteria met counts every criterion equally, which is a choice, not a recommendation.')
line('t-statistics', 'Clustered by person. Unclustered versions quoted in earlier discussion were about three times larger.')
line('Decile ratios', 'Measures with many tied values at the top, such as phys, split ties by average rank, which moves the ratios by several points.')
line('Ever-diagnosis items', 'Conditions are cumulative ever-diagnosed indicators, available only from the first condition inventory, which excludes almost all BHPS continuers.')
line('Formulas', 'The scorecard columns and the condition totals on the Weights sheet are formulas. Their results are stored in the file, so previewers show them, and Excel recalculates them on opening, including after you edit a threshold or a judgement cell.')
line('Limitation banks', "Six banks, each the four SF-12 testlets plus counts over the UKHLS impairment areas, fitted under the graded-response model and the generalised partial credit model by data_cleaning/07_build_limitation_banks.py. Anyone without a long-standing illness counts as having no limitation in every wave, and a top count category under half a percent of person-waves merges into the one below. All six share P-FUNC's person-waves, so adding them leaves the cost and mortality samples unchanged. The measures note's section on these banks has their figures, fit comparison and weights. Each limitation bank except P-FUNC is also fitted, under the graded-response model only, with the chronic conditions added in two ways: +CC, one count of the sixteen ever-diagnosed physical conditions, 0 to 5 and 6 or more; and +CG, P-FULL's six group items. These need the condition inventory, so they use P-FULL's person-waves; P-FUNC with the six groups is P-FULL itself, reproduced exactly.")
line('Reproducibility', 'prevention-health-clustering/descriptives/baseline_measures, scripts 01 to 11, recovered on 15 September 2026 from the session scratch scripts that built the first version; rerunning them reproduces every earlier cell exactly. Intermediate files are in data/processed/baseline_measures, which is not committed.')

# ---------------------------------------------------------------------------- Identification sheet
wsi['A1'] = 'Identification under three noise specifications and two kinds of moments'; wsi['A1'].font = f(True, 14)
wsi['A2'] = ("Balanced moments use people observed at all four ages; pairwise moments use, for each cell, everyone observed at both of its ages. "
             "Sign path: Corr(H, d) positive at 25 to 40 and negative at 75 to 90. Correlations are blank where V[H] and V[d] have opposite signs. See the Notes sheet.")
wsi['A2'].font = f(italic=True); wsi['A2'].alignment = WRAP; wsi.merge_cells('A2:T2'); wsi.row_dimensions[2].height = 30
IDB = [('c2b', 'Case 2, balanced'), ('c3', 'Case 3, balanced (scorecard)'), ('c4b', 'Case 4, balanced'),
       ('c2', 'Case 2, pairwise (scorecard)'), ('c3p', 'Case 3, pairwise'), ('c4p', 'Case 4, pairwise')]
SUB = [('admissible', 'Admissible in all four bands', None), ('signpath', 'Sign path', None),
       ('corr_25-40', 'Corr(H, d), 25–40', '+0.00;-0.00'), ('corr_40-60', 'Corr(H, d), 40–60', '+0.00;-0.00'),
       ('corr_60-75', 'Corr(H, d), 60–75', '+0.00;-0.00'), ('corr_75-90', 'Corr(H, d), 75–90', '+0.00;-0.00')]
SUB4 = SUB + [('admissible_validrho', 'Admissible at a valid ρ', None), ('signpath_validrho', 'Sign path at a valid ρ', None)]
c = wsi.cell(5, 1, 'Measure'); c.font = f(True); c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER; c.alignment = WRAP
wsi.column_dimensions['A'].width = 34
col_i = 2; yn_ranges = []
for sp, title in IDB:
    subs = SUB4 if sp.startswith('c4') else SUB
    wsi.merge_cells(start_row=4, start_column=col_i, end_row=4, end_column=col_i + len(subs) - 1)
    g = wsi.cell(4, col_i, title); g.font = f(True, 10, 'FFFFFF'); g.alignment = Alignment(horizontal='left', vertical='center')
    g.fill = PatternFill('solid', fgColor='1F4E79' if 'balanced' in title else '3A3A7A')
    for k, (sub, head, fmt) in enumerate(subs):
        cc = col_i + k
        h = wsi.cell(5, cc, head); h.font = f(True); h.alignment = CTR; h.fill = PatternFill('solid', fgColor='D9E1F2'); h.border = BORDER
        wsi.column_dimensions[get_column_letter(cc)].width = 11
        for n, (key, name, *_) in enumerate(M):
            v = df.set_index('key').loc[key, f'{sp}_{sub}']
            if isinstance(v, (np.floating, float)):
                v = None if np.isnan(v) else float(v)
            cell = wsi.cell(6 + n, cc, v); cell.border = BORDER; cell.font = f(); cell.alignment = Alignment(horizontal='center', vertical='top')
            if fmt: cell.number_format = fmt
        if fmt is None:
            yn_ranges.append(f"{get_column_letter(cc)}6:{get_column_letter(cc)}{5 + len(M)}")
    col_i += len(subs)
for n, (key, name, *_) in enumerate(M):
    c = wsi.cell(6 + n, 1, name); c.font = f(True); c.border = BORDER; c.alignment = WRAP
    c.fill = PatternFill('solid', fgColor='E8EEF7' if n >= 26 else 'FFFFFF')
wsi.row_dimensions[5].height = 44
for rng in yn_ranges:
    wsi.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"Yes"'], fill=green, font=Font(name=FONT, color='006100')))
    wsi.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"No"'], fill=red, font=Font(name=FONT, color='9C0006')))
wsi.freeze_panes = 'B6'

# ---------------------------------------------------------------------------- Weights sheet
ITEMS = [('GH', 'General health'), ('PF', 'Physical functioning'), ('RP', 'Role-physical'), ('BP', 'Bodily pain'), ('FUNC', 'Limitation count, capped at 3'),
         ('CVD', 'Heart disease and stroke'), ('METAB', 'Diabetes and high blood pressure'), ('RESP', 'Asthma, bronchitis, emphysema'), ('MSK', 'Arthritis'),
         ('CANCER', 'Cancer'), ('OTHER', 'Thyroid, liver, epilepsy')]
wsw['A1'] = 'Item weights'; wsw['A1'].font = f(True, 14)
wsw['A2'] = 'Weights on standardised items, summing to one within each index. phys weights its four subscales, which are built from the same six items as the testlets.'; wsw['A2'].font = f(italic=True)
IDX = [('phys', None)] + [(nm, k) for k, nm, *_ in M if k.startswith(('PC ', 'FS '))]
wsw.cell(4, 1, 'Item').font = f(True); wsw.column_dimensions['A'].width = 32
phys_w = {'PF': 0.285, 'RP': 0.268, 'BP': 0.227, 'GH': 0.219}
for j, (nm, k) in enumerate(IDX):
    c = wsw.cell(4, j + 2, nm); c.font = f(True); c.alignment = CTR; c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
    wsw.column_dimensions[get_column_letter(j + 2)].width = 15
wsw.row_dimensions[4].height = 60
for i, (it, lab) in enumerate(ITEMS):
    rr = 5 + i; wsw.cell(rr, 1, lab).font = f(); wsw.cell(rr, 1).border = BORDER
    for j, (nm, k) in enumerate(IDX):
        if k is None: v = phys_w.get(it)
        else:
            s = W[(W['index'] == k) & (W['item'] == it)]; v = float(s['weight'].iloc[0]) if len(s) else None
        c = wsw.cell(rr, j + 2, v); c.number_format = '0.000'; c.font = f(); c.border = BORDER; c.alignment = Alignment(horizontal='center')
tot = 5 + len(ITEMS)
wsw.cell(tot, 1, 'Condition groups, total').font = f(True)
for j, (nm, k) in enumerate(IDX):
    c = wsw.cell(tot, j + 2); c.border = BORDER
    if k is not None and k.endswith('SF+F+C'):
        cl = get_column_letter(j + 2); c.value = f'=SUM({cl}10:{cl}15)'; c.number_format = '0%'; c.font = f(True); c.alignment = Alignment(horizontal='center')
rr = tot + 3
wsw.cell(rr, 1, 'Graded-response discriminations').font = f(True, 12); rr += 1
wsw.cell(rr, 1, 'Not weights, but they play the role of loadings: higher means the item separates people more sharply. From grm_items.csv and grm2_items.csv.').font = f(italic=True); rr += 1
g1 = pd.read_csv(R0 / 'data/processed/measures/grm_items.csv').assign(spec='GRM original')
g2 = pd.read_csv(R0 / 'data/processed/measures/grm2_items.csv'); g2 = g2[g2['spec'].isin(['P-FUNC', 'P-FULL'])]
G = pd.concat([g1[['spec', 'item', 'a']], g2[['spec', 'item', 'a']]])
for j, s in enumerate(['GRM original', 'P-FUNC', 'P-FULL']):
    c = wsw.cell(rr, j + 2, s); c.font = f(True); c.alignment = CTR; c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
wsw.cell(rr, 1, 'Item').font = f(True); rr += 1
for it, lab in ITEMS:
    wsw.cell(rr, 1, lab).font = f(); wsw.cell(rr, 1).border = BORDER
    for j, s in enumerate(['GRM original', 'P-FUNC', 'P-FULL']):
        q = G[(G['spec'] == s) & (G['item'] == it)]; c = wsw.cell(rr, j + 2, float(q['a'].iloc[0]) if len(q) else None)
        c.number_format = '0.00'; c.font = f(); c.border = BORDER; c.alignment = Alignment(horizontal='center')
    rr += 1
rr += 2; wsw.cell(rr, 1, 'Implied weights in the equal-weight PHYS indices').font = f(True, 12); rr += 1
for a, b in (('One limitation', 'About 25 points of the 0 to 100 physical subscale mean, whose standard deviation is 24.3.'),
             ('One ever-diagnosed condition', 'About 20 points of the subscale mean, in PHYS with conditions.'),
             ('Frailty indices', 'Each deficit weighs 1/15 in the 15-deficit index and 1/31 in the 31-deficit index; the SF-12 items are graded from 0 to 1 by response step.')):
    wsw.cell(rr, 1, a).font = f(True); wsw.cell(rr, 2, b).font = f(); wsw.merge_cells(start_row=rr, start_column=2, end_row=rr, end_column=10); rr += 1

rr += 2; wsw.cell(rr, 1, 'Limitation banks: implied weights').font = f(True, 12); rr += 1
wsw.cell(rr, 1, "Shares of weight on standardised testlets, summing to one within each method, and the two models' slopes. GRM latent weights convert each discrimination to a loading on the underlying response, a over the square root of a squared plus 1.702 squared, and weight by loading over uniqueness; the polychoric factor weights are its twin. The partial credit model's exact weights are its slopes times the testlet standard deviation. Projections regress each model's 0 to 1 score on the standardised codes; the Pearson factor weights and first component are their linear twins. From descriptives/baseline_measures/10_limitation_note_outputs.py.").font = f(italic=True)
wsw.merge_cells(start_row=rr, start_column=1, end_row=rr, end_column=10); wsw.row_dimensions[rr].height = 58; wsw.cell(rr, 1).alignment = WRAP; rr += 2
LW = pd.read_csv(SCR / 'limitation_weights.csv')
LBANKS = {'P-FUNC': ['FUNC'], 'P-LIM1': ['LIM_PF'], 'P-LIM': ['LIM_PF', 'LIM_SC'], 'P-LIM4': ['LIM_MOB', 'LIM_DEX', 'LIM_CONT', 'LIM_SENS'],
          'P-LIM3': ['LIM_FL', 'LIM_SELF', 'LIM_SENS'], 'P-LIM3+O': ['LIM_FL', 'LIM_SELF', 'LIM_SENS', 'LIM_OTHER']}
LLAB = {'GH': 'General health', 'PF': 'Physical functioning', 'RP': 'Role-physical', 'BP': 'Bodily pain', 'FUNC': 'Limitations, capped 3+',
        'LIM_PF': 'Limitations, 0 to 5', 'LIM_SC': 'Sensory and continence', 'LIM_MOB': 'Mobility and lifting', 'LIM_DEX': 'Dexterity, coordination, care',
        'LIM_CONT': 'Continence', 'LIM_SENS': 'Hearing and sight', 'LIM_FL': 'Functional limitations', 'LIM_SELF': 'Self-care', 'LIM_OTHER': 'Other health problem'}
LMETH = [('grm_a', 'GRM discrimination', '0.00'), ('gpcm_a', 'Partial credit slope', '0.00'), ('grm_latent', 'GRM, latent', '0%'),
         ('polychoric_factor', 'Polychoric one-factor', '0%'), ('grm_projection', 'GRM score projected on codes', '0%'),
         ('gpcm_exact', 'Partial credit, exact', '0%'), ('gpcm_projection', 'Partial credit score projected on codes', '0%'),
         ('pearson_factor', 'Pearson one-factor', '0%'), ('pearson_pc1', 'Pearson first component', '0%')]
for bank, lims in LBANKS.items():
    names = ['GH', 'PF', 'RP', 'BP'] + lims
    c = wsw.cell(rr, 1, bank); c.font = f(True); c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
    for j, nm in enumerate(names):
        c = wsw.cell(rr, j + 2, LLAB[nm]); c.font = f(True); c.alignment = CTR; c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
    wsw.row_dimensions[rr].height = 30; rr += 1
    for key, lab, fmt in LMETH:
        wsw.cell(rr, 1, lab).font = f(); wsw.cell(rr, 1).border = BORDER
        vals = LW[(LW['bank'] == bank) & (LW['method'] == key)].set_index('item')['value']
        for j, nm in enumerate(names):
            c = wsw.cell(rr, j + 2, float(vals[nm])); c.number_format = fmt; c.font = f(); c.border = BORDER; c.alignment = Alignment(horizontal='center')
        rr += 1
    rr += 1

rr += 1; wsw.cell(rr, 1, 'Limitation banks with conditions: implied weights').font = f(True, 12); rr += 1
wsw.cell(rr, 1, "As above, graded-response model only; correlations and projections on the rows with a condition inventory.").font = f(italic=True); rr += 2
LLAB.update({'COND': 'Chronic conditions, count', 'CVD': 'Heart disease and stroke', 'METAB': 'Diabetes and high blood pressure',
             'RESP': 'Asthma, bronchitis, emphysema', 'MSK': 'Arthritis', 'CANCER': 'Cancer', 'OTHER': 'Thyroid, liver, epilepsy'})
CMETH = [('grm_a', 'GRM discrimination', '0.00'), ('grm_latent', 'GRM, latent', '0%'), ('polychoric_factor', 'Polychoric one-factor', '0%'),
         ('grm_projection', 'GRM score projected on codes', '0%'), ('pearson_factor', 'Pearson one-factor', '0%'),
         ('pearson_pc1', 'Pearson first component', '0%')]
CCODES = {'CC': ['COND'], 'CG': ['CVD', 'METAB', 'RESP', 'MSK', 'CANCER', 'OTHER']}
for bank in ['P-LIM1', 'P-LIM', 'P-LIM4', 'P-LIM3', 'P-LIM3+O']:
    for code, cond in CCODES.items():
        nm = f'{bank}+{code}'; names = ['GH', 'PF', 'RP', 'BP'] + LBANKS[bank] + cond
        c = wsw.cell(rr, 1, nm); c.font = f(True); c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
        for j, it in enumerate(names):
            c = wsw.cell(rr, j + 2, LLAB[it]); c.font = f(True); c.alignment = CTR; c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER
        wsw.row_dimensions[rr].height = 30; rr += 1
        for key, lab, fmt in CMETH:
            wsw.cell(rr, 1, lab).font = f(); wsw.cell(rr, 1).border = BORDER
            vals = LW[(LW['bank'] == nm) & (LW['method'] == key)].set_index('item')['value']
            for j, it in enumerate(names):
                c = wsw.cell(rr, j + 2, float(vals[it])); c.number_format = fmt; c.font = f(); c.border = BORDER; c.alignment = Alignment(horizontal='center')
            rr += 1
        rr += 1

# ---------------------------------------------------------------------------- Not run sheet
NR = [('SF-6D utility', 'Preference-based utility index scored from seven SF-12 items.', 'Includes mental health and vitality dimensions, discards most of the physical age gradient, and correlates more with the mental GRM than the physical one.', 'Empirics note, measure-choice appendix; measures note, how the measures relate.'),
      ('UKHLS MCS, GHQ-12 and the mental GRM', 'Mental health summaries.', 'Excluded by the no-mental-health criterion; kept for a separate mental channel.', 'Measures note.'),
      ('Combined GRM, seventeen items', 'One graded-response bank over physical, condition and mental items.', 'Fails one-dimensionality, with strong residual clustering among mental items, and carries a mental-health artefact of about 0.8 sd.', 'Measures note, one dimension or two.'),
      ('Correlated-factor PCS', 'Farivar and coauthors 2007 oblique scoring, or a UK promax equivalent.', 'Removes the negative mental weights but still gives vitality and social functioning positive weight.', 'prevention-health-clustering composites module.'),
      ('Equal-weight mean of all eight SF-12 subscales', 'Average of the physical and mental subscales.', 'Includes the mental subscales.', 'prevention-health-clustering composites module.'),
      ('Self-rated general health alone', 'The single sf1 item.', 'Five categories only, 13% at the best score, and a smaller age gradient.', 'Empirics note, measure-choice candidates table.'),
      ('Physical subscales without general health', 'Physical functioning, role-physical and bodily pain.', '38% of person-years at the best score, because those subscales saturate among the healthy.', 'Empirics note, measure-choice candidates table.'),
      ('Chronic condition count alone', 'Number of ever-diagnosed conditions.', 'Mostly zeros among the young; the count can only rise, so its covariance rows carry no persistence information; loses the BHPS continuers.', 'Measures note, moments grid.'),
      ('Limitation count alone', 'Number of Equality Act physical limitations.', 'About 82% of person-years at zero, and asked only of people with a long-standing illness.', 'This comparison.'),
      ('Physical subscales plus vitality', 'Adding sf6b, had a lot of energy.', 'Would cut the share at the best score to under 2%, but vitality loads almost equally on the mental factor.', 'This comparison; measures note loadings.'),
      ('P-REC and P-TIMED', 'P-FULL with diagnoses restricted to the last ten years, or split into recent and stale items.', 'Sensitivity variants for the ever-diagnosis question, not baseline candidates.', 'Measures note, the ever-diagnosis question.'),
      ('Health-stock index', 'Ordered probit of self-rated health on specific health problems, using the fitted index, as in Disney, Emmerson and Wakefield 2006 on the BHPS.', 'Not yet built; a candidate worth testing.', 'Literature discussion.'),
      ('Outcome-anchored index', 'Items weighted by how well they predict costs or mortality.', 'Would build cost into the measure, so it cannot test the convexity of costs in health.', 'Literature discussion.'),
      ('Biomarkers and physical performance', 'Grip strength, blood pressure and other nurse-visit measures.', 'Only in the nurse-visit waves, not a panel measure, and not held.', 'Measures note, variable inventory.'),
      ('Activities of daily living module', 'Difficulty with everyday tasks.', 'Asked only in waves 7, 9, 11 and 13.', 'Measures note, variable inventory.')]
wsn['A1'] = 'Measures considered but not run through the criteria'; wsn['A1'].font = f(True, 14)
for j, (h, w_) in enumerate([('Measure', 32), ('What it is', 50), ('Why not run', 70), ('Evidence', 40)]):
    c = wsn.cell(3, j + 1, h); c.font = f(True); c.fill = PatternFill('solid', fgColor='D9E1F2'); c.border = BORDER; wsn.column_dimensions[get_column_letter(j + 1)].width = w_
for i, rowv in enumerate(NR):
    for j, v in enumerate(rowv):
        c = wsn.cell(4 + i, j + 1, v); c.font = f(bold=(j == 0)); c.alignment = WRAP; c.border = BORDER
    wsn.row_dimensions[4 + i].height = 13.2 * max(math.ceil(len(rowv[2]) / 75), math.ceil(len(rowv[1]) / 55), 1) + 6
wsn.freeze_panes = 'A4'

wb.calculation = CalcProperties(fullCalcOnLoad=True)
wb.save(OUT)
print('saved', OUT)

# ---------------------------------------------------------------------------- verify every formula with a small evaluator
from openpyxl import load_workbook
wbv = load_workbook(OUT); m_ws = wbv['Measures']; n_ws = wbv['Notes']; w_ws = wbv['Weights']
def val(ref):
    sh, cell = (ref.split('!') + [None])[:2] if '!' in ref else ('Measures', ref)
    wsx = {'Measures': m_ws, 'Notes': n_ws, 'Weights': w_ws}[sh]; return wsx[cell.replace('$', '')].value
def cellval(sheet, coord):
    ws_ = {'Measures': m_ws, 'Weights': w_ws, 'Notes': n_ws}[sheet]; v = ws_[coord.replace('$', '')].value
    return evaluate(v, sheet) if isinstance(v, str) and v.startswith('=') else v
def evaluate(formula, sheet='Measures'):
    expr = formula[1:]
    expr = re.sub(r'\b([A-Z]{1,3})(\d+):([A-Z]{1,3})(\d+)\b',
                  lambda mo: repr([cellval(sheet, f'{get_column_letter(ci)}{ri}')
                                   for ri in range(int(mo.group(2)), int(mo.group(4)) + 1)
                                   for ci in range(column_index_from_string(mo.group(1)), column_index_from_string(mo.group(3)) + 1)]), expr)
    expr = re.sub(r"Notes!\$?([A-Z]+)\$?(\d+)", lambda mo: repr(cellval('Notes', f'{mo.group(1)}{mo.group(2)}')), expr)
    expr = re.sub(r'(?<![\w"\'])([A-Z]{1,3}\d+)\b', lambda mo: repr(cellval(sheet, mo.group(1))), expr)
    # Excel equality to Python, leaving >=, <= and already-doubled signs alone
    expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr)
    env = {'IF': lambda c, a, b: a if c else b, 'AND': lambda *x: all(x), 'OR': lambda *x: any(x),
           'COUNTIF': lambda xs, crit: sum(1 for x in xs if x == crit), 'SUM': lambda xs: sum(x for x in xs if isinstance(x, (int, float)))}
    return eval(expr, env)
checked = 0; expected_counts = {}
for rr_ in range(3, last_row + 1):
    for cdef in C:
        cell = m_ws[f"{L[cdef['id']]}{rr_}"]
        if isinstance(cell.value, str) and cell.value.startswith('='):
            res = evaluate(cell.value); checked += 1
            assert res in ('Yes', 'Partly', 'No') or isinstance(res, int), (cell.coordinate, cell.value, res)
            if cdef['id'] == 'sc_count': expected_counts[m_ws[f"A{rr_}"].value] = res
# independent Python mirror of the scorecard rules
th = {k: v for k, _, _, v, _, _ in TH}
mismatch = 0
for rr_ in range(3, last_row + 1):
    key = M[rr_ - 3][0]; row = df.set_index('key').loc[key]
    mirror = {
        'sc_cont': 'Yes' if row.distinct_values >= th['thr_cont_yes'] else 'Partly' if row.distinct_values >= th['thr_cont_partly'] else 'No',
        'sc_floor': 'Yes' if row.worst_60_84 < th['thr_floor_yes'] else 'Partly' if row.worst_60_84 < th['thr_floor_partly'] else 'No',
        'sc_convex': 'Yes' if row.cost_curv_capped_t > th['thr_t'] else 'Partly' if row.cost_curv_t > th['thr_t'] else 'No',
        'sc_panel': 'Yes' if row.share_wave2_entrants_kept >= th['thr_panel'] else 'No'}
    for k, exp in mirror.items():
        got = evaluate(m_ws[f"{L[k]}{rr_}"].value)
        if got != exp: mismatch += 1; print('MISMATCH', key, k, got, exp)
# header sanity: formulas point at the intended columns
for k, h in (('distinct_values', 'Distinct values'), ('worst_60_84', 'At worst score, ages 60–84'), ('cost_curv_capped_t', 'Capped cost curvature t'),
             ('share_wave2_entrants_kept', "Share of wave-2 entrants' person-years kept"), ('var_growth_60_85', 'Variance growth, 60 to 85'), ('c3_signpath', 'Sign path')):
    assert m_ws[f"{L[k]}2"].value == h, (k, m_ws[f"{L[k]}2"].value)
wcheck = [round(evaluate(w_ws.cell(tot, j + 2).value, 'Weights'), 3) for j in range(len(IDX)) if isinstance(w_ws.cell(tot, j + 2).value, str)]
print(f'formulas evaluated: {checked}; mirror mismatches: {mismatch}; weight-sheet condition totals: {wcheck}')
print('criteria met:', expected_counts)

# ---------------------------------------------------------------------------- cache the verified formula results in the file
import zipfile, shutil, html
SHEET_XML = {name: f"xl/worksheets/sheet{i + 1}.xml" for i, name in enumerate(wb.sheetnames)}
cache = {SHEET_XML['Measures']: {}, SHEET_XML['Weights']: {}}
for rr_ in range(3, last_row + 1):
    for cdef in C:
        cell = m_ws[f"{L[cdef['id']]}{rr_}"]
        if isinstance(cell.value, str) and cell.value.startswith('='): cache[SHEET_XML['Measures']][cell.coordinate] = evaluate(cell.value)
for j in range(len(IDX)):
    cell = w_ws.cell(tot, j + 2)
    if isinstance(cell.value, str) and cell.value.startswith('='): cache[SHEET_XML['Weights']][cell.coordinate] = evaluate(cell.value, 'Weights')
tmp = OUT.with_suffix('.tmp.xlsx')
with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename in cache:
            vals = cache[item.filename]; xml = data.decode('utf-8'); n_done = 0
            def sub(mo):
                global n_done
                coord, attrs, formula = mo.group(1), mo.group(2), mo.group(3)
                if coord not in vals: return mo.group(0)
                v = vals[coord]; n_done += 1
                if isinstance(v, str): return f'<c r="{coord}"{attrs} t="str"><f>{formula}</f><v>{html.escape(v)}</v></c>'
                return f'<c r="{coord}"{attrs}><f>{formula}</f><v>{v}</v></c>'
            xml = re.sub(r'<c r="([A-Z]+\d+)"([^>]*)><f>(.*?)</f><v></v></c>', sub, xml)
            assert n_done == len(vals), (item.filename, n_done, len(vals))
            data = xml.encode('utf-8')
        zout.writestr(item, data)
shutil.move(tmp, OUT)
chk = load_workbook(OUT, data_only=True)['Measures']
print('cached values written; sample:', [chk[f'{L[k]}3'].value for k in ('sc_cont', 'sc_floor', 'sc_convex', 'sc_panel', 'sc_ident', 'sc_exh1', 'sc_count')])
