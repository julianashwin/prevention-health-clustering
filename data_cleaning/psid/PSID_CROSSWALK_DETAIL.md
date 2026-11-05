# PSID crosswalk: detailed notes by domain

These are the notes from the second, independent pass over the codebooks (one analyst per domain). They record question wording in each wave, code changes, who was asked, and quirks found in the data. Variable choices here agree with `psid_crosswalk.csv` except where that file says otherwise.

## Self-rated health, work limitation, ADL, IADL, mobility, hospital stays, height and weight


### General
- **Source and layout.** All items are in section H of the family files (FAM1999ER to FAM2023ER) and are asked separately about the RP/Head and the SP/Wife.
  - 1999-2005: Wife items have their own question numbers: H25 SRH, H25a change, H26-H28 work limits, H32 hospital, H33/H34 ADL, H35 IADL, H46/H47 weight/height.
  - 2007+: Wife items reuse the Head numbers (H1-H23) after checkpoint H24 "WTR WIFE/SPOUSE IN FU NOW".
  - 1999-2001: Wife labels have no "-WIFE" suffix. They were identified by the question text ("she/her") and the section order.
- **Proxy reporting.** One family respondent answers for both people, so SP (and sometimes RP) data are often proxy reports. UKHLS uses self-report. The respondent can be identified with "WHO WAS RESPONDENT": ER13016 (1999), ER17019, ER24073, ER27879, ER40869, ER46697, ER52097, ER57901, ER65081, ER71164, ER77186, ER81522, ER85379 (2023).
- **Codes.** 1=Yes, 5=No, 8=DK, 9=NA/refused, 0=Inap throughout.
- **Verification.** For every reported variable, the codebook frequencies were checked against the parquet data. All 3,845 code cells match exactly. RP and SP series run in parallel and are stable across waves, apart from the universe breaks noted below.
- **2021 block non-response.** About 54 RPs and 43 SPs (mostly the same families) are coded 8=DK on essentially every H item. Treat these as missing, not as substantive DK.

### srh (self-rated general health)
- Variables: H1 (RP, all waves); H25 (SP 1999-2005); H1 (SP 2007+).
- The 5-category wording (excellent / very good / good / fair / poor) is identical in every wave. This matches the UKHLS GH item (SF-12 item 1). The main difference is proxy reporting for SP.
- Mean on the 1-5 scale drifts from 2.40 (1999) to 2.67 (2023) for RPs, consistent with sample ageing.

### srh_change2y
- Asked 2003+ only: H1a (SP H25a in 2003-05): "Compared to two years ago, is your health better, about the same, or worse?" (1/3/5).
- Follow-ups H1b/H1c (much/somewhat better or worse) are listed in the CSV notes.
- Not asked in 1999 or 2001. There is no UKHLS target counterpart.

### worklim_any / worklim_types / worklim_amount
- Variables: H2/H3/H4 (SP H26/H27/H28 in 1999-2005). Wording is unchanged 1999-2023.
- **worklim_any:** "Any physical or nervous condition that limits the type of work or the amount of work you can do". This is the closest PSID item to UKHLS SF-12 RP ("limited in the kind of work or other activities because of physical health"). It differs in three ways:
  - it has no 4-week reference period (it is a stock measure);
  - it covers work only;
  - it includes "nervous" (mental) conditions.
- **worklim_types (H3):** asked if H2=1. Code 7 = "can do nothing" (cannot work at all).
- **worklim_amount (H4):** asked if H2=1 and H3≠7. Graded 1 a lot / 3 somewhat / 5 just a little / 7 not at all.
- Suggested ordinal combination: no condition < not at all/just a little < somewhat < a lot < can do nothing.

### ADL difficulty: adl_*_diff (H9a-g; SP H33a-g in 1999-2005)
- **Wording.** Stable in every wave: "by yourself and without special equipment. Because of a health or physical problem, do you have any difficulty …". The seven items are:
  - bathing or showering
  - dressing
  - eating
  - getting in or out of bed or a chair
  - walking
  - getting outside
  - using the toilet, including getting to the toilet
- **Response format.** Yes/no only in every wave; the items are never graded.
- **Universe breaks:**
  - **1999 and 2001:** asked only of people aged 55+. Age checkpoints: ER15524/ER15632 (1999) and ER19689/ER19797 (2001). Under-55s are coded 0. Prevalence is therefore much higher (for example RP walking 21-23% versus 8-10% from 2003).
  - **2003-2011 and 2015-2023:** all RPs and SPs, with no filter.
  - **2013 only:** a screener applied. RPs or SPs who reported excellent health and no work limitation were skipped (coded 0) on bathing, dressing, eating, bed/chair, getting outside and toilet. The same skip applied to the IADL items meals, shopping, phone and light housework.
    - Walking (H9e), money (H11e) and heavy housework (H11j) were still asked of everyone. This was checked in the data: 1,561 RPs were screened, and 13 of them reported walking difficulty.
    - Recoding the screened cases to "No" is an assumption, but a small one.
- **UKHLS mapping.**
  - Bathing, dressing, eating and toilet approximate LIM_SELF (personal care). PSID has no continence item.
  - Walking, bed/chair and getting outside approximate LIM_FL (mobility).

### ADL help: adl_*_help
- "Does someone usually help you with that activity?" Asked only if the matching difficulty item = 1 (and age 55+ in 1999-2001).
- Variables by wave:
  - 1999-2003: H10a-g (SP H34a-g)
  - 2005: H10a1-g1 (SP H34a1-g1)
  - 2007+: H10a1-g1 for both persons
- From 2005 a parallel item asks whether special equipment is needed (H10x2/H34x2, e.g. ER27061 in 2005). It is not reported here. It is useful for grading severity: difficulty only, with equipment, or with personal help.

### IADL difficulty and health follow-up: iadl_*_diff / iadl_*_health
- Asked 2003+ only, of all RPs and SPs. Variables: H11a/c/e/g/j/l (SP H35 in 2003-05). Codes: 1 Yes, 5 No, 7 Doesn't do.
- The follow-up H11b/d/f/h/k/m ("Is this because of a health or physical problem?") is asked **only of people who answered 7 (doesn't do)**. A health-related IADL limitation is therefore: diff=1 OR (diff=7 AND health=1).
- **Changes across waves:**
  - From 2011 the intro adds "and without special equipment".
  - The shopping item changed from "personal toilet items or medicines" (2003-09) to "personal items or medicines" (2011+).
  - The number of RPs answering "doesn't do" roughly halves from 2009 to 2011 (meals 234→122, heavy housework 626→413). This is a series break.
  - The 2013 screener applies as described above; money and heavy housework were not screened.
  - In 2003, code 0 for a present RP (36 cases, "wild code") or SP (16 cases) means NA.
  - In 2021-2023 the share of follow-up answers saying "yes, because of health" falls (meals 0.40→0.24-0.28, shopping 0.69→0.37-0.45). I found no wording change that explains this.
- **UKHLS.** There is no direct target. Heavy housework and shopping approximate LIM_FL (lifting, carrying, mobility). These items could also serve as extra deficits in the Rockwood frailty index.

### mob_walk_climb / mob_bend_lift
- Asked only in 2017, 2019, 2021 and 2023 (H8a3, H8a4), of all RPs and SPs, yes/no.
- **mob_walk_climb:** "trouble walking several blocks or climbing a few flights of stairs because of health". This is close to UKHLS SF-12 PF (stairs), but it is binary and merges walking with stairs.
- **mob_bend_lift:** "trouble bending, lifting, or stooping". This approximates LIM_FL (lifting/carrying).
- No earlier equivalents exist. The nearest substitute is ADL walking.

### hosp_any / hosp_nights / hosp_weeks
- **1999 and 2001:** not asked. The only related information is family-level hospital and nursing-home expenditure: H64 ER15781 (1997-98) and ER16515D3 (1998); ER19842 and ER20456D3 (2000). This is not a substitute for person-level use.
- **hosp_any (2003+):** H8 (SP H32 in 2003-05): "a patient in a hospital overnight or longer at any time during <previous calendar year>". The 2007 and 2009 labels wrongly say "2004"; the question text says 2006/2008.
- **Nights:** H8a "How many nights … altogether?" is recorded either in days (hosp_nights, 1-365) or in weeks (hosp_weeks, 1-52), in separate variables.
  - The two are mutually exclusive except for 4 RP cases that have both (weeks plus extra days).
  - Total nights = days + 7 × weeks. DK is coded jointly (998 and 98).
  - Values are not banded. The maximum observed is 365 (2009). The mean is about 6-8 nights for RPs and 4-5 for SPs.
- **UKHLS.** The target uses any stay and nights in the past 12 months. PSID refers to the previous calendar year, so it is lagged relative to the interview.
- **2021 (calendar 2020).** Admissions dip (RP 7.2% versus 9-10% in other waves). The COVID-only items HCOVID10/11 (2021, 2023) are not a substitute.

### weight / height (BMI)
- **Weight in pounds, every wave:** H22 (RP), H46 (SP 1999-2005), H22 (SP 2007+).
  - Range 70-500 in 1999, 65-500 in 2001, 50-500 in 2003.
  - From 2005: bottom-coded at 50 and top-coded at 400.
  - From 2011 people could answer in kilos instead (weight_kg, fewer than 60 per person-wave). In that case the pounds variable is 0.
- **Height, every wave:** feet (H23/H47, range 4-7 to 2003, 2-7 from 2005) plus inches.
  - RP inches are 0-11 up to 2009.
  - SP inches (all waves) and RP inches (2011+) use 1-11, and 0 means either zero inches or Inap. Use feet (or metres) to tell them apart.
  - From 2011 people could answer in metres instead (height_m). DK/NA codes are 9.98/9.99 in 2011 (and 2013 SP) and 8.00/9.00 later.
- All values are self- or proxy-reported. BMI = 703 × lbs / in², or kg / m².

### Not covered here
- I searched the codebook text of the 1999-2023 family files for bodily pain, pain interference, hearing, eyesight/vision and doctor-visit counts. The search found no RP/SP items, apart from "health information from doctor" (1999) and COVID items. PSID therefore offers no counterpart in these waves for UKHLS SF-12 BP, LIM_SENS, or GP and outpatient counts.


## Chronic conditions


Source: family files FAM1999ER to FAM2023ER, section H (health). The items are asked about the Head/RP and the Wife/"Wife"/Spouse-Partner (SP) only. None of these diagnosed-condition items exist for other family-unit members, and none are in the individual file (IND2023ER). From 2017 the individual file does carry the AD8 cognitive-change items for every person aged 65+ (H5N2-H5N9 for RP/SP, H50b-H50i for other members). Every variable in `conditions.csv` was checked against its codebook entry (question text, codes, Inap line) and against the parquet data (it exists, the distribution is plausible, the onset/limits universe equals ever=1). The CSV has 1,404 rows: 54 concepts x 13 waves x {RP, SP}. Where an item does not exist, the variable is left empty.

### Structure common to all conditions

| period | condition list (question letter) | onset follow-up | other follow-ups |
|---|---|---|---|
| 1999-2003 | H5/H29 a-l: A stroke, B high blood pressure, C diabetes, D cancer, E lung disease, F heart attack, G heart disease, H emotional/psychiatric, I arthritis, J asthma, K memory loss, L learning disorder | H6/H30 "How long have you had this condition?": a DURATION split into four variables (# DAYS, # MOS, # WKS, # YRS) | H7/H31 limits |
| 2005 | list reordered, H5/H29 a-m: A stroke, B heart attack, C heart disease, D hypertension, E asthma, F lung, G diabetes, H arthritis, I memory, J learning, K cancer, L psych, **M other serious chronic condition (new)** | H6x1 AGE at first diagnosis (stroke and heart attack: age at the first event); other condition: H6M2 (wife: H30M1) | new: second stroke/heart attack, "got worse", cancer treatment status and type (2 mentions), psych diagnosis (3 mentions) |
| 2007-2009 | same as 2005; wife now uses H5-H7 numbering in her own section | same | H6M1 type of other condition: one coded mention, same code list through 2023 (2007: ER38263A1/ER39360A1; 2009: ER44236A1/ER45333A1) |
| 2011-2023 | stem becomes "Has a doctor **or other health professional EVER** told (you/HEAD) that (you/he/she) had..." (it was "Has a doctor ever told you..."); the other-condition item had "or other health professional" from 2005 | same; 2017+ adds 995 = "<1 year old" (volunteered) | "WTR EDITED" flags on the ever items and on the cancer/psych type mentions; H7x2 taking medication; "got worse" items dropped (second stroke/heart attack kept); 2017+ adds the H5N cognitive-change items for people aged 65+ |

- **Universe of the "ever" items:** every Head/RP (no Inap code) and every wife/spouse present in the family unit (0 = Inap otherwise), in every wave. Nothing is preloaded and nobody is skipped because of an earlier report. The panel data confirm this: among RPs who said "yes" in 2017, the share saying "no" in 2019 was 13% (hypertension, diabetes), 18% (heart attack), about 22-25% (stroke, cancer, asthma, arthritis, heart disease), 35% (psych), 41-44% (memory, learning, lung) and 53% (other chronic). 1999-2003 shows similar rates.
- **Universe of the follow-ups:** onset, limits and type are asked of everyone with ever=1 in that wave, including people diagnosed long ago and people already reported in earlier waves. Codes 8/9 on the ever item skip to Inap.
- **SP codebook glitch, 2009:** the Inap text of the 2009 wife "ever" items contains stray childhood-health wording ("Respondent is not Wife..."). The Inap count (4,025) equals the no-wife count (ER42019=0), so the universe is all wives.
- **Data quirks:**
  - RP heart attack has 0 = "Wild code" in 2005 (14 cases) and 2007 (9 cases); 1999 has 1 wild code on every RP item.
  - 2021 has more DK on the ever items: about 55-60 RP cases (0.6%) on every item, against 12 or fewer (about 0.1% or less) in other waves. 2023 has more NA/refused (about 0.8%, against about 0.3% in 2017-2019). A few 2021 cases with ever=5 have onset 999 or limits 9.

### How close this is to the UKHLS target (COND count and frailty deficits)

- UKHLS records a condition once diagnosed and never removes it. PSID re-asks "ever" in every wave, and many "yes" answers revert to "no".
- To build a UKHLS-like COND, either cumulate (yes in any earlier wave means yes from then on) or use the current-wave report and accept the noise. This is the biggest comparability issue.
- **Physical conditions available in all 13 waves (9):**
  - stroke, heart attack, heart disease, hypertension, asthma, lung disease, diabetes, arthritis, cancer.
  - PSID lumps "coronary heart disease, angina, congestive heart failure" into one item, and "bronchitis or emphysema" into one. UKHLS asks these separately, so a PSID count will be lower for the same health.
  - Other chronic (2005+) is a catch-all. Its single open-ended type, H6M1 (2007+), can recover UKHLS-style conditions such as thyroid (code 9), seizures/epilepsy (1) and kidney (3).
  - Memory loss is cognitive. Learning disorder is developmental and should not enter a physical count.
  - Psych ("any emotional, nervous, or psychiatric problems") is broader than UKHLS "clinical depression". Use the psych type mentions (code 1 = Depression, 2005+ only, all three mentions) for the frailty-index depression deficit.
- **Limits:** a graded 4-point scale per condition. UKHLS has no equivalent, but it could weight the frailty deficits.
- **Onset:**
  - 2005+: age at first diagnosis or first event.
  - 1999-2003: duration only; age at onset ≈ current age − duration.

### Per concept

- **stroke** (H5A/H29A in all periods): "a stroke?" Wording is unchanged. Onset from 2005 is age at the *first stroke* (an event). Prevalence is stable: RP 2.8-3.5%, SP 1.7-2.3%.
- **heart_attack** (H5F, then H5B from 2005): "a heart attack?" Onset from 2005 is age at the first heart attack. RP 3.5-4.3%, SP 1.0-1.6%. Wild codes in 2005 and 2007 (RP).
- **heart_disease** (H5G, then H5C): "coronary heart disease, angina, congestive heart failure?"
  - The wording is identical in every wave. There is never a separate "heart disease" or "other heart disease" option.
  - **Unexplained level break:** RP is 6.4-6.8% (1999-2003) and 7.7% (2005), then about 5.0% from 2007 to 2023. SP falls from 4.8% to about 2.5%. No wording change is documented, so treat 2005→2007 as a possible series break.
  - The onset question reads "first diagnosed with (this condition/any of these conditions)".
- **hypertension** (H5B, then H5D): "high blood pressure or hypertension?" Rises steadily (RP 22% → 33%).
- **asthma** (H5J, then H5E): "asthma?" Steps up between 2003 and 2005 (RP 7.7 → 9.6%; SP 8.5 → 9.5%), when the item moved up the reordered list, then trends up to about 13% (RP) and 12% (SP).
- **lung_disease** (H5E, then H5F): "chronic lung disease such as bronchitis or emphysema?" Wording is unchanged. UKHLS separates bronchitis, emphysema and COPD.
- **diabetes** (H5C, then H5G): "diabetes or high blood sugar?" Wording is unchanged. Rises steadily.
- **arthritis** (H5I, then H5H): "arthritis or rheumatism?" Wording is unchanged. About 25% of "yes" answers revert in the next wave.
- **memory_loss** (H5K, then H5I): "permanent loss of memory or mental ability?"
  - The wording is unchanged. The label reads "MENTAL LOSS" in 1999-2003 and for the wife, and "MEMORY LOSS" for the head from 2005.
  - Alternative from 2017: H5N2-H5N9, eight AD8-type "change in thinking/memory" items for RP/SP aged 65+. The individual file repeats them for every person aged 65+, with H50b-H50i for other members.
- **learning_disorder** (H5L, then H5J): "a learning disorder?" Wording is unchanged. Not relevant to a physical condition count.
- **cancer** (H5D, then H5K):
  - 1999-2003: "cancer or a malignant tumor, **excluding skin cancer**?"
  - From 2005 the head wording drops the exclusion. The 2005-2009 wife codebook still prints it, but wife cancer-type answers include skin cancer and wife prevalence jumps like the head's (RP 4.1 → 4.9%, SP 4.2 → 5.4%). So treat 2005+ as including skin cancer for both persons.
  - 2011+ has no exclusion for either person.
  - To harmonise with 1999-2003, drop cases whose only type is non-melanoma skin cancer (2005-2011 code 7; 2013+ codes 10 and 11). Mention 2 must be checked too.
  - From 2005, H6K3 (in treatment / remission / cured) can flag active cancer.
- **psych** (H5H, then H5L): "any emotional, nervous, or psychiatric problems?"
  - The wording is the same in every wave. Only the label changes, from "EMOTIONAL PROB" to "PSYCH PROBLEM" in 2005.
  - Strong upward trend: RP 5.3% (1999) to 12.7% (2023); SP to 14.7%.
- **other_chronic** (H5M/H29M, 2005+ only): "Is there any other serious, chronic condition that a doctor, or other health professional, ever told you that you have or have had?"
  - Not asked in 1999-2003.
  - Its open-ended type is H6M1 (2007+, one mention).
  - RP prevalence drifts from 15.2% (2005) to 13.1% (2019), then drops to 11.8% (2021) and 9.7% (2023). SP goes 16.1% → 14.7% → 12.3% → 9.2%. The wording is unchanged, so treat 2021-2023 as a possible mode or fieldwork break.
- **cancer_type_m1** (H6K4, mention 1 of 2; 2005+):
  - Codes 2005-2011: 1 breast, 2 colon, 3 lung, 4 lymphoma/leukemia, 5 melanoma, 6 prostate, 7 skin, 8 uterine, 9 ovarian, 10 cervical, 97 other.
  - **Recoded from 2013**, alphabetically: adds bladder and thyroid, and splits skin into melanoma, non-melanoma and DK type.
  - Mention-level flags ("H6K4 WTR EDITED (MEN 1/2)") exist for 2011-2023. They are not in the CSV, because they belong to the type item rather than to C_ever.
- **psych_type_m1** (H6L2, mention 1 of 3; 2005+):
  - 1 = Depression throughout.
  - 9 = PTSD in 2007-2009 and 2013+. It is not in the 2005 or 2011 lists, and the 2007 wife codebook leaves it out.
  - 10 ADHD, 11 ODD, 12 autism added in 2021.
  - Mention-level flags ("H6L2 WTR EDITED (MEN1-3)") exist for 2011-2023; they are not in the CSV.
- **C_ever_edited** ("H5x WTR EDITED"):
  - Exists in 2011 for H5A-H5L, and in 2013-2023 for H5A-H5J only (no cancer or psych flag after 2011). Never for H5M.
  - 1 means PSID staff set the ever item from the open-ended other-condition text H6M1. The C_ever variable already contains the edit.
  - 0-18 cases per item and wave, mostly lung disease and arthritis. Totals across all flags per wave: 26 (2011), 1 (2013), 37, 70 (2017), 45, 46, 34 (2023).

### Better or extra sources found

- H6M1, the other-condition type (2007+), for UKHLS conditions PSID does not ask by name.
- H7x2, taking medication for the condition (2011+), as a marker of active disease.
- Psych type mentions 1-3, to isolate depression.
- H5N2-H5N9 (2017+), for cognitive impairment.


## K6, respondent, spending and insurance


Person codes: RESP = the single family respondent; FU = family-level; RP/SP = RP/Head or SP/Wife record (the insurance rows point to variables in the individual file IND2023ER). Each finding was checked against the codebook text and Inap line, and against value counts in the parquet data.

### K6 distress: k6_sad, k6_nervous, k6_restless, k6_hopeless, k6_effort, k6_worthless, k6_scale
- **Waves.** Asked in 2001, 2003 and 2007-2023 (11 waves). It is not asked in 1999 or 2005: neither has an H59 block. The only items in those years are "ever told by a doctor: emotional, nervous or psychiatric problem", which are diagnosis items, not distress measures:
  - 1999: Head ER15494, Wife ER15602
  - 2005: Head ER27045, Wife ER27168
- **Who answers.** Only the one family respondent answers the K6, and only if that respondent is the Head/RP or the Wife/SP ("Spouse/Partner" from 2015). The universe is WHO WAS RESPONDENT codes 1-3. The K6 is never asked of both partners.
  - Attach the answers to the RP when WHO WAS RESPONDENT = 1, and to the SP when it is 2 or 3. The other partner has no K6 in that wave.
  - In couple families the K6 belongs to the SP about 57% of the time and to the RP about 43% (2003 and 2019 checks). So each partner's K6 comes and goes across waves.
  - 2001: respondents who are not Head/Wife (codes 4 and 7, about 126 cases) are coded 9 instead of 0.
- **Wording and codes.** Identical in 2001-2023: "In the past 30 days, about how often did you feel ...". Codes are 1=all of the time ... 5=none of the time, 8=DK, 9=NA, 0=Inap.
- **k6_scale.** PSID's sum score, 0-24 (each item rescored from 4 down to 0). It is 99 if any item is DK/NA or the respondent is not RP/SP. Recomputing it from the items matches at least 99.5% of cases in every wave. 2023 adds ER84929, a cut-point flag: 1 = score below 13, 2 = 13 or more.
- **Data.** The mean score is about 3.3-3.5 in 2001-2019, 3.96 in 2021 and 4.25 in 2023.
  - Part of this rise is the switch to web interviewing (40% of interviews in 2021, 73% in 2023). In 2021 the web mean was 4.26 against 3.76 by phone.
  - 2021 also has 54 RP/SP respondents with scattered item-level 0s; treat these as missing.
- **Closeness to UKHLS.** The K6 measures non-specific psychological distress over the past 30 days, the same construct as the GHQ-12 but on a different scale. The SF-12 mental items do not exist in PSID.
  - The key difference is coverage. UKHLS has a self-completion score for every adult in every wave; PSID has one score per family.

### k6_interfere (H59J)
- **Wording.** Unchanged in 2001-2023: "How much do these feelings usually interfere with your life or activities?" Codes are 1=a lot, 2=some, 3=a little, 5=not at all.
- **Universe (respondent only).** The data show it is asked only if at least one K6 item is "some of the time" or more often (codes 1-3).
  - It is Inap (0) when every item is "a little of the time" or "none of the time", which is about 50% of families.
  - The codebook text for 2011 onward says it is skipped for "no depressive feelings", but the data skip at "a little of the time".
- **Related items.** The follow-ups H59G-I (more or less often than usual) and H59K-L (days unable to work or with reduced work) exist in 2001-2011 and were dropped in 2013. 2003 also has two CIDI screener items, H59M and H59N.

### resp_who
- **Variable.** WHO WAS RESPONDENT exists in every wave and its codes are stable: 1=Head/RP, 2=Wife/Spouse, 3="Wife" cohabitor (to 2013) or Partner (2015 on), 4=other family member, 7=proxy from outside the family, 9=NA.
- **No health-section proxy flag.** Every RP and SP health item in the family file is answered by this one respondent. So:
  - RP health items are self-reports only when the code is 1. Otherwise the spouse or a proxy answered.
  - SP health items are self-reports only when the code is 2 or 3.
- From 2017, the section checkpoints M1CKPT and FOOD1CKPT also flag respondents who are other family members or proxies. This repeats codes 4 and 7.

### Health spending: spend_hosp, spend_doctor, spend_rx (raw and gen), spend_total_gen
- **Level.** Always the whole family's out-of-pocket spending. It is never broken down by person.
- **Reference period (the main break):**
  - 1999-2011: "[y-2] and [y-1] combined", two calendar years, for all items.
  - 2013: the hospital item covers 2012 only. The doctor and Rx codebook text still says "2011 and 2012 combined". But PSID's generated variables document "in 2012" and do not halve the amounts. Within the same families, the levels behave like one-year amounts: the median 2011-to-2013 ratio is about 0.7 (0.6 for the documented one-year hospital item), and the 2013-to-2015 ratio is about 1.0. These rows are marked medium confidence.
  - 2015-2023: previous calendar year only.
- **Raw codes.** 0 means nothing; for hospital from 2013 on, 0 is labelled "Inap: no OOP expenses". Brackets follow up DK and refusals. Missing codes are 999,998 (DK) and 999,999 (NA). For doctor and Rx in 1999-2003 they are 9,999,998 and 9,999,999.
- **Generated variables.** These are annual amounts from PSID's consumption series:
  - 1999-2011: raw amount / 2. 2013 on: the raw amount.
  - DK/NA are imputed by regression, so a few values are negative. Accuracy flags are named in each row.
  - spend_total_gen ("HEALTH CARE EXPENDITURE") equals hospital + doctor + Rx + insurance premiums exactly, in every wave. It includes premiums.
- **Web mode.** In 2021 and 2023 about 9-11% of web respondents give NA on each spending item, against about 1.5% by phone; the generated variables impute these values. Web respondents also report "nothing" less often.
- **Closeness to UKHLS: weak.** The UK cost proxy is priced use of services per person.
  - Out-of-pocket dollars depend mainly on insurance.
  - They are family-level.
  - They lump together doctor, outpatient surgery and dental, and hospital with nursing home.
- **Possible alternative.** H82 "total cost of all medical care" counts out-of-pocket plus insurer-paid costs. It is family-level and covers two years; 36-47% answer DK in 1999-2009 and 17% in 2011. It exists only in 1999-2011 and was dropped in 2013:
  - 1999 ER15799, 2001 ER19860, 2003 ER23297, 2005 ER27257
  - 2007 ER40432, 2009 ER46404, 2011 ER51765

### Insurance: ins_any_fu, ins_any_uninsured_fu, ins_premiums, ins_premiums_gen
- **ins_any_fu.**
  - 1999-2011: H60, "was anyone covered at any time in y-2 or y-1".
  - 2013-2023: H61D2, "does anyone currently have coverage". This is a different reference period, so it is marked medium confidence.
  - "Anyone uninsured": H61K (anyone without insurance at any time in y-2 or y-1) exists from 2013 only. It is reported as the extra concept ins_any_uninsured_fu. Before 2013 this can only be derived from the per-person months covered.
- **ins_premiums.**
  - 1999-2011: H63, a two-year total including payroll deductions. The wording drifts: 2005-2009 ask "how much were the premiums"; 2011 adds "only amounts you paid".
  - 2013 on: H61J, the current premium as an amount plus a time-unit variable (PER). The PER codes change between waves: 2013 uses 4=quarterly, 5=monthly, 6=year; 2015 on uses 3=week, 4=two weeks, 5=month, 6=year.
  - From 2013 the question is asked only if someone has employer-provided or privately bought coverage.
  - ins_premiums_gen (an extra concept) is PSID's annual figure, imputed: H63 / 2 up to 2011, amount × periods per year from 2013.

### ins_type_rp, ins_type_sp, ins_months_uninsured_rp/_sp
- **Not in any family file 1999-2023.** These items are asked per person in the family interview but stored in the individual file, IND2023ER.
  - Select the RP's record with sequence = 1 and relation = 10.
  - Select the SP's record with sequence 1-20 and relation 20 or 22.
- **Coverage type (first mention).**
  - 1999-2011: H61, the type held during y-2 to y-1.
  - 2011 also has H61E, current type (ER34129). It matches the later series and could be used instead.
  - 2013 on: H61E, current coverage only.
  - From 2013, the military, TRICARE and VA codes (6, 7, 11) are merged into code 12.
- **Months.**
  - 1999-2011 record months COVERED: H62/H62A, renamed H61B/H61C from 2005. Months uninsured = 12 − covered.
  - 2013 on record months UNINSURED (H61M/H61N), asked only when H61K = 1.
  - Each row gives the y-1 variable. The y-2 variables are:
    - 1999 ER33522, 2001 ER33622, 2003 ER33722, 2005 ER33824
    - 2007 ER33924, 2009 ER34027, 2011 ER34126, 2013 ER34233
    - 2015 ER34383, 2017 ER34592, 2019 ER34800, 2021 ER35002, 2023 ER35202
- Insurance has no UKHLS counterpart. It matters only as a control, because out-of-pocket spending depends on coverage.

### visits_doctor, visits_outpatient, nursing_home_nights: none exist
- No PSID family or individual file for 1999-2023 has a count of doctor/GP visits, outpatient visits or nursing-home nights.
  - Individual nursing-home stays and weeks exist only in 1991-92.
  - The family variable TYPE INSTITUTION only flags families living in a hospital or nursing home.
  - The only related information is the family's out-of-pocket dollars.
- **Nearest per-person item: inpatient stays.** H8 (RP) and H32/H8 (SP) ask whether the person was a patient in hospital overnight in year y-1. A follow-up records nights, or weeks in a separate variable. It exists for RP and SP in 2003-2023 only (not 1999 or 2001). The variables are listed as any stay / nights / weeks:

  | Year | RP | SP |
  |---|---|---|
  | 2003 | ER23089 / 90 / 91 | ER23216 / 17 / 18 |
  | 2005 | ER27056 / 57 / 58 | ER27179 / 80 / 81 |
  | 2007 | ER38267 / 68 / 69 | ER39364 / 65 / 66 |
  | 2009 | ER44240 / 41 / 42 | ER45337 / 38 / 39 |
  | 2011 | ER49578 / 79 / 80 | ER50696 / 97 / 98 |
  | 2013 | ER55326 / 27 / 28 | ER56442 / 43 / 44 |
  | 2015 | ER62448 / 49 / 50 | ER63564 / 65 / 66 |
  | 2017 | ER68511 / 12 / 13 | ER69638 / 39 / 40 |
  | 2019 | ER74519 / 20 / 21 | ER75646 / 47 / 48 |
  | 2021 | ER80667 / 68 / 69 | ER80865 / 66 / 67 |
  | 2023 | ER84639 / 40 / 41 | ER84840 / 41 / 42 |

  - The 2007 and 2009 labels say "2004", but the question text refers to 2006 and 2008.
  - 2021 and 2023 also have COVID hospital nights: RP ER80570/ER84539, SP ER80768/ER84740.
- **Consequence for the cost proxy.** In PSID a UK-style proxy can price only inpatient nights, for RP and SP, from 2003. The GP and outpatient parts have to be dropped, imputed from another source (for example MEPS), or replaced by family out-of-pocket spending.


## Individual file: identifiers, mortality, weights


Source: IND2023ER Release 2 (85,536 persons, one row each). Every variable in `individual.csv` was checked three ways: its label, its full codebook entry (question, codes, Inap. universe) and its distribution in `IND2023ER.parquet`. Variable names follow a fixed pattern in each wave block (ER33501 for 1999, ER33601 for 2001, ..., ER35101 for 2023). In 2015 there is a shift of one, because the new ER34304 (same-sex partner flag) pushes age to ER34305 and birth year to ER34307. Tables are built by `/tmp/claude-0/-home-claude/8a6409eb-0028-56c6-bac7-614479d4e93c/scratchpad/work/build_individual.py`.

### Identifiers and structure

- **id_1968 / id_person.** The person ID is `ER30001*1000 + ER30002`. It is unique for all 85,536 records. ER30002 values 1-169 mark people who are sample by origin or birth. Values of 170 and above mark movers-in or nonsample people; ER32006 gives their sample status.
- **fam_id.** This is the per-wave "YYYY INTERVIEW NUMBER". It merges 1:1 with the FAM file's `FAMILY INTERVIEW (ID) NUMBER`:
  - 1999 ER13002, 2001 ER17002, 2003 ER21002, 2005 ER25002
  - 2007 ER36002, 2009 ER42002, 2011 ER47302, 2013 ER53002
  - 2015 ER60002, 2017 ER66002, 2019 ER72002, 2021 ER78002, 2023 ER82002

  All seq 1-20 IDs match. **Trap:** the interview number is also filled in for people who were not yet born or had not yet moved in at that wave (seq=0, WHY NONRESPONSE=97, for example 20,641 people in 1999). So `fam_id>0` does not mean the person was present that wave. Always select on the sequence number.
- **seq_num.** The codes are the same in all 13 waves:
  - 1-20: in the family unit (FU) at this interview. 01 is the Head/RP and 02 is the Spouse/Partner when present.
  - 51-59: in an institution.
  - 71-80: moved out between interviews and was not interviewed.
  - 81-89: alive at the previous interview but died before this one.
  - 0: Inap.

  Check: RP = seq 01 & rel 10 gives exactly one person per FAM-file family in every wave. SP = seq 02 & rel 20/22.
- **rel_to_rp.**
  - 1999-2013: Head/"Wife" coding. 20 = legal wife; 22 = female cohabitor of 12+ months; 88 = first-year cohabitor; 90 = legal husband of Head; 98 = nonrelative, which includes same-sex partners.
  - 2015: the wording changes to Spouse/Partner, and 90 becomes "uncooperative legal spouse". Same-sex partners are still coded 98, with a new flag (ER34304, 60 cases).
  - 2017 onward: Reference Person; 22 = partner of any sex; new code 92 (uncooperative partner). 65-79 same-sex couples per wave are now coded 20/22.
  - In opposite-sex couples the RP is still male (only 1 exception, in 2023).
  - For seq>50 the relationship refers to the previous wave's Head/RP, so a dead Head keeps code 10.
- **age.** Age at the interview. 1 = newborn up to the 2nd birthday; 999 = NA. **For seq 81-89 records it is age at death** (death year minus birth year), not age at interview.
- **birth_year / birth_month.** There is no time-invariant birth year or month in IND2023ER. Both are re-recorded every wave (ER33505/ER33506 ... ER35105/ER35106). I give a time-invariant row listing all 13 per-wave variables, plus per-wave rows. Use the latest valid value. Across waves, 239 people have conflicting birth years and 85 have conflicting birth months.
- **respondent** (extra). RESPONDENT? per wave: 1 = respondent, 5 = not. Only about one person per family is the respondent, so all other-member (OFUM) data and the non-respondent partner's data are proxy reports. This matches the UKHLS "self vs proxy" item.
- **sample_type.** Derived from ranges of ER30001, quoting the codebook:
  - 1-2930: "1968 SRC cross-section sample"
  - 5001-6872: "1968 Census [SEO] sample"
  - 3001-3511: "Immigrant sample added in 1997 and 1999" (3442-3511 = 1999 recontacts)
  - 4001-4851: "Immigrant sample added in 2017 and 2019". "Values of 4001-4462 and ER32052=2019 indicate the Multiplicity sample"; 4700-4851 = 2019 additions.
  - 7001-9308: "Latino sample added in 1990 and 1992"

  ER32052 (year the cohort began) is needed to split the 2017 and 2019 cohorts, and it matches the ranges exactly. The Latino sample has no data in 1999-2023 (dropped after 1995). Part of the SEO sample was dropped in 1997 (WHY NONRESPONSE/TYPE OF RECORD 80/40). Design variables: ER31996 (stratum) and ER31997 (cluster).

### Mortality

- **death_year: ER32050**, "More Precise Year of Death from the 1968-2023 PSID Death File". There is one value per person.
  - 0 = Inap, meaning "not deceased". It really means "not known dead", not "confirmed alive".
  - 1967-2023 = exact year (7,816 people).
  - Range code = 293 people. The digits are the last two digits of the first possible year followed by the last two digits of the last possible year. Examples: 9901 = 1999-2001, 709 = 2007-2009, 1016 = 2010-2016, 1921 = 2019-2021, 2123 = 2021-2023.
  - 9999 = NA year (12 people).

  In total 8,121 people have a death record. Any value outside 1967-2023 other than 9999 is a range, so a rule like "1900-2100 = a year" misreads 1921 and 2123. 3,610 exact death years fall in 1999-2023, and 3,165 decedents were in a responding FU at some point in 1999-2023.
- **death_month.** Not in the public file. The only substitute: for deaths reported by a still-responding family (WHETHER MOVED IN/OUT = 7, i.e. seq 81-89), MONTH and YEAR MOVED IN/OUT give the month and year of the "move out", which is the death. The month is valid for 92-99% of these cases, and the year equals ER32050 for 96.5% of them. Deaths learnt after attrition have no month.
- **death_source.** There is no source, editing or imputation flag. ER32004 is the closest: code 2 = died while connected to a responding main family (4,387); code 3 = died after leaving (3,734). It agrees exactly with ER32050>0. The 2007 attritor project adds a vital-status search for attriters: ER33946 status and ER33947 information source (451 found dead).
- **Per-wave death flags.** There are three, nested:
  1. WHY NONRESPONSE = 41 (`death_flag`). This counts every death the study learnt of that wave, including people who had already attrited. It flags 212-407 people per wave, and 897 in 2007 because of the attritor project.
  2. seq 81-89 = WHETHER MOVED IN/OUT 7. This counts only deaths in still-responding families, 106-154 per wave.
  3. ER32050 for the year itself.

  **Key point:** among people in a responding FU at wave t who died before t+2, only 48% (1,378 of 2,862, pooled 1999-2023) get seq 81-89. The rest have seq 0, mostly because they lived alone and the family itself became nonresponse. A mortality hazard based on seq 81-89 alone would understate deaths by about half. Use WHY NONRESPONSE=41 or ER32050.
- **Attrition summaries:**
  - ER32007: most recent nonresponse year; 2023 = currently nonresponse, which includes all deceased.
  - ER32008: first nonresponse year.
  - ER32005: ever out of the study for a year or more.
  - ER32001: always in a responding FU.
  - ER32002: ever institutionalised.
  - ER32006: sample status. Nonsample people are not followed and have zero longitudinal weight.

  No "last year in panel" variable exists; derive it as the last wave with seq 1-20.
- Comparison with UKHLS ("deaths learnt by the survey"): seq 81-89 and ER32004=2 are the closest analogue. ER32050 also includes deaths learnt after attrition through PSID tracking or the death file.

### Per-wave person characteristics

- **emp_status.** Same codes 1-9 in every wave: working, temporarily laid off, unemployed/looking, retired, permanently disabled, keeping house, student, other, NA. For RP/SP it is the prioritised version of the first 3 mentions of FAM BC1/DE1; for OFUMs, the first 2 mentions of G75; priority order 2,1,3,4,5,7,6,8,9. Universe: aged 16+; deceased and people institutionalised in both waves are excluded. From 2003 it also excludes people who moved out before January of t-1. Only the code-6 wording changes: "HouseWife", then "HouseSpouse" (2015), then "Keeping house". This maps well onto UKHLS jbstat, except that it has no self-employed split and no separate long-term sick code (5 = permanently disabled).
- **educ_years.** Highest grade completed, 1-17. Before 2013, 98 = DK and 99 = NA; from 2013, 99 covers DK/NA/refused. This is the IND version: GED without college is set to 12, whereas the FAM file keeps the grade. RP/SP values were not re-asked each wave before 2013 (re-asked in 1985 and 2009). From 2013 there is an education update. In 2013 only seq 1-20 have values, and coverage of movers-out varies by wave. The UKHLS target is the highest qualification, so years must be mapped to levels (12 = HS/GED, 16 = BA, 17 = postgraduate). Degree-type items exist from 2013 (IND "TYPE OF HIGHEST DEGREE") and in the FAM file for RP/SP.
- **ins_any / ins_type / ins_months_py** (US-specific; no UKHLS counterpart).
  - 1999-2009: no person-level yes/no item. Coverage at any time in the two previous calendar years comes from the H61 type, mention 1, where >0 means covered. Code 0 mixes "not covered" with Inap. Mentions 2-4 and months covered in t-2 and t-1 are also available.
  - 2011: both systems. H61D "covered now" (ER34128) and the current type (ER34129-31) are used here; the prior-2-year items (ER34121-27) are the alternative.
  - 2013-2023: "covered now" (H61D3, 1/5) and the current type. Codes 6, 7 and 11 are merged into 12. Months **uninsured** (not covered) are recorded for t-2 and t-1.
  - From 2017, movers-out are excluded.
  - So "any coverage" changes from "ever in the last 2 years" to "now" in 2011/2013, and months covered becomes months uninsured.
  - The share not covered among seq 1-20 jumps from 12% (2019) to 16% (2021) and 20% (2023); among children it goes 6%, 13%, 19%. Check this before using it.
- **nonresponse_reason / record_type / follow_status / moved_in_out** (moved_in_out and record_type are extra rows). WHY NONRESPONSE has a stable code list. Codes 15/25 ("asked to be removed") were added in 2011. Code 56 exists only to 2003. Code 5 was redefined in 2023 ("FU in a foreign country 2+ waves"). Code 97 (not yet born/moved in) is absent in 2023. TYPE OF IND RECORD codes 6/8 combine movers-out with deaths.

### Weights

- **wt_long** is "CORE/IMM INDIVIDUAL LONGITUDINAL WT" in every wave.
  - 1999-2015: core plus the 1997/99 Immigrant samples. 1999-2005 values were recalculated in Release 3 (revised 1993-2005 weights); 2007 was re-estimated.
  - 2017: **zero for the 2017 Immigrant refresher**.
  - From 2019: includes the 2017/19 refresher.
  - Values are nonzero only for sample members (ER32006 1-4), including those coded 51-59, 71-80 and 81-89 in that wave. They are on a relative scale (mean about 25-31; maximum rises from 168 to 283).
- **wt_cross** is "CORE/IMM INDIVIDUAL CROSS-SECTION WT" in every wave. It is population-scaled: seq 1-20 sum to 260M (1999) and 320M (2023). It covers the 2017 refresher from 2017 and the 2019 additions from 2019. It is **nonzero for every record with seq>0, including movers-out and the deceased**, so restrict to seq 1-20 (plus 51-59 if institutions are included).


## Income, education, race, region and employment


General conventions (apply to all RP/SP concepts)
- Who is who: Head + Wife/"Wife" (cohabiting partner) through 2013. In 2015 it is Head + Spouse/Partner, and from 2017 Reference Person (RP) + Spouse/Partner. Through 2015 the Head is always the male of a male-female couple. From 2017 same-sex couples appear as RP+SP.
- SP variables are 0/Inap when there is no wife or spouse in the FU. In many of them 0 also carries a substantive meaning: zero hours or income, not Hispanic, or no grades. Use age of SP > 0 as the spouse-presence flag.
- Income, hours and wages always refer to the PREVIOUS calendar year (wave y -> year y-1). Generated (PSID-constructed) amounts have all missing data assigned, so there are no DK/NA codes. Raw items keep DK/NA codes.
- Race, Hispanic origin and education (sections K/L) are asked only when a person first becomes Head/Wife, or in splitoff/recontact families. Otherwise they are carried forward; the K1/L1 checkpoints are cited per wave in the CSV. All Heads/Wives were re-asked race and Hispanic origin in 2005 and education in 2009. From 2013 returning RP/SP report education updates.

### fam_id
- "<YEAR> FAMILY INTERVIEW (ID) NUMBER". It is wave-specific, not a longitudinal id. Link it to people through IND2023ER "<YEAR> INTERVIEW NUMBER" (ER33501, ER33601, ... ER34301, ER34501, ER34701, ER34901, ER35101).
- In 1999, 10001-10071 are the immigrant recontact sample.

### iw_month, iw_day, iw_year
- "MONTH/DAY/YEAR CURRENT IW". Stable across waves; a direct match for the UKHLS interview date.
- Fieldwork runs into the next January in some waves, so iw_year can differ from the wave: 2013 (14 cases), 2017 (74, Jan-Feb 2018, mostly the new immigrant refresher), 2019 (18), 2021 (75), 2023 (91).

### region
- Label is "REGION OF 1999 INTERVIEW" in 1999 and "CURRENT REGION" from 2001.
- Codes: 4 Census regions, plus 5=Alaska/Hawaii and 6=foreign country. 9=NA/DK appears in 2001 only (3 cases). 1999 has 2 wild codes (0).
- Much coarser than the 12 UK GORs. If finer geography is needed, build Census divisions from state.

### state
- "CURRENT STATE": FIPS 1-56, 0=US territory or foreign country, 99=DK/NA.
- Alternative: "PSID STATE OF RESIDENCE CODE" (ER13004 ... ER82003), the same information in PSID 1-51 coding. I chose FIPS because its label is literally "current state".

### fam_size, n_children
- "# IN FU" (1-20) and "# CHILDREN IN FU" (0-18). Children are persons aged 0-17 who are neither Head/RP nor Wife/SP, whatever their relationship. Definitions are stable.
- The UKHLS equivalence scale (modified OECD) needs children under 14. Take the ages of other members from the IND file if an exact replica is wanted. Otherwise use sqrt(fam_size) or OECD weights with the under-18 count.

### income_total
- "TOTAL FAMILY INCOME[-yyyy]" ("... LAST YEAR" in 2003). It covers the previous calendar year: taxable income of Head/Wife and other FU members (OFUMs), plus transfers, plus Social Security, with components imputed.
- Coding:
  - Can be negative (1-19 cases per wave).
  - Bottom code -999,999 in 2001-2009, -999,997 from 2011; top code 9,999,999 / 9,999,997. These codes are never reached.
  - Zero income grows from 39 families (1999) to 287 (2023). From 2013, 0 is labelled "Inap.: no family money income".
- Versus the UKHLS target (household net income, equivalised, ranked within wave):
  - PSID gives gross (pre-tax) annual income of the family unit (FU) for the previous year. UKHLS uses current household net income.
  - Ranking within wave removes inflation, and makes the gross/net difference less important.

### wt_fam
- "CORE/IMMIGRANT FAMILY WEIGHT" (longitudinal family weight). The wording varies: 1999 "1999 CORE/IMMIGRANT FAMILY WEIGHT", 2001 "... NUMBER 1", 2003 "CORE/IMM FAMILY LONGITUDINAL WEIGHT", 2005+ "<YEAR> CORE/IMMIGRANT FAM WEIGHT NUMBER 1".
- Zero weights:
  - 1999-2003: 146, 211 and 257 families respectively (all members have zero individual weight). The upper limit is 300 in 2001/2003.
  - 2017: 0 for the 452 new immigrant-refresher families (ER66009=4001-4462).
- From 2005 the family weight is the mean of the members' individual longitudinal weights.
- Cross-sectional family weights exist only in 1999, 2001, 2003, 2017 and 2019 (ER16519, ER20459, ER24180, ER71571, ER77632).
- UKHLS uses person weights. For person-level RP/SP analysis, the individual weights in the IND file are a closer match.

### age
- "AGE OF HEAD/REFERENCE PERSON" and "AGE OF WIFE/SPOUSE". Range 14-120, 999=NA/DK, and for SP 0=no wife/spouse.
- A direct match for UKHLS age at interview. Birth year (IND file) can be used to check consistency across waves.

### sex
- RP: "SEX OF HEAD/REFERENCE PERSON", 1/2, all waves.
- SP: "SEX OF SPOUSE" exists only from 2015, so 1999-2013 are left empty. Before 2015 the Wife/"Wife" is female by construction; verified that a female Head never has a wife present.
  - In 2015 all spouses are female.
  - From 2017 there are 18-25 male-male and 43-54 female-female couples per wave, and 1 female RP with a male spouse in 2023.
- Best consistent alternative for both persons: IND ER32000 (sex of individual).

### marital (RP)
- A3 "HEAD/REFERENCE PERSON MARITAL STATUS" = legal status, so cohabitors report e.g. never married.
- The 1999 code list has no NA code, and 2001 lists no DK/NA codes. Codes are otherwise stable.
- Alternative: "MARITAL STATUS-GENERATED" (ER16423 ... ER85782), where 1=married or permanently cohabiting.

### race_m1
- Labels: "L40/95 RACE OF HEAD 1" / "K34/87 RACE OF WIFE 1" (1999-2003); L40 / K34 (2005) / K40 (2007+) "RACE OF ...-MENTION 1".
- Definition change in 2005:
  - 1999-2003 codes: 1 White, 2 Black, 3 Am. Indian/Aleut/Eskimo, 4 Asian/PI, 5 Latino origin, 6 other colour, 7 other.
  - 2005+ OMB codes: 1 White, 2 Black, 3 AIAN, 4 Asian, 5 Native Hawaiian/PI, 7 Other, with Hispanic origin asked separately just before.
  - The 2005 re-ask covered all Heads/Wives. For the same heads, 2003-2005 agreement is 0.907; most changes are old code 5 (Latino) becoming White (186) or Other (104).
- Carry-forward:
  - The 2007/2009 codebook says all Heads were asked, but the values for continuing Heads are identical to 2005.
  - Otherwise race is asked of new RP/SP only and carried forward.
- UKHLS "white vs other": use race_m1==1 (optionally also excluding Hispanic and multiple mentions) from 2005. A better option for all waves is to treat the 2005 answer as time-invariant and copy it to earlier waves through the IND link.

### hispanic
- NOT ASKED 1999-2003 (left empty).
- 2005: "L39A/K33A SPANISH DESCENT", asked of all Heads/Wives. 2007+: "L39/K39", asked of new RP/SP only (RP 2007/2009 is effectively carried forward, as for race).
- Codes: 1 Mexican, 2 Mexican American, 3 Chicano, 4 Puerto Rican, 5 Cuban, 7 other Hispanic; 0 = not Hispanic (SP: also no spouse). DK/NA are 8/9 to 2011 and a combined 9 from 2013.
- Near-substitutes for 1999-2003:
  - race_m1 code 5 "Latino origin";
  - "L41/K34A PRIMARY ETHNIC GROUP" code 4 "Nonspecific Hispanic identity" (weak);
  - best: back-fill from the 2005 answer.

### educ_years
- "COMPLETED ED-HD/WF" (from 2015 "-SP", from 2017 "-RP"). Values: 0-16 grades, 17 = some postgraduate (shown as 1-17 from 2013), 99=NA/DK. For SP, 0 = no wife or no grades.
- Collection history:
  - Asked when a person becomes Head/Wife, then carried forward.
  - Re-asked of all Heads/Wives in 2009. A Release 2 correction recalculated 2009 and brought it forward to 2011.
  - From 2013, the value is raised when an update reports more education.
- NA is about 5% before 2009 and about 1% after.
- A GED without college is coded as the last grade (the IND-file version codes 12).
- This is only a proxy for the UKHLS "highest qualification"; the degree items are closer.

### degree_hs, degree_college, degree_highest
- Head items are L44/L54/L55 in all waves. Wife items are K37/K47/K48 in 1999-2005 and renumbered K44/K54/K55 from 2007, with the same questions.
- Universe:
  - US education only: people educated only abroad, or with no education, are 0/Inap.
  - L54 is asked if the person attended college for at least 1 year.
  - L55 is asked if L54=Yes.
  - HS code 4 "college level only" is listed from 2003 but is never used.
- DK/NA coding varies by wave (8/9, then 9, then 8/9 in 2011, then 9 from 2013). For L55, 98/99 become a single 99 from 2015.
- These items are carried forward and were re-asked in 2009: the RP college-degree Yes share goes from 24% (1999) to 31% (2009) and 34% (2023).
- From 2013, degrees earned since the last interview are recorded in separate items:
  - L79/K79 (2013: ER57722/ER57612);
  - L78F1/K78F1 "HIGHEST DEGREE EARNED" (RP ER64895, ER70966, ER76984, ER81212, ER85189; SP ER64756, ER70828, ER76839, ER81085, ER85062);
  - "YEAR HIGHEST EDUCATION UPDATED".
  - Combine these with L55 to get the current highest degree.
- UKHLS mapping: none / HS or GED / some college / AA / BA / postgraduate (codes 3-6). This is a good match for the degree / other higher / school-level hierarchy.

### emp_status_m1, emp_status_m2, emp_status_m3
- Questions: B1/D1 (1999-2001), BC1/DE1 (2003+): "working now, looking for work, retired, keeping house, a student, or what?". This is current status at interview, with up to 3 mentions, and the same 8 categories in every wave.
- Coding changes:
  - DK/NA: 98 and 99 in 1999-2001, a single 99 from 2003.
  - Wild codes: RP 2003 (0, 22), RP 2005 (0), SP 2001 (9, 35), SP 2007 (32).
  - RP code 8 "other" is empty in 2013-2019.
  - 2021 and 2023 have many more temporary layoffs (RP 128/175) and DK/NA (RP 40/76).
- Versus UKHLS jbstat:
  - PSID code 5 merges permanently and temporarily disabled; there is no separate long-term sick category.
  - Code 2 merges temporary layoff with sick/maternity leave (UKHLS treats leave as employed).
  - Use mentions 2-3 for precedence rules, e.g. retired + disabled.

### hours_annual
- "HEAD/REF PERSON TOTAL HOURS OF WORK-yyyy" and "WIFE/SPOUSE ...". It covers the previous calendar year, all jobs including overtime, and is imputed.
- The maximum is 5,840 to 2011 and 5,824 from 2013.

### labour_income
- "LABOR INCOME (OF) HEAD/WIFE/REF PERSON/SPOUSE". Previous calendar year, imputed. It excludes farm income and the labour part of unincorporated business income (these are separate variables).
- RP: wages, bonuses, overtime, tips, commissions, professional practice, misc. labour income and extra job. The component list varies slightly by wave.
- SP 1999-2013: equal to the single raw G52 question "how much did she earn altogether from work", with DK/NA imputed (verified: identical for every valid G52 value). From 2015 the SP version is itemised like the RP's, which is a possible break in SP levels in 2015.

### wages
- RP: generated "WAGES AND SALARIES OF HEAD/REF PERSON-yyyy" (imputed).
  - Raw alternative: G13 "WAGES/SALARY" (ER14416 ... ER83121), which keeps 9999998=DK and 9999999=NA. Raw NA rises to 427 and 552 cases in 2021 and 2023.
- SP 1999-2013: there is no separate wife wage item. The only "WAGES/SALARY OF WIFE" variable (ER14757 ... ER54309) is the raw G52 total-earnings question, not imputed, and equal to labour_income SP. Confidence is medium; the generated labour_income SP is the better analysis variable.
- SP 2015+: generated "WAGES AND SALARIES OF SPOUSE-yyyy". The raw alternative is "G52/G13 WAGES/SALARY OF SPOUSE" (ER61349, ER67401, ER73424, ER79526, ER83495).

### year_retired
- Questions: B2/D2 (1999-2001), BC2/DE2 (2003+), "year retired". Asked only if any employment-status mention = 4.
- Codes: 9998=DK and 9999=NA; 0 = not retired, and for SP also no spouse.
- Data issues:
  - Wild codes: 2005 (RP 1085 and 9997; SP 1083) and 2007 (RP 2200).
  - About 110 RP NAs per wave in 2003-2007.
  - The wording becomes "(last) retire" from 2017.
- Not a UKHLS target item, but useful for checking retirement status.

