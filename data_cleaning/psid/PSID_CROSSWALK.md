# PSID variables for the UKHLS analysis, 1999–2023

This note goes with `psid_crosswalk.csv`, which lists, for every concept, wave and person, the PSID variable that records it. `01_build_psid_panel.py` reads that file and builds the person-wave panel. Waves are the 13 biennial family files from 1999 to 2023, plus the cross-year individual file IND2023ER and the childbirth history CAH85_23.

## How the mapping was checked

Each variable was identified twice, independently. I did one pass. Five separate agents did the other, working only from the codebooks and the data and without seeing my answers: functioning, chronic conditions, mental health and spending, the individual file, and income and labour. Every variable was then checked three ways: the full codebook entry (question wording, codes and who was asked), the value counts in the data against the codebook frequencies, and stability across waves and between the Reference Person and the Spouse.

The two passes found the same variable in every one of the 2,625 variable-by-wave entries that both of them list, with no disagreements. The only judgement call is which 2011 health-insurance question to use, since that wave asks about both the past two years and the present.

| Domain | Found by both, identical | Found by the agents only | Found by me only |
|---|---|---|---|
| Self-rated health, work limitation, ADL, IADL, mobility, hospital stays, height and weight | 942 | 0 | 0 |
| Chronic conditions: ever diagnosed, onset, limits, type mentions | 1,036 | 144 | 78 |
| K6, family respondent, out-of-pocket spending, insurance | 166 | 71 | 0 |
| Individual file: IDs, sequence number, relationship, age, employment status, education, weights, death year | 201 | 77 | 13 |
| Income, education, race, region, employment, hours (family file) | 280 | 239 | 8 |

Entries found by one pass only are extra items the other pass was not asked for, such as degree questions, hours and wages, or the second and third mentions of a cancer or psychiatric type. I checked each agent-only entry against its label and person, and each of mine against the codebook and the data. The `second_pass` column in the CSV records which applies to each row, and `codes` and `notes` give the response codes and who was asked.

Finally, each PSID item was matched to the UKHLS variable that the repo's own code uses (`src/prevention_health_clustering/measures/` and `data_cleaning/0*.py` on `main`), not to the paper's description of it. The tables below name both.

## Structural differences from the UKHLS

Read these first, because they matter more than any single variable.

1. **Two-year spacing.** The waves are two years apart, so the covariance rows step in two-year lags and the one-year "spike" in the UKHLS moments cannot be seen directly.
2. **Only two people per family answer the health questions.** The Reference Person (RP, called Head before 2017) and the Spouse or long-term partner (SP, called Wife before 2017). Other adults in the family have no health data. The panel is therefore RP and SP person-waves: 169,223 person-waves aged 20–89 (the contracts' age window), from 25,489 people, of whom 19,955 have three or more waves.
3. **Proxy reports.** One family respondent answers the health section for both RP and SP. RP answers are self-reports in 69 percent of rows and SP answers in 57 percent. The panel carries a `self_report` flag. The UKHLS health items are all self-reports.
4. **Reference periods differ.** Hospital stays and spending refer to the previous calendar year rather than the past 12 months. Conditions are "has a doctor ever told you", re-asked in full every wave. Between 13 and 53 percent of "yes" answers turn into "no" two years later, depending on the condition. The UKHLS asks the full list only at a person's first interview and then only about new diagnoses, and `measures/chronic.py` cumulates the answers (once diagnosed, always diagnosed). Doing the same in PSID makes the count grow with time in the panel, so the panel carries three versions of it; see COND below.
5. **Interview mode.** About 40 percent of 2021 interviews and 73 percent of 2023 interviews were done on the web. This raises K6 scores and item non-response on spending.

## The health measure, testlet by testlet

UKHLS definitions are from `measures/grm.py`, `measures/grm2.py` and `measures/health.py`.

| Testlet | UKHLS (repo code) | PSID item | Panel column | Waves | How close |
|---|---|---|---|---|---|
| GH | `sf1` general health, 5 categories | H1 "Would you say your health in general is excellent, very good, good, fair, or poor?" | `srh` | All | Same question and categories |
| PF | `sf2a` moderate activities + `sf2b` climbing several flights of stairs, each limited a lot, a little or not at all | H8a3 "any trouble either walking several blocks or climbing a few flights of stairs, because of your health?" (yes/no) | `mob_walk_climb` | 2017–2023 | Close to `sf2b` in content but binary. Nothing comparable before 2017 |
| RP | `sf3a` accomplished less + `sf3b` limited in kind of work or activities, because of physical health, past four weeks, five frequencies | H2–H4: a "physical or nervous condition that limits the type or amount of work", then how much, or cannot work at all | `worklim_ord` (0–3) | All | Same idea, but no four-week window, work only, and it includes "nervous" conditions |
| BP | `sf5` pain interfered with work | None | – | – | Not available in any wave |
| LIM_FL | Count of four impairment areas, `disdif` 1 mobility ("moving around at home and walking"), 2 lifting or carrying, 3 manual dexterity, 10 co-ordination; no long-standing illness counts as none | Mobility: ADL difficulty walking, getting in or out of bed or a chair, or getting outside (any of the three). Lifting: H8a4 "trouble bending, lifting, or stooping" | `lim_fl` (mobility, 0–1); `lim_fl_17` (mobility + lifting, 0–2) | `lim_fl` all; `lim_fl_17` 2017+ | Two of the four areas. No dexterity or co-ordination item |
| LIM_SELF | `disdif` 11 personal care ("washing, dressing") and 4 continence | ADL difficulty bathing, dressing, eating or using the toilet (any of the four) | `lim_self` (0–1) | All; ages 55+ only in 1999 and 2001 | Personal care matches. No continence item |
| LIM_SENS | `disdif` 5 hearing and 6 sight | None for adults | – | – | Not available |
| COND | Count of 16 ever-diagnosed physical conditions in six groups, top category 6+ | Nine "doctor ever told you" conditions | `cond_count_conf` (suggested), `cond_count_phys`, `cond_count_now` | All | Fewer conditions, and a choice of how to cumulate: see the next two sections |

The UKHLS impairment areas are counts of areas of difficulty. PSID asks ADL difficulty item by item, so an area counts if the person has difficulty with any of its ADL items. The IADL items (housework, meals, shopping, money, telephone) have no UKHLS counterpart in the measure, so they are kept only as frailty deficits.

### COND, condition by condition

UKHLS codes and groups are from `measures/chronic.py`.

| UKHLS group | UKHLS conditions (`hcond` code) | PSID item |
|---|---|---|
| Cardiovascular | congestive heart failure (3), coronary heart disease (4), angina (5), heart attack (6), stroke (7) | One item for "coronary heart disease, angina, congestive heart failure", plus heart attack and stroke |
| Metabolic | diabetes (14), high blood pressure (16) | Diabetes or high blood sugar; high blood pressure |
| Respiratory | asthma (1), emphysema (8), chronic bronchitis (11) | Asthma; one item for "chronic lung disease such as bronchitis or emphysema" |
| Musculoskeletal | arthritis (2) | Arthritis or rheumatism |
| Cancer | cancer (13) | Cancer or a malignant tumour, with non-melanoma skin cancer dropped in every wave, as the 1999–2003 question did |
| Other | hyperthyroidism (9), hypothyroidism (10), liver condition (12), epilepsy (15) | None. The open-ended "other serious chronic condition" (2007+) has thyroid and seizure codes, but only 0.6 and 0.3 percent of RPs named them in 2019, far below their prevalence, so they stay out |
| Not in COND | clinical depression (17) | "Emotional, nervous or psychiatric problems"; depression is the type mention, 2005+ |

So the PSID count runs to 9 rather than 16, and is lower for the same health, because PSID asks one question where the UKHLS asks two or three. With the repo's merge rule (a top category under 0.5 percent of person-waves merges down), the top category is 6 or more for `cond_count_conf` and `cond_count_now`, as in the UKHLS, and 7 or more for the cumulated `cond_count_phys` (2003–2023, ages 20–89).

### Which condition count to use

Cumulating the PSID answers the UKHLS way makes the count drift up with time in the panel. PSID re-asks the whole list every wave, so every wave is a fresh chance to report a condition, rightly or wrongly, and cumulation keeps any "yes" for good. UKHLS respondents answer the full list once and are then asked only about new diagnoses. At ages 50–59 (2011 on), the cumulated count averages 1.16 at a person's first wave and 1.6 to 1.7 after four or more waves, while the count from each wave's own answers stays between 1.07 and 1.22. Requiring a condition to be reported twice removes more than half of that rise, so most of it comes from single reports that are never repeated. The builder therefore gives three versions, each built from the matching per-condition columns:

| Panel column | Built from | Rule | Mean at 50–59: first wave, 4+ waves later |
|---|---|---|---|
| `cond_count_phys` | `*_ever_abs` | Cumulated: once reported, always counted (the UKHLS rule) | 1.16, 1.6–1.7 |
| `cond_count_conf` | `*_ever_conf` | Reported this wave, or in at least two waves so far. Uses no later waves, so it is safe in prediction exercises | 1.16, 1.3–1.4 |
| `cond_count_now` | `*_ever` | This wave's answers only | 1.16, 1.1–1.2 |

Decision (5 November): the measure uses `cond_count_phys`, the UKHLS rule, to stay as close as possible to the UKHLS version. `cond_count_conf` and `cond_count_now` are kept for robustness checks.

## Frailty index

The UKHLS index (`build_frailty_index`) is the share of 31 equal-weight deficits: the six SF-12 physical items graded 0 to 1, the eight impairment areas, and the 17 inventory conditions including depression, scored only when every deficit is observed.

PSID has GH, work limitation, PF and lifting (2017+), mobility, personal care, the nine conditions and the psychiatric item. That is 14 deficits before 2017 and 16 from 2017. Hosseini, Kopecky and Zhao (2022), whom the paper cites for log frailty, build their frailty index on PSID from the ADL, IADL and condition items, each counted as its own deficit, so the PSID index follows them: it uses everything PSID asks and has a published PSID benchmark.

`data_cleaning/psid/03_build_psid_frailty.py` builds it from 28 deficits, as in their Appendix Table 1: the 13 ADL and IADL difficulties, the 12 ever-diagnosed conditions (kept once reported, the `*_ever_abs` columns), obesity (BMI 30 or more), has ever smoked and smokes now. The two smoking items come from the family file: H13 "Do you smoke cigarettes?" (`smoke_now`) and H16 "Did you ever smoke cigarettes?" (`smoke_everq`), which is asked only of people who do not smoke now, so `ever_smoked` is 1 for current smokers and the H16 answer for everyone else (the two items for the RP and the SP in each of the 13 waves, 52 rows of the CSV). A person-wave is scored only when every deficit is observed, as in the UKHLS code. "Another serious chronic condition" is first asked in 2005, so in 2003 the index is the share of the 27 deficits asked; scoring 2005 on those 27 moves the mean by 0.002 and keeps the ranks (Spearman 0.985). On their sample (heads and spouses aged 25 and over, 2003–2017) the index reproduces their Table 1: mean 0.111 against 0.11, by age band 0.081, 0.144 and 0.256 against 0.08, 0.15 and 0.25, median 0.071 against 0.07, 95th percentile 0.357 against 0.36, and 95,532 person-waves on 20,120 people against 96,759 on 19,996.

## Mental health

The UKHLS mental GRM (`measures/grm2.py`) uses the SF-12 mental items (MH: `sf6a`, `sf6c`; RE: `sf4a`, `sf4b`; SF: `sf7`) and the GHQ-12 in two wording testlets, without the depression diagnosis. PSID has none of these. Its only distress scale is the Kessler K6 (six items on the past 30 days, 0–24), asked in 2001, 2003 and 2007–2023. Only the family respondent answers it, so it is on 64 percent of person-waves in those years, and 12,117 people have it at least three times. It measures the same thing as the GHQ-12, so a one-channel mental score is possible on a smaller sample.

## Cost proxy

The UKHLS index (`measures/cost.py`, waves 7–15) prices GP visits (`hl2gp`, band 0–4), out-patient attendances (`hl2hop`, band 0–4) and in-patient stays (`hosp`, with nights `hospd`), and drops stays for childbirth (`hospch`).

PSID has no GP or out-patient counts in any wave. It has whether the person was in hospital overnight in the previous calendar year and for how many nights, for RP and SP from 2003. It does not ask why, so the panel flags women who gave birth in that calendar year (from the childbirth history) and, as the UKHLS code does, treats their stay as maternity: `hosp_any_nonmat` and `hosp_nights_nonmat` drop it. That removes 2,922 stays and lowers the stay rate from 10.1 to 8.1 percent (2003+, ages 20–89). The UKHLS in-patient formula can be applied unchanged; the GP and out-patient parts will be missing. The family file also has out-of-pocket spending on hospitals, doctors and prescriptions, but only for the family as a whole.

## Mortality

The UKHLS contract uses `dcsedfl_dv == 1` (a death learnt at a later wave) and puts the event on the person's last observed wave. The PSID counterpart is `ER32050`, the year of death: an exact year, or a range written as two two-digit years (709 means 2007–2009). Use it rather than the "died since last interview" sequence numbers 81–89, which catch only about half the deaths, because people who lived alone drop out of the family files when they die.

- `died`: the study knows the person died (2,727 people in the panel).
- `mort_event`: 1 on the person's last panel row if they died within two years of that interview (2,511 people), 0 elsewhere. PSID also traces deaths long after a family stops responding, which the UKHLS cannot; the 216 people who died more than two years after their last interview are treated as censored, though `died` and `death_year_lo` keep them.
- `died_by_next_wave`: the same on every row, missing in 2023.
- `death_conflict`: 10 people have a death year earlier than their last interview.

Two-year death rates by age are 0.2 percent at 20–39, 0.6 at 40–49, 1.3 at 50–59, 2.6 at 60–69, 6.4 at 70–79 and 16.7 at 80–89, about what US life tables give.

## Observables

UKHLS definitions are from `data_cleaning/06_build_observables.py` and `descriptives/_paper_common.py`.

| UKHLS | PSID panel column |
|---|---|
| Sex | `sex` (individual file) |
| Ethnicity: first valid `racel_dv`, white against other, or five groups (white, mixed, Asian, Black, other) | `race_group`, per person: Hispanic, white, Black, Asian, other (US groups in place of the UK ones). Also `white_nonhisp` per row. Hispanic origin is asked from 2005; in 1999–2003 "Latino" was a race code |
| Highest qualification: best recorded at age 25 or over, in four groups (degree or higher, A-level, GCSE, other or none) | `educ_best`, per person: best level at 25 or over, in four groups (less than high school, high school or GED, some college, bachelor's or more) |
| Equivalised household net income in the month before interview (modified OECD scale), percentile rank among adult respondents in the wave | `faminc_eq` and `faminc_rank`: total family income in the previous calendar year, before tax, over the modified OECD scale built from the ages of everyone in the family; percentile rank among RP and SP rows in the wave |
| Region: first valid `gor_dv` (12 regions) | `region` (four census regions) and `state` |
| Birth year (`birthy`) | `birth_year`: the most common value of the per-wave birth year |
| Weights | Individual longitudinal and cross-sectional weights every wave |

## Labour market

The paper's employment checks use `jbstat`: 1–2 employed, 4 retired, 8 long-term sick or disabled. The PSID counterparts are `employed` (working now, or temporarily laid off or on sick or maternity leave, which is under 1 percent), `retired` and `lt_sick_disabled` ("permanently or temporarily disabled"), from the individual file's prioritised status `emp_status`. 77 percent of the panel are employed at 25–64. There is no self-employed split. The family file adds up to three status mentions, annual hours, wages and labour income for both RP and SP.

## Breaks to handle, by wave

- **1999 and 2001:** ADLs asked only of people aged 55+. No IADLs, hospital stays or two-year health change. Condition onset is recorded as duration, not age.
- **1999 and 2005:** no K6.
- **2005:** conditions renumbered and "other chronic condition" added. The cancer question stops excluding skin cancer; to keep the series consistent, cases whose only cancer is non-melanoma skin cancer are dropped.
- **2005 to 2007:** heart disease prevalence falls from 7.7 to 5.0 percent for RPs and from 4.8 to 2.5 percent for SPs, with no change in wording. Treat it as a possible break.
- **2009 to 2011:** "doesn't do" answers on the IADL items halve, with a small wording change.
- **2013:** most ADL and IADL items skipped for people in excellent health with no work limitation. The panel codes these as "no difficulty", as the UKHLS code does for people with no long-standing illness.
- **2017:** the two mobility items start, Head and Wife become Reference Person and Spouse, and same-sex couples are coded as spouses or partners.
- **2021:** about 50 families answered "don't know" to the whole health section (set to missing). Web interviewing starts.
- **Spending:** 1999–2011 questions cover two calendar years, and PSID's generated amounts halve them. 2013 is partly one-year. From 2015 the questions cover the previous year.
- **2021 and 2023:** the non-maternity hospital stay rate (ages 20–89) falls from about 8 percent in 2015–2019 to 6.1 and 6.3 percent. The 2021 answer refers to 2020, when elective care was postponed; that it stays low in 2023 may partly be the web mode. K6 rises in the same waves: the mean goes from about 3.45 to 3.97 and 4.25, and the share scoring 13 or more from 3.9 to 5.4 and 6.3 percent. Both call for wave effects or a check without these waves.
- **Employment** at 25–64 falls from 80 percent before 2009 to 74–75 percent in 2009–2013 and 76 percent in 2021. These are the recession and COVID rather than changes in the questions, but they matter for the employment checks.

## The panel builder

`01_build_psid_panel.py` reads the parquet files and this crosswalk and writes one row per RP or SP per wave, with every item recoded (yes = 1, no = 0; don't know, refused and not asked become missing, except the 2013 skip above). On top of the raw items it adds the UKHLS-style variables named above (`lim_fl`, `lim_fl_17`, `lim_self`, `cond_count_conf`, `cond_count_phys`, `cond_count_now`, `worklim_ord`, `died`, `mort_event`, `hosp_nights_nonmat`, `employed`, `educ_best`, `race_group`, `birth_year`, `faminc_rank`), plus each condition in three versions (`*_ever`, `*_ever_conf`, `*_ever_abs`), BMI, smoking (`smoke_now`, `smoke_everq`, `ever_smoked`) and K6 on the respondent's own row. It runs in under a minute and gives 170,710 person-waves for 25,628 people (all ages) and 223 columns.

```bash
.venv/bin/python data_cleaning/psid/01_build_psid_panel.py
```

The default paths are `data/raw/psid/parquet`, `data_cleaning/psid/psid_crosswalk.csv` and `data/processed/psid/psid_panel.parquet`.

`PSID_CROSSWALK_DETAIL.md` has the second pass's notes, domain by domain: question wording in each wave, code changes, who was asked, and data quirks.

## Decisions for you

1. **Graded-response model.** Re-estimate it on PSID with the testlets PSID has (GH, RP, LIM_FL, LIM_SELF and COND in every wave, PF and the lifting area from 2017), rather than anchor PSID items to the UKHLS parameters. I recommend re-estimating, because BP and LIM_SENS are missing and three more testlets are narrower.
2. **Start year.** 1999 and 2001 lack IADLs and hospital stays and ask ADLs only of people aged 55+. Starting in 2003 gives 11 waves with complete items; starting in 1999 gives 13 with gaps the model has to absorb.
3. **Proxy answers.** Keep them with the `self_report` flag (my suggestion), or restrict the health items to self-reports, which would drop about 36 percent of rows.
4. **Cost proxy.** Price in-patient nights only (2003+), at NHS or US unit costs, or use family out-of-pocket spending instead.
5. **Condition count.** Use `cond_count_conf` for COND (my suggestion), the UKHLS rule `cond_count_phys`, or each wave's own answers, `cond_count_now`.
