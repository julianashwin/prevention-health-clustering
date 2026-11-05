#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/coauthor_papers}"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
get() { local d="$OUT/$1" f="$2" u="$3"; mkdir -p "$d"
  if [ -s "$d/$f" ] && head -c 4 "$d/$f" | grep -q '%PDF'; then echo "skip   $1/$f"; return; fi
  if curl -sSL -A "$UA" --max-time 120 -o "$d/$f" "$u" && head -c 4 "$d/$f" | grep -q '%PDF'; then echo "ok     $1/$f"; else echo "FAILED $1/$f  <- $u"; rm -f "$d/$f"; fi; }
S=https://personal.lse.ac.uk/spinnewi
for p in Health_Inequality_draft LTU_Belgium_draft SHI_draft LTU_MS AERInsights KLRS_draft valueUI DMS_draft DISTL_draft selectionUI Insurance_Identification MST_draft registryconsumption Frictions_Selection_Policies_WEB dynamicUI perceptions_welfare perceptions_education PakistanCIT bargaining biasedbeliefs information screening training capital cepr_pension RiskScores_EMS UI_LSE HB_expectations ARE_draft UIreview; do get spinnewijn "$p.pdf" "$S/$p.pdf"; done
get spinnewijn PP_atomistic_ecological_fallacies.pdf "$S/P&P_draft.pdf"
get scott Ashwin_Scott_etal_2026_reframing_epidemiological_transition_NatureAging.pdf https://www.nature.com/articles/s43587-026-01210-2.pdf
get scott Ashwin_Scott_2025_bayesian_later_life_mortality_JRSSA.pdf https://profandrewjscott.com/wp-content/uploads/2025/01/Siler_International-84.pdf
get scott Ellison_Scott_Sinclair_2021_economic_value_targeting_aging.pdf https://profandrewjscott.com/wp-content/uploads/2021/01/allswell_080121-1.pdf
get scott Old_Scott_2023_healthy_ageing_trends_England_JEoA.pdf https://lbsresearch.london.edu/id/eprint/3021/1/1-s2.0-S2212828X23000300-main.pdf
get scott Old_Scott_2023_healthy_ageing_trends_England_WP.pdf https://profandrewjscott.com/wp-content/uploads/2023/07/Longevity_Health_Ageing_ELSA-311.pdf
get scott Kotschy_Bloom_Scott_2024_limits_chronological_age_IZA.pdf https://docs.iza.org/dp17427.pdf
get scott Kotschy_Bloom_Scott_2024_limits_chronological_age_revision.pdf https://profandrewjscott.com/wp-content/uploads/2024/11/KBS_2024-On-the-Limits-of-Chronological-Age_Revision.pdf
get scott Ellison_Scott_2017_managing_UK_national_debt.pdf https://www.lse.ac.uk/CFM/assets/pdf/CFM-Discussion-Papers-2017/CFMDP2017-27-Paper.pdf
get scott Scott_2021_older_persons_post_covid_UN.pdf https://www.un.org/development/desa/ageing/wp-content/uploads/sites/24/2021/02/Andrew-Scott_Paper.pdf
get ashwin Ashwin_Chhabra_Rao_2023_LLM_qualitative_bias_arxiv.pdf https://arxiv.org/pdf/2309.17147v2
get ashwin Ashwin_Chhabra_Rao_2023_LLM_qualitative_bias_WorldBank.pdf https://documents1.worldbank.org/curated/en/099433311072326082/pdf/IDU09959393309484041660b85d0ab10e497bd1f.pdf
get ashwin Ashwin_Beaudry_Ellison_2024_neural_network_learning.pdf https://users.ox.ac.uk/~exet2581/NNL_for_NLE.pdf
get ashwin Ashwin_Kalamara_Saiz_2021_nowcasting_news_ECB.pdf "https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2616~58494f90b7.en.pdf"
get ashwin Ahrens_Ashwin_Calliess_Nguyen_2021_bayesian_topic_regression.pdf https://arxiv.org/pdf/2109.05317
echo "Done. Files in $OUT"
