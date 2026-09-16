## Items coded exactly as the GRM banks, higher = better health, ages 20-90.
suppressMessages({library(data.table); library(arrow)})
R0 <- "/Users/julianashwin/Documents/GitHub/prevention-health-clustering"
it <- as.data.table(read_parquet(file.path(R0, "data/interim/sf12_items_long.parquet"),
  col_select = c("pidp", "wave", "age", "sf1", "sf2a", "sf2b", "sf3a", "sf3b", "sf5", "health", paste0("disdif", 1:12), "disdif96")))
cc <- as.data.table(read_parquet(file.path(R0, "data/processed/measures/chronic_conditions.parquet"),
  col_select = c("pidp", "wave", "n_cvd", "n_metab", "n_resp", "n_msk", "n_cancer", "n_other")))
d <- merge(it, cc, by = c("pidp", "wave"), all.x = TRUE)[!is.na(age) & age >= 20 & age <= 90]
v <- function(x, k) fifelse(x %in% 1:k, as.numeric(x), NA_real_)
d[, `:=`(GH = 6 - v(sf1, 5), PF = v(sf2a, 3) + v(sf2b, 3) - 1, RP = v(sf3a, 5) + v(sf3b, 5) - 1, BP = 6 - v(sf5, 5))]
asked <- rowSums(!is.na(d[, c(paste0("disdif", 1:12), "disdif96"), with = FALSE])) > 0
cnt <- rowSums(d[, paste0("disdif", c(1, 2, 3, 10, 11)), with = FALSE] == 1, na.rm = TRUE)
func <- fifelse(!is.na(d$health) & d$health == 2, 0, fifelse(!is.na(d$health) & d$health == 1 & asked, cnt, NA_real_))
d[, FUNC := 3 - pmin(func, 3)]
d[, `:=`(CVD = 2 - pmin(n_cvd, 2), METAB = 2 - pmin(n_metab, 2), RESP = 2 - pmin(n_resp, 2),
         MSK = 1 - pmin(n_msk, 1), CANCER = 1 - pmin(n_cancer, 1), OTHER = 1 - pmin(n_other, 1))]
SF <- c("GH", "PF", "RP", "BP"); CONDS <- c("CVD", "METAB", "RESP", "MSK", "CANCER", "OTHER")
CONTENT <- list(`SF-12` = SF, `SF-12 + limitations` = c(SF, "FUNC"), `SF-12 + limitations + conditions` = c(SF, "FUNC", CONDS))
