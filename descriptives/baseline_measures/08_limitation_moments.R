## Exhibit 1 and identification-barchart inputs for the limitation banks, h scale.
## Same sample and estimators as 06_master_ident.R: person-ages flagged s4 in
## the redo_concepts panel, four ages two years apart pooled over base ages
## within four bands. Cases 2 (one AR(1) variance) and 3 (AR(1) variance free by
## age) profile rho over 0.05-0.97; Case 4 (AR(1) plus a one-period error) over
## 0.30-0.97. Each runs on balanced and on pairwise moments, at the
## best-fitting rho and at the best-fitting admissible rho. Writes
##   limitation_profiles.csv       mean (pooled-sd units) and variance relative
##                                 to ages 30-34 by age, 5-year centred means
##   limitation_corr_band.csv      Corr(H, d) and admissibility by band
##   limitation_decomposition.csv  observed moments and fitted layers by cell
suppressMessages({library(data.table); library(arrow); library(tidyverse)})
R0 <- "/Users/julianashwin/Documents/GitHub/prevention-health-clustering"
OUT <- file.path(R0, "data/processed/baseline_measures")   # intermediate data, gitignored
setwd("/Users/julianashwin/Documents/GitHub/AnalysisForEIT/exploratory_w_johannes/redo_concepts")
SLUG <- c(`P-FUNC` = "pfunc", `P-LIM1` = "plim1", `P-LIM` = "plim", `P-LIM4` = "plim4", `P-LIM3` = "plim3", `P-LIM3+O` = "plim3o")
MEAS <- c(as.vector(outer(SLUG, c("grm", "gpcm"), \(s, m) paste0("h_", s, "_", m))),
          as.vector(outer(SLUG[-1], c("cc", "cg"), \(s, cd) paste0("h_", s, cd, "_grm"))))
sc <- as.data.table(read_parquet(file.path(R0, "data/processed/measures/limitation_scores.parquet"),
                                 col_select = c("pidp", "wave", "age", all_of(MEAS))))
## the linear twins of 12_simple_twins.py, so the same moments can be read off them
tw <- as.data.table(read_parquet(file.path(OUT, "simple_twins.parquet")))
TWIN <- setdiff(names(tw), c("pidp", "wave", "age"))
sc <- merge(sc, tw[, c("pidp", "wave", ..TWIN)], by = c("pidp", "wave"), all.x = TRUE)
MEAS <- c(MEAS, TWIN)
## controls for the condition banks: each limitation bank's graded-response score
## restricted to the rows with a condition inventory, so the comparison holds the people fixed
for (s in SLUG[-1]) sc[[paste0("h_", s, "_grm_condsample")]] <- fifelse(is.na(sc$h_plim3cc_grm), NA_real_, sc[[paste0("h_", s, "_grm")]])
MEAS <- c(MEAS, paste0("h_", SLUG[-1], "_grm_condsample"))
s4 <- readRDS("data/ukhls_panel.rds")[, .(pidp, wave, s4)]
d <- merge(sc, s4, by = c("pidp", "wave"))[s4 == TRUE]
AGES <- 20:90; PPL <- unique(d$pidp)
W <- lapply(MEAS, function(v) { X <- matrix(NA_real_, length(PPL), length(AGES)); ok <- !is.na(d[[v]])
  X[cbind(match(d$pidp[ok], PPL), match(d$age[ok], AGES))] <- d[[v]][ok]; X }); names(W) <- MEAS
BANDS <- list(`25-40` = c(25, 40), `40-60` = c(40, 60), `60-75` = c(60, 75), `75-90` = c(75, 90))
lt <- function(S) S[lower.tri(S, diag = TRUE)]
offs <- c(0, 2, 4, 6)
pj <- local({ g <- expand.grid(r = 1:4, c = 1:4); g <- g[g$r >= g$c, ]; list(i = offs[g$c], j = offs[g$r]) })
mom <- function(v, lo, hi, how) { K <- 10; num <- rep(0, K); den <- rep(0, K)
  for (a0 in lo:hi) { cols <- match(a0 + offs, AGES); if (anyNA(cols)) next
    X <- W[[v]][, cols, drop = FALSE]; bal <- rowSums(!is.na(X)) == 4; if (sum(bal) < 150) next
    if (how == "balanced") { m <- lt(cov(X[bal, , drop = FALSE])); w <- rep(sum(bal), K) }
    else { m <- lt(cov(X, use = "pairwise.complete.obs")); w <- lt(crossprod(!is.na(X))) }
    if (anyNA(m)) next; num <- num + w * m; den <- den + w }
  num / den }
RHO <- list(case2 = seq(0.05, 0.97, 0.02), case3 = seq(0.05, 0.97, 0.02), case4 = seq(0.30, 0.97, 0.01))
design <- function(md, r) { sm <- cbind(1, pj$i + pj$j, pj$i * pj$j)
  switch(md, case2 = cbind(sm, r^abs(pj$i - pj$j)),
             case3 = cbind(sm, sapply(offs, \(k) r^(pj$j - pj$i) * (pj$i == k))),
             case4 = cbind(sm, r^abs(pj$i - pj$j), as.numeric(pj$i == pj$j))) }
ok_det <- function(b) b[1] >= 0 && b[3] >= 0 && b[2]^2 <= b[1] * b[3] * (1 + 1e-6) && all(b[-(1:3)] >= -1e-8)
fit <- function(m, md) { best <- NULL; best_ok <- NULL
  for (r in RHO[[md]]) { X <- design(md, r); b <- tryCatch(qr.solve(X, m), error = function(e) NULL); if (is.null(b)) next
    ss <- sum((m - X %*% b)^2)
    if (is.null(best) || ss < best$ss) best <- list(b = b, ss = ss, r = r, X = X)
    if (ok_det(b) && (is.null(best_ok) || ss < best_ok$ss)) best_ok <- list(b = b, ss = ss, r = r, X = X) }
  list(best = best, best_ok = best_ok) }
corr_of <- function(b) if (b[1] * b[3] > 0) b[2] / sqrt(b[1] * b[3]) else NA_real_
SPECS <- c("case2", "case3", "case4")
cells <- tibble(i = pj$i, j = pj$j) |>
  mutate(cell = if_else(i == j, paste0("V(", i / 2, ")"), paste0("C(", i / 2, ",", j / 2, ")")),
         region = case_when(i == j ~ "diagonal", i == 0 ~ "first row", TRUE ~ "interior"))
dec <- list(); cor_rows <- list()
for (v in MEAS) for (how in c("balanced", "pairwise")) for (bn in names(BANDS)) {
  b <- BANDS[[bn]]; m <- mom(v, b[1], b[2] - 6, how)
  for (md in SPECS) {
    f <- fit(m, md); bb <- f$best$b; X <- f$best$X; k <- ncol(X)
    spike_col <- if (md == "case4") k else integer(0)
    ar_cols <- setdiff(4:k, spike_col)
    noise <- as.numeric(X[, ar_cols, drop = FALSE] %*% bb[ar_cols])
    spike <- if (md == "case4") as.numeric(X[, k] * bb[k]) else rep(0, nrow(X))
    vo <- f$best_ok
    cor_rows[[length(cor_rows) + 1]] <- tibble(measure = v, spec = md, moments = how, band = bn,
      corr = corr_of(bb), chd = bb[2], ok = ok_det(bb), rho = f$best$r, any_valid = !is.null(vo),
      corr_valid = if (is.null(vo)) NA_real_ else corr_of(vo$b), chd_valid = if (is.null(vo)) NA_real_ else vo$b[2],
      rho_valid = if (is.null(vo)) NA_real_ else vo$r)
    dec[[length(dec) + 1]] <- bind_cols(cells, tibble(measure = v, spec = md, moments = how, band = bn, observed = m,
      VH = bb[1], CHd = (pj$i + pj$j) * bb[2], Vd = pj$i * pj$j * bb[3], noise = noise, spike = spike,
      ok = ok_det(bb), rho = f$best$r))
  }
}
fwrite(list_rbind(cor_rows), file.path(OUT, "limitation_corr_band.csv"))
fwrite(list_rbind(dec), file.path(OUT, "limitation_decomposition.csv"))
prof <- map(MEAS, function(v) { x <- d[[v]]; ok <- !is.na(x); zz <- (x - mean(x[ok])) / sd(x[ok])
  tibble(age = d$age[ok], z = zz[ok]) |> summarise(mean = mean(z), var = var(z), n = n(), .by = age) |>
    filter(between(age, 22, 88)) |> arrange(age) |>
    mutate(mean = as.numeric(stats::filter(mean, rep(1/5, 5), sides = 2)),
           var = as.numeric(stats::filter(var, rep(1/5, 5), sides = 2)), measure = v) }) |> list_rbind() |> drop_na()
v0 <- prof |> filter(between(age, 30, 34)) |> summarise(v0 = mean(var), .by = measure)
prof <- prof |> left_join(v0, by = "measure") |> mutate(vrel = var / v0)
fwrite(prof, file.path(OUT, "limitation_profiles.csv"))
cr <- list_rbind(cor_rows) |> mutate(corr = round(corr, 3)) |> pivot_wider(id_cols = c(measure, spec, moments), names_from = band, values_from = corr)
print(as.data.frame(cr), row.names = FALSE)
