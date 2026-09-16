suppressMessages({library(data.table); library(arrow); library(tidyverse)})
SCR <- "/Users/julianashwin/Documents/GitHub/prevention-health-clustering/data/processed/baseline_measures"  # intermediate data, gitignored
setwd("/Users/julianashwin/Documents/GitHub/AnalysisForEIT/exploratory_w_johannes/redo_concepts")
cb <- as.data.table(read_parquet(file.path(SCR, "master_measures.parquet"))); M <- setdiff(names(cb), c("pidp", "wave", "age"))
s4 <- readRDS("data/ukhls_panel.rds")[, .(pidp, wave, s4)]; d <- merge(cb, s4, by = c("pidp", "wave"))[s4 == TRUE]
AGES <- 20:90; PPL <- unique(d$pidp)
W <- lapply(M, function(v) { X <- matrix(NA_real_, length(PPL), length(AGES)); ok <- !is.na(d[[v]])
  X[cbind(match(d$pidp[ok], PPL), match(d$age[ok], AGES))] <- d[[v]][ok]; X }); names(W) <- M
BANDS <- list(`25-40` = c(25, 40), `40-60` = c(40, 60), `60-75` = c(60, 75), `75-90` = c(75, 90))
lt <- function(S) S[lower.tri(S, diag = TRUE)]
pj <- local({ o <- c(0, 2, 4, 6); g <- expand.grid(r = 1:4, c = 1:4); g <- g[g$r >= g$c, ]; list(i = o[g$c], j = o[g$r]) })
mom <- function(v, lo, hi, how) { K <- 10; num <- rep(0, K); den <- rep(0, K)
  for (a0 in lo:hi) { cols <- match(a0 + c(0, 2, 4, 6), AGES); if (anyNA(cols)) next
    X <- W[[v]][, cols, drop = FALSE]; bal <- rowSums(!is.na(X)) == 4; if (sum(bal) < 150) next
    if (how == "balanced") { m <- lt(cov(X[bal, , drop = FALSE])); w <- rep(sum(bal), K) }
    else { m <- lt(cov(X, use = "pairwise.complete.obs")); w <- lt(crossprod(!is.na(X))) }
    if (anyNA(m)) next; num <- num + w * m; den <- den + w }
  num / den }
## Noise layers: Case 2 one AR(1) variance; Case 3 an AR(1) variance free at each age;
## Case 4 one AR(1) variance plus a one-period error on the diagonal, profiled on the floored
## grid of testing_metrics (below 0.30 the slow term is a second spike, collinear with it).
RHO <- list(case2 = seq(0.05, 0.97, 0.02), case3 = seq(0.05, 0.97, 0.02), case4 = seq(0.30, 0.97, 0.01))
design <- function(md, r) { sm <- cbind(1, pj$i + pj$j, pj$i * pj$j)
  switch(md, case2 = cbind(sm, r^abs(pj$i - pj$j)),
             case3 = cbind(sm, sapply(c(0, 2, 4, 6), \(k) r^(pj$j - pj$i) * (pj$i == k))),
             case4 = cbind(sm, r^abs(pj$i - pj$j), as.numeric(pj$i == pj$j))) }
ok_det <- function(b) b[1] >= 0 && b[3] >= 0 && b[2]^2 <= b[1] * b[3] * (1 + 1e-6) && all(b[-(1:3)] >= -1e-8)
## best-fitting rho, and the best-fitting rho at which the answer is admissible (NULL if none)
fit <- function(m, md) { best <- NULL; best_ok <- NULL
  for (r in RHO[[md]]) { X <- design(md, r); b <- tryCatch(qr.solve(X, m), error = function(e) NULL); if (is.null(b)) next
    ss <- sum((m - X %*% b)^2)
    if (is.null(best) || ss < best$ss) best <- list(b = b, ss = ss, r = r)
    if (ok_det(b) && (is.null(best_ok) || ss < best_ok$ss)) best_ok <- list(b = b, ss = ss, r = r) }
  list(best = best, best_ok = best_ok) }
yn <- function(x) if_else(x, "Yes", "No")
## c3 and c2 are the scorecard's two estimators; c2b and c3p swap their moments, and c4b and c4p
## are Case 4 on each. The sign path reads the sign of C[H,d], which is the sign of the correlation
## whenever that is defined; for Case 4 a second admissibility flag uses the valid-rho convention.
SPECS <- list(c3 = c("case3", "balanced"), c2 = c("case2", "pairwise"), c2b = c("case2", "balanced"),
              c3p = c("case3", "pairwise"), c4b = c("case4", "balanced"), c4p = c("case4", "pairwise"))
A <- map(M, function(v) {
  out <- list(key = v)
  moms <- map(c(balanced = "balanced", pairwise = "pairwise"), \(how) map(BANDS, \(b) mom(v, b[1], b[2] - 6, how)))
  for (nm in names(SPECS)) {
    md <- SPECS[[nm]][1]; how <- SPECS[[nm]][2]
    cr <- map(names(BANDS), \(bn) { f <- fit(moms[[how]][[bn]], md); bb <- f$best$b
      c(corr = if (bb[1] * bb[3] > 0) unname(bb[2] / sqrt(bb[1] * bb[3])) else NA_real_, chd = unname(bb[2]),
        ok = ok_det(bb), anyok = !is.null(f$best_ok), chd_ok = if (is.null(f$best_ok)) NA_real_ else unname(f$best_ok$b[2])) })
    cr <- do.call(rbind, cr)
    out[[paste0(nm, "_admissible")]] <- yn(all(cr[, "ok"] == 1))
    out[[paste0(nm, "_signpath")]] <- yn(cr[1, "chd"] > 0 && cr[4, "chd"] < 0)
    for (k in 1:4) out[[paste0(nm, "_corr_", names(BANDS)[k])]] <- cr[k, "corr"]
    if (md == "case4") {
      out[[paste0(nm, "_admissible_validrho")]] <- yn(all(cr[, "anyok"] == 1))
      out[[paste0(nm, "_signpath_validrho")]] <- yn(isTRUE(cr[1, "chd_ok"] > 0 && cr[4, "chd_ok"] < 0))
    }
  }
  as_tibble(out) }) |> list_rbind()
lp <- as.data.table(readRDS("/Users/julianashwin/Documents/Research/LowDim/data/ukhls_long_panel.rds"))[, .(pidp, wave, died_next_wave)]
mm <- merge(cb, lp, by = c("pidp", "wave"))[!is.na(died_next_wave) & between(age, 20, 90)]
mm <- mm[complete.cases(mm[, ..M])]
MO <- map(M, function(v) { x <- mm[[v]]; y <- mm$died_next_wave; z <- (x - mean(x)) / sd(x)
  b <- pmin(9, floor(rank(z, ties.method = "average") / length(z) * 10)); r <- tapply(y, b, mean)
  tibble(key = v, mort_slope = unname(coef(glm(y ~ z, family = binomial))[2]), mort_ratio = unname(r[1] / r[length(r)])) }) |> list_rbind()
res <- A |> left_join(MO, by = "key")
fwrite(res, file.path(SCR, "master_r.csv"))
cat(sprintf("identification sample: s4 person-ages; mortality common sample %s person-waves, %d deaths\n", format(nrow(mm), big.mark = ","), sum(mm$died_next_wave)))
options(width = 250); print(as.data.frame(res |> mutate(across(where(is.numeric), \(x) round(x, 2)))), row.names = FALSE)
