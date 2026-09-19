## One-factor weights for the limitation banks, on exactly the rows and codes
## the GRM and GPCM were fitted to (data_cleaning/archive/07_build_limitation_banks.py).
## Pearson and polychoric correlations are computed once over every testlet:
## both are pairwise quantities, so each bank's matrix is a submatrix.
## Weights follow the workbook convention: loading over uniqueness on
## standardised items, summing to one; the first principal component of the
## Pearson matrix for reference.
suppressMessages({library(data.table); library(arrow); library(lavaan)})
R0 <- "/Users/julianashwin/Documents/GitHub/prevention-health-clustering"
OUT <- file.path(R0, "data/processed/baseline_measures")   # intermediate data, gitignored
X <- as.data.table(read_parquet(file.path(R0, "data/processed/measures/limitation_bank_items.parquet")))
D0 <- X   # all P-FUNC-sample rows; condition items are NaN outside the condition sample
SF <- c("GH", "PF", "RP", "BP")
BANKS <- list(`P-FUNC` = "FUNC", `P-LIM1` = "LIM_PF", `P-LIM` = c("LIM_PF", "LIM_SC"),
              `P-LIM4` = c("LIM_MOB", "LIM_DEX", "LIM_CONT", "LIM_SENS"),
              `P-LIM3` = c("LIM_FL", "LIM_SELF", "LIM_SENS"),
              `P-LIM3+O` = c("LIM_FL", "LIM_SELF", "LIM_SENS", "LIM_OTHER"))
ALL <- unique(c(SF, unlist(BANKS)))
D <- as.data.frame(D0[, ..ALL])
R_p <- cor(D)
t0 <- Sys.time()
R_y <- lavCor(D, ordered = ALL, output = "cor")
cat(sprintf("%s rows; polychoric matrix over %d testlets in %.0f s\n", format(nrow(D), big.mark = ","), length(ALL),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
one_factor <- function(Rm, n) {
  f <- factanal(covmat = Rm, factors = 1, n.obs = n, rotation = "none")
  lam <- f$loadings[, 1]; lam <- lam * sign(sum(lam))
  list(lam = lam, psi = f$uniquenesses)
}
rows <- list()
for (b in names(BANKS)) {
  it <- c(SF, BANKS[[b]])
  for (cm in c("pearson", "polychoric")) {
    Rm <- if (cm == "pearson") R_p[it, it] else R_y[it, it]
    f <- one_factor(Rm, nrow(D)); w <- f$lam / f$psi
    rows[[length(rows) + 1]] <- data.table(bank = b, item = it, method = paste0(cm, "_factor"),
                                           loading = f$lam, uniqueness = f$psi, weight = w / sum(w))
  }
  v <- eigen(R_p[it, it], symmetric = TRUE)$vectors[, 1]; v <- v * sign(sum(v))
  rows[[length(rows) + 1]] <- data.table(bank = b, item = it, method = "pearson_pc1", loading = NA_real_,
                                         uniqueness = NA_real_, weight = v / sum(v))
}
## condition banks: correlations recomputed on the rows with a condition inventory
CODES <- list(CC = "COND", CG = c("CVD", "METAB", "RESP", "MSK", "CANCER", "OTHER"))
Xc <- X[!is.na(COND)]
ALLC <- unique(c(ALL, unlist(CODES)))
Dc <- as.data.frame(Xc[, ..ALLC])
Rc_p <- cor(Dc)
t0 <- Sys.time()
Rc_y <- lavCor(Dc, ordered = ALLC, output = "cor")
cat(sprintf("%s condition-sample rows; polychoric matrix over %d testlets in %.0f s\n", format(nrow(Dc), big.mark = ","),
            length(ALLC), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
for (b in setdiff(names(BANKS), "P-FUNC")) for (code in names(CODES)) {
  it <- c(SF, BANKS[[b]], CODES[[code]]); bn <- paste0(b, "+", code)
  for (cm in c("pearson", "polychoric")) {
    Rm <- if (cm == "pearson") Rc_p[it, it] else Rc_y[it, it]
    f <- one_factor(Rm, nrow(Dc)); w <- f$lam / f$psi
    rows[[length(rows) + 1]] <- data.table(bank = bn, item = it, method = paste0(cm, "_factor"),
                                           loading = f$lam, uniqueness = f$psi, weight = w / sum(w))
  }
  v <- eigen(Rc_p[it, it], symmetric = TRUE)$vectors[, 1]; v <- v * sign(sum(v))
  rows[[length(rows) + 1]] <- data.table(bank = bn, item = it, method = "pearson_pc1", loading = NA_real_,
                                         uniqueness = NA_real_, weight = v / sum(v))
}
WT <- rbindlist(rows)
fwrite(WT, file.path(OUT, "limitation_factor_weights.csv"))
fwrite(as.data.table(R_p, keep.rownames = "item"), file.path(OUT, "limitation_cor_pearson.csv"))
fwrite(as.data.table(R_y, keep.rownames = "item"), file.path(OUT, "limitation_cor_polychoric.csv"))
print(dcast(WT, bank + item ~ method, value.var = "weight"), digits = 3)
