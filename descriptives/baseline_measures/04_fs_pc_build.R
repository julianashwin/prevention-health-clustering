## Principal-component and factor-score indices on the GRM banks' items.
SCR <- "/Users/julianashwin/Documents/GitHub/prevention-health-clustering/data/processed/baseline_measures"  # intermediate data, gitignored
source("/Users/julianashwin/Documents/GitHub/prevention-health-clustering/descriptives/baseline_measures/03_fs_pc_items.R")
suppressMessages(library(lavaan))
TAG <- c(`SF-12` = "SF", `SF-12 + limitations` = "SF+F", `SF-12 + limitations + conditions` = "SF+F+C")
polyc <- function(X) lavCor(as.data.frame(X), ordered = names(X), output = "cor")
one_factor <- function(Rm, n) {
  f <- factanal(covmat = Rm, factors = 1, n.obs = n, rotation = "none")
  lam <- f$loadings[, 1]; lam <- lam * sign(sum(lam))
  list(lam = lam, psi = f$uniquenesses)
}
first_pc <- function(Rm) { v <- eigen(Rm, symmetric = TRUE)$vectors[, 1]; v <- v * sign(sum(v)); setNames(v, colnames(Rm)) }
scores <- list(); W <- list()
for (k in names(CONTENT)) {
  items <- CONTENT[[k]]
  ok <- complete.cases(d[, items, with = FALSE]); X <- d[ok, items, with = FALSE]
  Z <- scale(as.matrix(X))
  t0 <- Sys.time()
  R_p <- cor(X); R_poly <- polyc(X)
  mats <- list(pearson = R_p, polychoric = R_poly)
  if (TAG[[k]] == "SF+F+C") {                     # within-age polychoric: 10-year bands, n-weighted average
    ab <- cut(d$age[ok], c(19, 29, 39, 49, 59, 69, 79, 90))
    parts <- lapply(split(seq_len(nrow(X)), ab), \(ix) list(R = polyc(X[ix]), n = length(ix)))
    mats$`polychoric within age` <- Reduce(`+`, lapply(parts, \(p) p$R * p$n)) / sum(sapply(parts, `[[`, "n"))
  }
  cat(sprintf("\n=== %s: %s complete cases; correlations in %.1f s ===\n", k, format(nrow(X), big.mark = ","),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  for (cm in names(mats)) {
    Rm <- mats[[cm]]
    if (cm != "polychoric within age") {
      w <- first_pc(Rm); nm <- paste("PC", cm, TAG[[k]])
      s <- as.numeric(Z %*% w); scores[[nm]] <- data.table(pidp = d$pidp[ok], wave = d$wave[ok], v = (s - mean(s)) / sd(s))
      W[[nm]] <- data.table(index = nm, item = items, weight = w / sum(w))
    }
    f <- one_factor(Rm, nrow(X)); w <- f$lam / f$psi; nm <- paste("FS", cm, TAG[[k]])
    s <- as.numeric(Z %*% w); scores[[nm]] <- data.table(pidp = d$pidp[ok], wave = d$wave[ok], v = (s - mean(s)) / sd(s))
    W[[nm]] <- data.table(index = nm, item = items, weight = w / sum(w), loading = f$lam, uniqueness = f$psi)
    cat(sprintf("  one-factor on %-22s loadings %s | min uniqueness %.3f\n", cm,
                paste(sprintf("%s %.2f", items, f$lam), collapse = ", "), min(f$psi)))
  }
}
WT <- rbindlist(W, fill = TRUE)
cat("\n=== weights, standardised items, summing to one ===\n")
print(dcast(WT, item ~ index, value.var = "weight")[order(match(item, CONTENT[[3]]))], digits = 3)
out <- Reduce(function(a, b) merge(a, b, by = c("pidp", "wave"), all = TRUE),
              lapply(names(scores), \(nm) setnames(copy(scores[[nm]]), "v", nm)))
write_parquet(out, file.path(SCR, "fs_pc_measures.parquet"))
fwrite(WT, file.path(SCR, "fs_pc_weights.csv"))
cat("\nwritten:", nrow(out), "rows,", length(scores), "indices\n")
