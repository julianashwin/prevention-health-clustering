## =============================================================================
## make_framework_figures.R  -  conceptual-framework package
##
## Generates every SIMULATED figure of conceptual_framework.tex, consolidated
## from the original producer scripts (verbatim blocks, unused figures
## dropped; each section carries its source file's full setup, so sections are
## self-contained and run in order). Illustrative parameter values,
## exaggerated for legibility.
## The empirical figures (emp_fig_*.pdf) are produced by the data pipeline,
## which lives in the empirical companion repo; they are included here as
## static files - see README.md.
##
## Run from this folder:  Rscript make_framework_figures.R
## Writes into figures/.
## =============================================================================


###############################################################################
## SECTION: two worlds (Figure 1)
## (from make_framework_figures.R)
###############################################################################
## =============================================================================
## make_framework_figures.R
## Generates the illustrative figures for framework_identification_v2.tex:
##
##   figures/fig_two_worlds.pdf
##     Two populations with the IDENTICAL DISTRIBUTION of health at every age
##     (both exactly Gaussian with the same mean and variance):
##     (a) continuously distributed Gaussian types (H_i, d_i) + iid Gaussian
##         noise, paths coloured by tercile of the decline rate d_i;
##     (b) one mean path + Gaussian random-walk noise with innovation variances
##         matched to (a)'s variance profile.
##     Every cross-sectional statistic coincides; the worlds differ only in
##     their autocovariance structure. (Continuous heterogeneity is what makes
##     the distributional match exact: a few discrete types would produce a
##     multimodal cross-section distinguishable from the Gaussian world.)
##
##   figures/fig_lag_profile.pdf
##     (a) lag profile Cov(dh_t, dh_{t+k}) for the two worlds + an AR(1)-noise
##         world: plateau = V_d, exact zero for the random walk, slow rise under
##         AR(1) (partial identification).
##     (b) recovery of V_d across noise scenarios by four estimators: Prop 2
##         (level moments), its growth-regression twin (identical), the textbook
##         EB shortcut (biased), and the Prop-3 lag-2 moment (robust).
## =============================================================================

set.seed(7)
dir.create("figures", showWarnings = FALSE)
okabe <- c("#0072B2", "#E69F00", "#009E73")

## ---------- calibration for the two-worlds figure ---------------------------
## Continuous type heterogeneity: (H_i, d_i) jointly Gaussian. Both worlds are
## then exactly Gaussian at every age, so matching mean and variance matches
## the entire cross-sectional distribution.
ages <- 0:12; T <- length(ages)
VH   <- 0.43                       # Var(H)
Vd   <- 0.0044                     # Var(d)
Cc   <- 0.6 * sqrt(VH * Vd)        # Cov(H,d): corr 0.6 (compounding)
dbar <- -0.0733                    # mean decline; mean H = 0
sd_iid <- 0.15

draw_Hd <- function(N) {
  L  <- chol(matrix(c(VH, Cc, Cc, Vd), 2, 2))
  Hd <- matrix(rnorm(N * 2), N, 2) %*% L
  cbind(H = Hd[, 1], d = dbar + Hd[, 2])
}
varA <- VH + 2 * ages * Cc + ages^2 * Vd + sd_iid^2   # Var(h_a) in world A
sig2_eta <- diff(varA)                                # matched RW innovations (= 2C + (2a-1)Vd > 0)

simA <- function(N) {  # continuous types + iid noise (exactly Gaussian at each age)
  Hd <- draw_Hd(N)
  h  <- Hd[, 1] %o% rep(1, T) + Hd[, 2] %o% ages +
        matrix(rnorm(N * T, sd = sd_iid), N, T)
  list(h = h, d = Hd[, 2])
}
simB <- function(N) {  # random walk, no types, matched mean & variance profile
  w <- matrix(0, N, T)
  w[, 1] <- rnorm(N, sd = sqrt(varA[1]))
  for (t in 2:T) w[, t] <- w[, t - 1] + rnorm(N, sd = sqrt(sig2_eta[t - 1]))
  rep(1, N) %o% (dbar * ages) + w
}
simC <- function(N, rho = 0.7, sig2_ar = 0.1) {  # continuous types + AR(1) noise
  Hd <- draw_Hd(N)
  u <- matrix(0, N, T)
  u[, 1] <- rnorm(N, sd = sqrt(sig2_ar))
  for (t in 2:T) u[, t] <- rho * u[, t - 1] + rnorm(N, sd = sqrt(sig2_ar * (1 - rho^2)))
  Hd[, 1] %o% rep(1, T) + Hd[, 2] %o% ages + u
}

## sanity check: the two variance profiles match (means match by construction;
## both worlds Gaussian, so matched mean + variance = identical distributions)
Nchk <- 4e5
cat("max |Var_A(h_a) - Var_B(h_a)| (empirical, N = 4e5): ",
    max(abs(apply(simA(Nchk)$h, 2, var) - apply(simB(Nchk), 2, var))), "\n")

## ---------- Figure 1: two worlds --------------------------------------------
Nshow <- 45
A <- simA(Nshow); B <- simB(Nshow)
tercA <- cut(A$d, quantile(A$d, c(0, 1/3, 2/3, 1)), labels = FALSE, include.lowest = TRUE)
## bold lines: tercile-mean trajectories (terciles of d_i), from a large draw
big <- draw_Hd(2e5)
tb  <- cut(big[, 2], quantile(big[, 2], c(0, 1/3, 2/3, 1)), labels = FALSE, include.lowest = TRUE)
Hm  <- tapply(big[, 1], tb, mean); dm <- tapply(big[, 2], tb, mean)
colmap <- c(3, 2, 1)  # fastest decliners green (bottom), slowest blue (top)
ylim <- range(A$h, B)

pdf("figures/fig_two_worlds.pdf", width = 10, height = 4.2, pointsize = 13)
par(mfrow = c(1, 2), mar = c(3.6, 3.6, 2.2, 0.8), mgp = c(2.2, 0.7, 0))
plot(NULL, xlim = range(ages), ylim = ylim, xlab = "age (wave)", ylab = "health, h",
     main = "(a) Deterministic divergence: types")
for (i in 1:Nshow)
  lines(ages, A$h[i, ], col = adjustcolor(okabe[colmap[tercA[i]]], 0.25), lwd = 0.8)
for (j in 1:3) lines(ages, Hm[j] + dm[j] * ages, col = okabe[colmap[j]], lwd = 3.5)
plot(NULL, xlim = range(ages), ylim = ylim, xlab = "age (wave)", ylab = "health, h",
     main = "(b) Stochastic accumulation: random walk")
for (i in 1:Nshow) lines(ages, B[i, ], col = adjustcolor("grey35", 0.3), lwd = 0.8)
lines(ages, dbar * ages, col = "black", lwd = 3.5)
dev.off()

###############################################################################
## SECTION: case anatomy charts (Cases 1-4, K(r) blocks)
## (from make_case_anatomy.R)
###############################################################################
## =============================================================================
## make_case_anatomy.R
##
## Anatomy charts for the case ladder of framework_identification_v5.tex.
## Each case gets ONE figure with TWO panels - the two curves the proposition
## reads:
##   left  panel: the variance profile V(a) = Var(h_a),  a = 0,1,2
##   right panel: the first row       C0(a) = Cov(h_0,h_a),  anchored at V(0)
##
##   fig_case1_curves.pdf - Case 1: iid, stationary s2_u
##   fig_case2_curves.pdf - Case 2: AR(1), stationary s2_u
##   fig_case3_curves.pdf - Case 3: AR(1), age-specific v_a
##                          (right panel adds the interior cells that give V_d)
##   fig_case4_curves.pdf - Case 4: AR(1) + one-period measurement error
##                          (the spike lives only on the diagonal; the fade is
##                          read from off-diagonal drops only)
##   fig_K_bars.pdf       - the prevention combination K(s) assembled from
##                          identified blocks (not a raw moment)
##
## The earlier six-slot charts are generated by make_case_anatomy_sixslot.R and
## remain on disk. Illustrative values, exaggerated for legibility.
## =============================================================================

dir.create("figures", showWarnings = FALSE)

cols <- c(VH = "#0072B2", Vd = "#009E73", CHd = "#E69F00", s2u = "grey72",
          s2w = "grey72", s2e = "grey45")
txt_col <- c(VH = "white", Vd = "white", CHd = "white", s2u = "grey25",
             s2w = "grey25", s2e = "white")

cstack <- function(x, segs, w = 0.38, cex = 0.78) {
  y0 <- 0
  for (sg in segs) {
    nm <- sg[[1]]; v <- sg[[2]]
    rect(x - w, y0, x + w, y0 + v, col = cols[nm], border = "white", lwd = 1)
    text(x, y0 + v / 2, sg[[3]], col = txt_col[nm], cex = cex)
    y0 <- y0 + v
  }
  invisible(y0)
}
key_block <- function(entries, ex, ey, dy = 0.19, sq = 0.05, cex = 0.78) {
  for (k in seq_along(entries)) {
    yy <- ey - (k - 1) * dy
    rect(ex, yy - sq, ex + 2 * sq, yy + sq, col = cols[entries[[k]][[1]]], border = NA)
    text(ex + 2.6 * sq, yy, entries[[k]][[2]], adj = 0, cex = cex)
  }
}

## shared illustrative parameters
VH <- 1.0; C2 <- 0.22; Vd2 <- 0.18; s2 <- 0.5; rho <- 0.6
v3 <- c(0.50, 0.65, 0.40, 0.55)                 # Case 3: v_0..v_3

x3 <- c(1, 2.2, 3.4)                            # three-bar panel positions

## variance-profile panel (shared by Cases 1-2; Case 3 passes its own hats)
panel_V <- function(hats, hat_labs, ylim_top, curve = TRUE, note = NULL) {
  plot(NULL, xlim = c(0.45, 4.0), ylim = c(0, ylim_top), xaxt = "n", xlab = "",
       ylab = "population moment", bty = "n", xaxs = "i", yaxs = "i")
  segs <- list(
    list(list("VH", VH, expression(V[H]))),
    list(list("VH", VH, expression(V[H])),
         list("CHd", 2 * C2, expression(2 * C[Hd])),
         list("Vd", Vd2, expression(V[d]))),
    list(list("VH", VH, expression(V[H])),
         list("CHd", 4 * C2, expression(4 * C[Hd])),
         list("Vd", 4 * Vd2, expression(4 * V[d])))
  )
  for (k in 1:3)
    cstack(x3[k], c(segs[[k]], list(list("s2u", hats[k], hat_labs[[k]]))))
  if (curve) {
    zz <- seq(0, 2, 0.02)
    lines(x3[1] + zz * (x3[3] - x3[1]) / 2, VH + s2 + 2 * C2 * zz + Vd2 * zz^2,
          lty = 2, lwd = 1.8, col = "grey15")
  }
  if (!is.null(note)) text(2.2, ylim_top - 0.35, note, cex = 0.74, col = "grey15")
  axis(1, at = x3, tick = FALSE, line = -0.6, cex.axis = 0.9,
       labels = expression(V(0), V(1), V(2)))
}

## first-row panel: bars V(0), C0(1), ..., C0(amax), with the line V_H + a C_Hd
panel_row <- function(hats, hat_labs, amax, ylim_top, xright = 4.0) {
  xs <- 1 + 1.2 * (0:amax)
  plot(NULL, xlim = c(0.45, xright), ylim = c(0, ylim_top), xaxt = "n", xlab = "",
       ylab = "", bty = "n", xaxs = "i", yaxs = "i")
  for (a in 0:amax) {
    segs <- list(list("VH", VH, expression(V[H])))
    if (a > 0) segs <- c(segs, list(list("CHd", a * C2,
      if (a == 1) expression(C[Hd]) else parse(text = sprintf("%d*C[Hd]", a)))))
    if (hats[a + 1] > 0) segs <- c(segs, list(list("s2u", hats[a + 1],
                                                   hat_labs[[a + 1]])))
    cstack(xs[a + 1], segs)
  }
  lines(c(xs[1], xs[amax + 1] + 0.5), VH + C2 * c(0, amax + 0.5 / 1.2),
        lty = 2, lwd = 1.8, col = "grey15")
  axis(1, at = xs, tick = FALSE, line = -0.6, cex.axis = 0.9,
       labels = parse(text = c("V(0)", sprintf("C[0](%d)", 1:amax))))
  invisible(xs)
}

## ---- Case 1 -----------------------------------------------------------------
pdf("figures/fig_case1_curves.pdf", width = 10, height = 4.9, pointsize = 13)
layout(matrix(1:2, 1, 2), widths = c(1, 1))
par(mar = c(3.6, 3.6, 3.6, 0.5), mgp = c(2.3, 0.7, 0))
panel_V(hats = rep(s2, 3),
        hat_labs = rep(list(expression(sigma[u]^2)), 3), ylim_top = 4.4)
key_block(list(
  list("Vd",  expression(V[d] == 0.5 * (V(2) - 2 * V(1) + V(0)))),
  list("CHd", expression(C[Hd] == 0.5 * (V(1) - V(0) - V[d]))),
  list("s2u", expression("grey hat constant: differences remove it"))),
  ex = 0.55, ey = 4.2)
title("variance profile: curvature and slope", cex.main = 0.95, line = 0.6)
mtext(expression(paste(bold("Case 1 (i.i.d., stationary): "),
                       "the profile gives ", C[Hd], " and ", V[d],
                       "; one row cell splits ", V[H], " from ", sigma[u]^2)),
      side = 3, line = 2.0, cex = 0.95, adj = 0)
par(mar = c(3.6, 2.2, 3.6, 0.5))
panel_row(hats = c(s2, 0, 0),
          hat_labs = list(expression(sigma[u]^2), NULL, NULL),
          amax = 2, ylim_top = 4.4)
key_block(list(
  list("VH",  expression(V[H] == C[0](1) - C[Hd])),
  list("s2u", expression(sigma[u]^2 == V(0) - V[H])),
  list("CHd", expression(paste("row on the line ", V[H] + a * C[Hd],
                " from ", a == 1)))),
  ex = 0.55, ey = 4.2)
title("first row: the level split", cex.main = 0.95, line = 0.6)
dev.off()

## ---- Case 2 -----------------------------------------------------------------
pdf("figures/fig_case2_curves.pdf", width = 10, height = 4.9, pointsize = 13)
layout(matrix(1:2, 1, 2), widths = c(1, 1))
par(mar = c(3.6, 3.6, 3.6, 0.5), mgp = c(2.3, 0.7, 0))
panel_V(hats = rep(s2, 3),
        hat_labs = rep(list(expression(sigma[u]^2)), 3), ylim_top = 4.4)
key_block(list(
  list("Vd",  expression(V[d] == 0.5 * (V(2) - 2 * V(1) + V(0)))),
  list("CHd", expression(C[Hd] == 0.5 * (V(1) - V(0) - V[d]))),
  list("s2u", expression("same as Case 1: memory does not enter " * V(a)))),
  ex = 0.55, ey = 4.2)
title("variance profile: unchanged", cex.main = 0.95, line = 0.6)
mtext(expression(paste(bold("Case 2 (AR(1), stationary): "),
                       "the profile reads are unchanged; the row decays at ",
                       rho, " per step")),
      side = 3, line = 2.0, cex = 0.95, adj = 0)
par(mar = c(3.6, 2.2, 3.6, 0.5))
panel_row(hats = c(s2, rho * s2, rho^2 * s2),
          hat_labs = list(expression(sigma[u]^2), expression(rho * sigma[u]^2),
                          expression(rho^2 * sigma[u]^2)),
          amax = 2, ylim_top = 4.4)
key_block(list(
  list("s2u", expression(paste(L[a] == C[0](a) - a * C[Hd],
                "  =  ", V[H] + rho^a * sigma[u]^2))),
  list("s2u", expression(rho == (L[1] - L[2]) / (L[0] - L[1]))),
  list("VH",  expression(paste(sigma[u]^2 == (L[0] - L[1]) / (1 - rho),
                ",   ", V[H] == L[0] - sigma[u]^2)))),
  ex = 0.55, ey = 4.2)
title("first row: the fade gives the split", cex.main = 0.95, line = 0.6)
dev.off()

## ---- Case 3 -----------------------------------------------------------------
pdf("figures/fig_case3_curves.pdf", width = 10.6, height = 4.9, pointsize = 13)
layout(matrix(1:2, 1, 2), widths = c(0.40, 0.60))
par(mar = c(3.6, 3.6, 3.6, 0.5), mgp = c(2.3, 0.7, 0))
panel_V(hats = v3[1:3],
        hat_labs = list(expression(v[0]), expression(v[1]), expression(v[2])),
        ylim_top = 4.4, curve = FALSE,
        note = expression("free " * v[a] * ": the profile alone gives nothing"))
title("variance profile: not readable alone", cex.main = 0.95, line = 0.6)
mtext(expression(paste(bold("Case 3 (AR(1), age-specific "), bold(v[a]),
                       bold("): "), "the row gives ", (list(rho, v[0], V[H], C[Hd])),
                       "; interior cells give ", V[d])),
      side = 3, line = 2.0, cex = 0.95, adj = 0)
par(mar = c(3.6, 2.2, 3.6, 0.5))
xs <- panel_row(hats = c(v3[1], rho * v3[1], rho^2 * v3[1], rho^3 * v3[1]),
                hat_labs = list(expression(v[0]), expression(rho * v[0]),
                                expression(rho^2 * v[0]), expression(rho^3 * v[0])),
                amax = 3, ylim_top = 4.4, xright = 8.4)
## interior cells, set apart: these deliver V_d
xi <- c(6.3, 7.5)
abline(v = 5.6, lty = 3, col = "grey60")
cstack(xi[1], list(list("VH", VH, expression(V[H])),
                   list("CHd", 3 * C2, expression(3 * C[Hd])),
                   list("Vd", 2 * Vd2, expression(2 * V[d])),
                   list("s2u", rho * v3[2], expression(rho * v[1]))))
cstack(xi[2], list(list("VH", VH, expression(V[H])),
                   list("CHd", 4 * C2, expression(4 * C[Hd])),
                   list("Vd", 3 * Vd2, expression(3 * V[d])),
                   list("s2u", rho^2 * v3[2], expression(rho^2 * v[1]))))
axis(1, at = xi, tick = FALSE, line = -0.6, cex.axis = 0.9,
     labels = expression(C(1, 2), C(1, 3)))
text(mean(xi), 2.95, expression("interior: " * V[d] * " and " * v[1]),
     cex = 0.78, col = "grey15")
key_block(list(
  list("s2u", expression(paste(S[1] == C[0](2) - 2 * C[0](1) + V(0),
                ",   ", S[2] == C[0](3) - 2 * C[0](2) + C[0](1)))),
  list("s2u", expression(paste(rho == S[2] / S[1],
                ",   ", v[0] == S[1] / (1 - rho)^2))),
  list("VH",  expression(paste(V[H] == V(0) - v[0],
                ",   ", C[Hd] == C[0](1) - V(0) + (1 - rho) * v[0])))),
  ex = 0.55, ey = 4.2)
title("first row (four cells), then two interior cells", cex.main = 0.95, line = 0.6)
dev.off()

## ---- K(s) assembled from identified blocks ----------------------------------
## K(s) = A(s) C_Hd + B(s) V_d with A(s) = sum (a-s+1), B(s) = sum a(a-s+1);
## here abar = 2: K(1) = 3 C_Hd + 5 V_d, K(2) = C_Hd + 2 V_d.
pdf("figures/fig_K_bars.pdf", width = 6.2, height = 4.4, pointsize = 13)
par(mar = c(3.2, 3.6, 3.6, 0.6), mgp = c(2.3, 0.7, 0))
plot(NULL, xlim = c(0.5, 3.1), ylim = c(0, 2.15), xaxt = "n", xlab = "",
     ylab = "return units", bty = "n", xaxs = "i", yaxs = "i")
cstack(1.1, list(list("CHd", 3 * C2, expression(3 * C[Hd])),
                 list("Vd", 5 * Vd2, expression(5 * V[d]))), w = 0.32)
cstack(2.3, list(list("CHd", C2, expression(C[Hd])),
                 list("Vd", 2 * Vd2, expression(2 * V[d]))), w = 0.32)
axis(1, at = c(1.1, 2.3), tick = FALSE, line = -0.6,
     labels = expression(K(1), K(2)))
key_block(list(
  list("CHd", expression(A(s) * C[Hd])),
  list("Vd",  expression(B(s) * V[d]))),
  ex = 2.62, ey = 2.0, dy = 0.17, cex = 0.85)
title(expression("The prevention combination " * K(s) * ", built from identified blocks"),
      cex.main = 0.95, line = 2.0)
mtext(expression(paste(K(s) == A(s) * C[Hd] + B(s) * V[d],
                       "  (here " * bar(a) == 2 * "); not a raw moment")),
      side = 3, line = 0.7, cex = 0.8, col = "grey35")
mtext(expression(paste("a later start has a larger ", V[d], " share (Result 5(iii))")),
      side = 1, line = 1.9, cex = 0.8, col = "grey35")
dev.off()

cat("written: figures/fig_case{1,2,3}_curves.pdf, fig_K_bars.pdf\n")


## ---- Case 4 -----------------------------------------------------------------
## AR(1) persistent part (s2w, fading by rho along the row) + one-period
## measurement error (s2e, on the diagonal only). s2w + s2e = the s2 of
## Cases 1-2, so the totals match across the figures.
s2w <- 0.32; s2e <- 0.18
pdf("figures/fig_case4_curves.pdf", width = 10, height = 4.9, pointsize = 13)
layout(matrix(1:2, 1, 2), widths = c(1, 1))
par(mar = c(3.6, 3.6, 3.6, 0.5), mgp = c(2.3, 0.7, 0))
plot(NULL, xlim = c(0.45, 4.0), ylim = c(0, 4.4), xaxt = "n", xlab = "",
     ylab = "population moment", bty = "n", xaxs = "i", yaxs = "i")
segsV <- list(
  list(list("VH", VH, expression(V[H]))),
  list(list("VH", VH, expression(V[H])),
       list("CHd", 2 * C2, expression(2 * C[Hd])),
       list("Vd", Vd2, expression(V[d]))),
  list(list("VH", VH, expression(V[H])),
       list("CHd", 4 * C2, expression(4 * C[Hd])),
       list("Vd", 4 * Vd2, expression(4 * V[d]))))
for (k in 1:3)
  cstack(x3[k], c(segsV[[k]],
                  list(list("s2w", s2w, expression(sigma[w]^2)),
                       list("s2e", s2e, expression(sigma[epsilon]^2)))))
zz <- seq(0, 2, 0.02)
lines(x3[1] + zz * (x3[3] - x3[1]) / 2,
      VH + s2w + s2e + 2 * C2 * zz + Vd2 * zz^2,
      lty = 2, lwd = 1.8, col = "grey15")
axis(1, at = x3, tick = FALSE, line = -0.6, cex.axis = 0.9,
     labels = expression(V(0), V(1), V(2)))
key_block(list(
  list("Vd",  expression(V[d] == 0.5 * (V(2) - 2 * V(1) + V(0)))),
  list("CHd", expression(C[Hd] == 0.5 * (V(1) - V(0) - V[d]))),
  list("s2w", expression("both hats constant: differences remove them"))),
  ex = 0.55, ey = 4.2)
title("variance profile: unchanged", cex.main = 0.95, line = 0.6)
mtext(expression(paste(bold("Case 4 (AR(1) + measurement error): "),
      "the diagonal reads survive; the row splits ", V[H], ", ",
      sigma[w]^2, " and ", sigma[epsilon]^2)),
      side = 3, line = 2.0, cex = 0.95, adj = 0)
par(mar = c(3.6, 2.2, 3.6, 0.5))
amax <- 3; xs <- 1 + 1.05 * (0:amax)
plot(NULL, xlim = c(0.45, 5.1), ylim = c(0, 4.4), xaxt = "n", xlab = "",
     ylab = "", bty = "n", xaxs = "i", yaxs = "i")
fade_lab <- list(expression(sigma[w]^2), expression(rho * sigma[w]^2),
                 expression(rho^2 * sigma[w]^2), expression(rho^3 * sigma[w]^2))
for (a in 0:amax) {
  segs <- list(list("VH", VH, expression(V[H])))
  if (a > 0) segs <- c(segs, list(list("CHd", a * C2,
    if (a == 1) expression(C[Hd]) else parse(text = sprintf("%d*C[Hd]", a)))))
  segs <- c(segs, list(list("s2w", rho^a * s2w, fade_lab[[a + 1]])))
  if (a == 0) segs <- c(segs, list(list("s2e", s2e, expression(sigma[epsilon]^2))))
  cstack(xs[a + 1], segs)
}
lines(c(xs[1], xs[amax + 1] + 0.45), VH + C2 * c(0, amax + 0.45 / 1.05),
      lty = 2, lwd = 1.8, col = "grey15")
axis(1, at = xs, tick = FALSE, line = -0.6, cex.axis = 0.9,
     labels = parse(text = c("V(0)", sprintf("C[0](%d)", 1:amax))))
key_block(list(
  list("s2w", expression(rho == (tilde(C)[0](2) - tilde(C)[0](3)) /
                                 (tilde(C)[0](1) - tilde(C)[0](2)))),
  list("VH",  expression(V[H] == tilde(C)[0](1) - rho * sigma[w]^2)),
  list("s2e", expression(sigma[epsilon]^2 == V(0) - V[H] - sigma[w]^2))),
  ex = 0.55, ey = 4.2)
title("first row: fade among off-diagonal cells only", cex.main = 0.95, line = 0.6)
dev.off()

###############################################################################
## SECTION: denoising anatomy (appendix)
## (from make_case_anatomy_sixslot.R)
###############################################################################
## =============================================================================
## make_case_anatomy.R
##
## Anatomy charts for the case ladder of framework_identification_v4.tex:
##
##   fig_case0_anatomy.pdf   - Case 0: iid, C_Hd = 0        (2 waves)
##   fig_case0_anatomy_full.pdf - Case 0 on the full six-slot skeleton
##                             (mirrors Cases 1-2; profile line, faded checks)
##   fig_case1_anatomy.pdf   - Case 1: iid, C_Hd free       (3 waves, variance-first)
##   fig_case2_anatomy.pdf   - Case 2: stationary AR(1)     (3 waves, variance-first)
##   fig_denoise_anatomy.pdf - appendix: independent, age-specific s2_a
##                             (3 waves, Var* denoising)
##
## Shared skeleton: moments in matrix order along the x-axis with a
## distance-from-diagonal sub-axis; faded bars = overidentifying checks not
## needed for identification. Illustrative values, exaggerated for legibility.
## =============================================================================

dir.create("figures", showWarnings = FALSE)

cols <- c(VH = "#0072B2", Vd = "#009E73", CHd = "#E69F00", s2u = "grey72")
txt_col <- c(VH = "white", Vd = "white", CHd = "white", s2u = "grey25")

cstack <- function(x, segs, w = 0.38, cex = 0.78, faint = FALSE) {
  y0 <- 0
  for (sg in segs) {
    nm <- sg[[1]]; v <- sg[[2]]
    fill <- if (faint) adjustcolor(cols[nm], 0.30) else cols[nm]
    bord <- if (faint) "grey88" else "white"
    tcol <- if (faint) "grey62" else txt_col[nm]
    rect(x - w, y0, x + w, y0 + v, col = fill, border = bord, lwd = 1)
    text(x, y0 + v / 2, sg[[3]], col = tcol, cex = cex)
    y0 <- y0 + v
  }
  if (faint) text(x, y0 + 0.10, "(check)", cex = 0.62, col = "grey55")
  invisible(y0)
}
float_bar <- function(x, y0, y1, nm, lab, w = 0.15, cex = 0.8) {
  rect(x - w, y0, x + w, y1, col = cols[nm], border = "white", lwd = 1)
  text(x, (y0 + y1) / 2, lab, col = txt_col[nm], cex = cex, font = 2)
}
guide <- function(y, x0, x1) segments(x0, y, x1, y, lty = 3, col = "grey45")
key_block <- function(entries, ex, ey, dy = 0.19, sq = 0.05, cex = 0.82) {
  for (k in seq_along(entries)) {
    yy <- ey - (k - 1) * dy
    rect(ex, yy - sq, ex + 2 * sq, yy + sq, col = cols[entries[[k]][[1]]], border = NA)
    text(ex + 2.6 * sq, yy, entries[[k]][[2]], adj = 0, cex = cex)
  }
}

## shared illustrative parameters
VH <- 1.0; C2 <- 0.22; Vd2 <- 0.18; s2 <- 0.5
s2a <- c(0.45, 0.55, 0.40); rho <- 0.6
xx <- c(1, 2.05, 3.1, 4.15, 5.2, 6.55)
lab6 <- expression(Var(h[0]), Cov(h[0], h[1]), Var(h[1]),
                   Cov(h[1], h[2]), Var(h[2]), Cov(h[0], h[2]))
subaxis <- function(at = xx, d = c("0", "1", "0", "1", "0", "2")) {
  mtext(d, side = 1, line = 1.4, at = at, cex = 0.72, col = "grey35")
  mtext(expression(t - s * " :"), side = 1, line = 1.4, at = 0.45, adj = 0,
        cex = 0.72, col = "grey35")
}

## ---- Case 0: iid, C_Hd = 0  (2 waves) ---------------------------------------

## ---- Denoising (appendix): independent, age-specific s2_a (Var* device) -----
sm <- function(s, t) VH + (s + t) * C2 + s * t * Vd2

pdf("figures/fig_denoise_anatomy.pdf", width = 9, height = 5.5, pointsize = 13)
par(mar = c(5.2, 3.6, 3.4, 0.6), mgp = c(2.3, 0.7, 0))
plot(NULL, xlim = c(0.45, 7.15), ylim = c(0, 4.1), xaxt = "n", xlab = "",
     ylab = "population moment", bty = "n", xaxs = "i", yaxs = "i")
cstack(xx[1], list(list("VH", VH, expression(V[H])),
                   list("s2u", s2a[1], expression(v[0]))))
cstack(xx[2], list(list("VH", VH, expression(V[H])),
                   list("CHd", C2, expression(C[Hd]))))
cstack(xx[3], list(list("VH", VH, expression(V[H])),
                   list("CHd", 2 * C2, expression(2 * C[Hd])),
                   list("Vd", Vd2, expression(V[d])),
                   list("s2u", s2a[2], expression(v[1]))))
cstack(xx[4], list(list("VH", VH, expression(V[H])),
                   list("CHd", 3 * C2, expression(3 * C[Hd])),
                   list("Vd", 2 * Vd2, expression(2 * V[d]))))
cstack(xx[5], list(list("VH", VH, expression(V[H])),
                   list("CHd", 4 * C2, expression(4 * C[Hd])),
                   list("Vd", 4 * Vd2, expression(4 * V[d])),
                   list("s2u", s2a[3], expression(v[2]))))
cstack(xx[6], list(list("VH", VH, expression(V[H])),
                   list("CHd", 2 * C2, expression(2 * C[Hd]))))
segments(xx[2], sm(0, 1), xx[4], sm(1, 2), lty = 2, lwd = 1.8, col = "grey15")
axis(1, at = xx, tick = FALSE, line = -0.6, cex.axis = 0.82, labels = lab6)
subaxis()
key_block(list(
  list("s2u", expression(paste(Var^{"*"}, (h[a]) == 0.5 *
                (Cov(h[a - 1], h[a]) + Cov(h[a], h[a + 1])),
                "   (edges by extrapolation)"))),
  list("s2u", expression(v[a] == Var(h[a]) - Var^{"*"} * (h[a]))),
  list("Vd",  expression(V[d] == 0.5 * (Var^{"*"} * (h[2]) - 2 * Var^{"*"} * (h[1])
                + Var^{"*"} * (h[0])))),
  list("CHd", expression(C[Hd] == 0.5 * (Var^{"*"} * (h[1]) - Var^{"*"} * (h[0]) - V[d]))),
  list("VH",  expression(V[H] == Var^{"*"} * (h[0])))),
  ex = 0.62, ey = 3.92)
title(expression(paste("Age-varying noise (", v[a],
                       " free): denoise the variances, then read as Case 1")),
      cex.main = 1.0, line = 2.1)
mtext("three waves; the noise-free variance profile is rebuilt from the neighbouring covariances",
      side = 3, line = 0.7, cex = 0.85, col = "grey35")
mtext(expression(paste("the base of each grey hat is ", Var^{"*"} * (h[a]),
                       ": dashed = neighbour interpolation at the interior age")),
      side = 1, line = 2.9, cex = 0.76, col = "grey35")
dev.off()

###############################################################################
## SECTION: random-walk anatomy in levels (appendix)
## (from make_identification_anatomy.R)
###############################################################################
## =============================================================================
## make_identification_anatomy.R
##
## "Anatomy" charts for the identification propositions:
##   - observable moments on the x-axis, drawn as stacked bars showing their
##     composition in model components;
##   - identified parameters drawn as FLOATING bars bridging the gaps between
##     adjacent moments (each identification formula = one visible step).
##
## Visual language (fixed across all charts):
##   V_H blue | V_d green | C_Hd orange | sigma^2_u / sigma^2_eps grey |
##   sigma^2_eta pink
##
## Charts:
##   fig_prop1_anatomy.pdf  - legacy Case 0 chart (superseded by
##                            make_case_anatomy.R, kept for reference)
##   fig_prop3_anatomy.pdf  - difference moments under random walk + iid noise
##   fig_prop3_anatomy_levels.pdf - random-walk case in level covariances
##   fig_variance_bands.pdf - (a) Var(h_a) decomposed into bands over age
##   fig_cov_heatmap.pdf    - (b) the covariance surface as an annotated matrix
## Values are illustrative population identities (no simulation), exaggerated
## for legibility.
## =============================================================================

dir.create("figures", showWarnings = FALSE)

cols <- c(VH = "#0072B2", B = "#0072B2", Vd = "#009E73", CHd = "#E69F00",
          s2u = "grey72", s2eps = "grey72", s2eta = "#CC79A7")
lab_expr <- list(VH  = expression(V[H]),
                 B   = expression(V[H] + Var(w[0])),
                 Vd  = expression(V[d]),
                 CHd = expression(C[Hd]),
                 s2u = expression(sigma[u]^2),
                 s2eps = expression(sigma[epsilon]^2),
                 s2eta = expression(sigma[eta]^2))
txt_col <- c(VH = "white", B = "white", Vd = "white", CHd = "white",
             s2u = "grey25", s2eps = "grey25", s2eta = "white")

## ---- template helpers -------------------------------------------------------
stack_bar <- function(x, segs, w = 0.38, cex = 0.95) {  # solid moment bar
  y0 <- 0
  for (k in seq_along(segs)) {                          # index loop: names may repeat
    nm <- names(segs)[k]
    rect(x - w, y0, x + w, y0 + segs[[k]], col = cols[nm], border = "white", lwd = 1)
    text(x, y0 + segs[[k]] / 2, lab_expr[[nm]], col = txt_col[nm], cex = cex)
    y0 <- y0 + segs[[k]]
  }
  invisible(y0)
}
float_bar <- function(x, y0, y1, nm, w = 0.15, lab = lab_expr[[nm]], cex = 0.95) {
  rect(x - w, y0, x + w, y1, col = cols[nm], border = "white", lwd = 1)
  text(x, (y0 + y1) / 2, lab, col = txt_col[nm], cex = cex, font = 2)
}
guide <- function(y, x0, x1) segments(x0, y, x1, y, lty = 3, col = "grey45")
key_block <- function(entries, ex, ey, dy = 0.16, sq = 0.05, cex = 0.9) {
  for (k in seq_along(entries)) {
    yy <- ey - (k - 1) * dy
    rect(ex, yy - sq, ex + 2 * sq, yy + sq, col = cols[entries[[k]][[1]]], border = NA)
    text(ex + 2.6 * sq, yy, entries[[k]][[2]], adj = 0, cex = cex)
  }
}

## =============================================================================
## Proposition 1: two waves  (C_Hd = 0, iid noise, stationary variance)
## =============================================================================
VH <- 1.0; s2 <- 0.5; Vd <- 0.4
cov01 <- VH; v0 <- VH + s2; v1 <- VH + s2 + Vd
xC <- 1; x0 <- 2.3; x1 <- 3.6; w <- 0.38


## =============================================================================
## Proposition 3 in LEVEL covariances: the surface gains a staircase
## Cov(h_s,h_t) = (V_H + omega^2) + (s+t) C_Hd + st V_d + G(min(s,t)),  s < t,
## with G(m) = m sigma^2_eta (constant innovation variances for display).
## Bars grouped by ROW (earlier wave s): within-row slopes never touch the
## staircase, so C_Hd and V_d read exactly as in Prop 2; the staircase (pink)
## appears only when the earlier wave advances.
## =============================================================================
Bl <- 0.9; Cl <- 0.2; Vdl <- 0.16; snl <- 0.26
c01 <- Bl + Cl; c02 <- Bl + 2 * Cl
c12 <- Bl + 3 * Cl + 2 * Vdl + snl; c13 <- Bl + 4 * Cl + 3 * Vdl + snl
xs <- c(1, 2.1, 3.5, 4.6); w4 <- 0.34; fw <- 0.15

pdf("figures/fig_prop3_anatomy_levels.pdf", width = 8.5, height = 5.4, pointsize = 13)
par(mar = c(4.8, 3.6, 3.4, 0.6), mgp = c(2.3, 0.7, 0))
plot(NULL, xlim = c(0.45, 5.25), ylim = c(0, 3.15), xaxt = "n", xlab = "",
     ylab = "population moment", bty = "n", xaxs = "i", yaxs = "i")
guide(c01, xs[1] + w4, 1.55 - fw); guide(c02, 1.55 + fw, xs[2] - w4)
guide(c02, xs[2] + w4, 2.8 - fw);  guide(c12, 2.8 + fw, xs[3] - w4)
guide(c12, xs[3] + w4, 4.05 - fw); guide(c13, 4.05 + fw, xs[4] - w4)
stack_bar(xs[1], c(B = Bl, CHd = Cl), w = w4, cex = 0.75)
stack_bar(xs[2], c(B = Bl, CHd = Cl, CHd = Cl), w = w4, cex = 0.75)
stack_bar(xs[3], c(B = Bl, CHd = Cl, CHd = Cl, CHd = Cl, Vd = Vdl, Vd = Vdl,
                   s2eta = snl), w = w4, cex = 0.75)
stack_bar(xs[4], c(B = Bl, CHd = Cl, CHd = Cl, CHd = Cl, CHd = Cl, Vd = Vdl,
                   Vd = Vdl, Vd = Vdl, s2eta = snl), w = w4, cex = 0.75)
float_bar(1.55, c01, c02, "CHd", cex = 0.75)                        # row-0 slope
float_bar(2.8, c02, c02 + Cl, "CHd", cex = 0.75)                    # between rows:
float_bar(2.8, c02 + Cl, c02 + Cl + 2 * Vdl, "Vd",
          lab = expression(2 * V[d]), cex = 0.75)                   #   quadratic part
float_bar(2.8, c02 + Cl + 2 * Vdl, c12, "s2eta", cex = 0.75)        #   + the staircase
float_bar(4.05, c12, c12 + Cl, "CHd", cex = 0.75)                   # row-1 slope:
float_bar(4.05, c12 + Cl, c13, "Vd", cex = 0.75)                    #   steeper by V_d
axis(1, at = xs, tick = FALSE, line = -0.6, cex.axis = 0.9,
     labels = expression(Cov(h[0], h[1]), Cov(h[0], h[2]),
                         Cov(h[1], h[2]), Cov(h[1], h[3])))
mtext("row s = 0  (staircase-free)", side = 1, line = 1.5, at = 1.55,
      cex = 0.8, col = "grey35")
mtext("row s = 1", side = 1, line = 1.5, at = 4.05, cex = 0.8, col = "grey35")
key_block(list(
  list("B",     expression(V[H] + Var(w[0]) == 2 * Cov(h[0], h[1]) - Cov(h[0], h[2]))),
  list("CHd",   expression(C[Hd] == Cov(h[0], h[2]) - Cov(h[0], h[1]))),
  list("Vd",    expression(V[d] == group("[", Cov(h[1], h[3]) - Cov(h[1], h[2]), "]") - C[Hd])),
  list("s2eta", expression(sigma[eta]^2 == group("[", Cov(h[1], h[2]) - Cov(h[0], h[2]), "]")
                           - C[Hd] - 2 * V[d]))),
  ex = 0.6, ey = 3.0, dy = 0.18, cex = 0.85)
title("Anatomy of the random-walk case: level covariances gain a staircase",
      cex.main = 1.0, line = 2.1)
mtext(expression(paste(Cov(h[s], h[t]) == V[H] + Var(w[0]) + (s + t) * C[Hd] +
                       s * t * V[d], " + min(s,t) ", sigma[eta]^2,
                       "   (constant ", sigma[eta]^2, " for display)")),
      side = 3, line = 0.7, cex = 0.82, col = "grey35")
mtext("within-row slopes never touch the staircase: row 0 reads the i.i.d. off-diagonal formulas verbatim;",
      side = 1, line = 2.9, cex = 0.78, col = "grey35")
mtext(expression(paste("row 1's steeper slope reveals ", V[d],
                       "; diagonals (not shown) sit ", sigma[epsilon][","][a]^2,
                       " above the off-diagonal formula")),
      side = 1, line = 3.8, cex = 0.78, col = "grey35")
dev.off()

###############################################################################
## SECTION: surface triangles (appendix)
## (from make_heatmap_figures.R)
###############################################################################
## =============================================================================
## make_heatmap_figures.R
##
## Triangular-heatmap figures for framework_identification_v5.tex (replacing
## the 3d persp() charts wherever a covariance matrix is shown):
##
##   figures/fig_surface_layers_tri.pdf
##     The covariance matrix built layer by layer (level, tilt, curvature,
##     staircase, spike, sum), each layer as a lower-storage triangular heatmap
##     coloured by the parameter that generates it (appendix, general surface).
##
##   figures/fig_surface_menu_tri.pdf
##     The general template as a menu: smooth layers for richer deterministic
##     bases (top row) and noise layers for different processes (bottom row).
##
##   figures/fig_case_reads.pdf
##     One annotated triangle per case of the main proposition: which cells
##     each case reads (v5 ladder: iid / AR(1) stationary / AR(1) age-specific).
##
##   figures/fig_row_reads.pdf
##     Rows of the matrix (fixed earlier wave s) under three noise processes:
##     the "show it in the data" diagnostic - drop-then-straight (iid),
##     straight-but-stepped (random walk), curved approach (AR(1)).
##
## Illustrative values, exaggerated for legibility. Palette as in the anatomy
## charts.
## =============================================================================

dir.create("figures", showWarnings = FALSE)

cols <- c(B = "#0072B2", CHd = "#E69F00", Vd = "#009E73",
          eta = "#CC79A7", eps = "grey55", AR = "#D55E00", sum = "grey40")

## ---- triangle-drawing helper ------------------------------------------------
## Draws cells (s, t) for 0 <= s <= t <= amax; s runs down the left axis,
## t along the bottom; the diagonal is outlined. Optional used(s, t) predicate:
## cells outside it are drawn pale (present but not needed for the reads).
draw_tri <- function(zfun, col, main, amax = 5, labels = TRUE,
                     sub = NULL, cex_main = 1.0, used = NULL) {
  plot(NULL, xlim = c(-0.9, amax + 0.55), ylim = c(-amax - 0.55, 0.9),
       axes = FALSE, xlab = "", ylab = "", asp = 1)
  vals <- outer(0:amax, 0:amax, zfun)
  zmax <- max(vals, na.rm = TRUE); zmin <- min(vals, na.rm = TRUE)
  ramp <- colorRampPalette(c("#FFFFFF", col))(64)
  for (s in 0:amax) for (t in s:amax) {
    z <- zfun(s, t)
    if (is.na(z)) {                                  # unused cell: faint outline
      rect(t - 0.5, -s - 0.5, t + 0.5, -s + 0.5,
           col = "grey96", border = "grey88", lwd = 0.5)
      next
    }
    ci <- if (zmax > zmin + 1e-9) 1 + round(63 * (z - zmin) / (zmax - zmin)) else 32
    pale <- !is.null(used) && !used(s, t)
    rect(t - 0.5, -s - 0.5, t + 0.5, -s + 0.5,
         col = if (pale) adjustcolor(ramp[ci], 0.25) else ramp[ci],
         border = if (pale) "grey90" else "grey80", lwd = 0.5)
    if (labels) text(t, -s, sprintf("%.2f", z), cex = 0.52,
                     col = if (pale) "grey70" else "grey25")
  }
  for (a in 0:amax)                                   # outline the diagonal
    rect(a - 0.5, -a - 0.5, a + 0.5, -a + 0.5, border = "grey10", lwd = 1.5)
  text(0:amax, 0.62, 0:amax, cex = 0.62, col = "grey30")       # t labels
  text(-0.72, -(0:amax), 0:amax, cex = 0.62, col = "grey30")   # s labels
  text(amax / 2, 0.98, "later wave t", cex = 0.6, col = "grey45", xpd = NA)
  text(-1.12, -amax / 2, "earlier wave s", cex = 0.6, col = "grey45",
       srt = 90, xpd = NA)
  title(main, cex.main = cex_main, line = 1.35)
  if (!is.null(sub)) {
    if (!is.list(sub)) sub <- list(sub)
    for (i in seq_along(sub))
      mtext(sub[[i]], side = 1, line = 0.1 + 0.9 * (i - 1), cex = 0.56,
            col = "grey30")
  }
}

## ---- illustrative parameters (as in the surface figures) --------------------
B <- 1.0; Cc <- 0.18; Vd <- 0.12; s2n <- 0.28; s2e <- 0.35
rho <- 0.6; s2ar <- 0.5

## ---- Figure 1: the matrix, layer by layer (appendix) ------------------------
pdf("figures/fig_surface_layers_tri.pdf", width = 10.2, height = 6.6,
    pointsize = 12)
par(mfrow = c(2, 3), mar = c(1.6, 1.4, 2.5, 0.4))
draw_tri(function(s, t) B + 0 * s,            cols["B"],
         expression("level:  " * B))
draw_tri(function(s, t) Cc * (s + t),          cols["CHd"],
         expression("tilt:  " * (s + t) * C[Hd]))
draw_tri(function(s, t) Vd * s * t,            cols["Vd"],
         expression("curvature:  " * s * t * V[d]))
draw_tri(function(s, t) s2n * pmin(s, t),      cols["eta"],
         expression("staircase:  " * G * "(min(s, t))"))
draw_tri(function(s, t) s2e * (s == t),        cols["eps"],
         expression("diagonal spike:  " * sigma[epsilon]^2))
draw_tri(function(s, t) B + Cc * (s + t) + Vd * s * t + s2n * pmin(s, t) +
           s2e * (s == t),                     cols["sum"],
         "the matrix: sum of the five")
dev.off()

## ---- Figure 2: the general template as a menu (appendix) --------------------
pdf("figures/fig_surface_menu_tri.pdf", width = 10.2, height = 6.6,
    pointsize = 12)
par(mfrow = c(2, 3), mar = c(1.6, 1.4, 2.5, 0.4))
draw_tri(function(s, t) 1 + 0 * s,             cols["B"],
         expression("smooth layer,  f = {1}"))
draw_tri(function(s, t) 1 + 0.18 * (s + t) + 0.12 * s * t, cols["B"],
         expression("f = {1, a}   (our linear case)"))
draw_tri(function(s, t) 1 + 0.18 * (s + t) + 0.12 * s * t +
           0.01 * (s^2 + t^2) + 0.002 * s^2 * t^2, cols["B"],
         expression("f = {1, a, " * a^2 * "}"))
draw_tri(function(s, t) 0.5 * (s == t),        cols["eps"],
         "noise layer,  i.i.d.")
draw_tri(function(s, t) 0.28 * pmin(s, t),     cols["eta"],
         "random walk")
draw_tri(function(s, t) 0.5 * 0.6^(t - s),     cols["AR"],
         expression("AR(1):  " * sigma[u]^2 * rho^{t - s}))
dev.off()