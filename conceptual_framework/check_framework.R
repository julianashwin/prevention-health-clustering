## =============================================================================
## check_framework.R  -  conceptual-framework package
##
## Numerical verification of every result in conceptual_framework.tex that is
## flagged "verified numerically": the case reads (Cases 1-4 and the five-wave
## general member of the proposition, including exact recovery on population
## moments), the denoising and D_k objects, the targeting results (Result 6 in
## linear and general form, the life-course channels, the local-reading
## identities), and the growth-regression representation (congruence under
## three noise processes, and exact level-route/coefficient-route equivalence).
##
## Run from this folder:  Rscript check_framework.R
## Each block prints true vs estimated; simulated Ns give 2-3 decimal
## agreement, population-moment blocks are exact.
## =============================================================================

set.seed(42)
N  <- 2e6
Tw <- 6                       # waves a = 0..5

## type distribution
VH  <- 0.36; Vd <- 0.04; CHd <- 0.05
Sig <- matrix(c(VH, CHd, CHd, Vd), 2)
L   <- chol(Sig)
Hd  <- matrix(rnorm(2 * N), N) %*% L
H   <- Hd[, 1]; d <- Hd[, 2]

smooth_h <- function() outer(rep(1, N), 0:(Tw - 1)) * d + H   # H + a d

report <- function(name, true, est)
  cat(sprintf("  %-28s true %8.4f   est %8.4f\n", name, true, est))

## ---- Cases 0-1: stationary iid ----------------------------------------------
s2u <- 0.09
h   <- smooth_h() + matrix(rnorm(N * Tw, 0, sqrt(s2u)), N)
V   <- apply(h, 2, var)
C01 <- cov(h[, 1], h[, 2])

Vd1  <- (V[3] - 2 * V[2] + V[1]) / 2
CHd1 <- (V[2] - V[1] - Vd1) / 2
VH1  <- C01 - CHd1
s2u1 <- V[1] - VH1

cat("Cases 0-1 (stationary iid) - variance-first reads:\n")
report("V_d  (curvature of Var)", Vd, Vd1)
report("C_Hd (slope of Var)",     CHd, CHd1)
report("V_H  (one off-diag cell)", VH, VH1)
report("s2_u",                     s2u, s2u1)

## ---- Case 2: stationary AR(1), three-wave variance-first solve ---------------
rho <- 0.6; s2u <- 0.09
u <- matrix(0, N, Tw)
u[, 1] <- rnorm(N, 0, sqrt(s2u))
for (t in 2:Tw) u[, t] <- rho * u[, t - 1] + rnorm(N, 0, sqrt(s2u * (1 - rho^2)))
h  <- smooth_h() + u
V  <- apply(h, 2, var)

Vd2   <- (V[3] - 2 * V[2] + V[1]) / 2
CHd2  <- (V[2] - V[1] - Vd2) / 2
A0    <- V[1]
A1    <- cov(h[, 1], h[, 2]) - CHd2
A2    <- cov(h[, 1], h[, 3]) - 2 * CHd2
rho2  <- (A1 - A2) / (A0 - A1)
s2u2  <- (A0 - A1) / (1 - rho2)
VH2   <- A0 - s2u2

cat("\nCase 2 (AR(1)) - three-wave variance-first solve:\n")
report("V_d  (variance curvature)", Vd, Vd2)
report("C_Hd (variance slope)",     CHd, CHd2)
report("rho  (fade ratio)",         rho, rho2)
report("s2_u",                      s2u, s2u2)
report("V_H",                       VH, VH2)

## ---- Case 2 diagnostic: the five-wave D_k ladder ----------------------------
dh <- h[, 2:Tw] - h[, 1:(Tw - 1)]
Dk <- function(k) {
  idx <- 1:(ncol(dh) - k)
  mean(sapply(idx, function(t) cov(dh[, t], dh[, t + k])))
}
D1 <- Dk(1); D2 <- Dk(2); D3 <- Dk(3)
rhoL <- (D3 - D2) / (D2 - D1)
cat("\nCase 2 diagnostic - D_k ladder (five waves; spike-robust):\n")
cat(sprintf("  D_k: %.4f  %.4f  %.4f  (-> V_d = %.4f)\n", D1, D2, D3, Vd))
report("rho (ladder gap ratio)", rho, rhoL)

## ---- Denoising (appendix): independent noise, age-specific s2_a -------------
s2a <- c(0.09, 0.13, 0.07, 0.11, 0.10, 0.08)
h   <- smooth_h() + sapply(1:Tw, function(t) rnorm(N, 0, sqrt(s2a[t])))
V   <- apply(h, 2, var)
Cst <- function(s, t) cov(h[, s + 1], h[, t + 1])   # zero-based wave indices

Vst0 <- 2 * Cst(0, 1) - Cst(0, 2)                   # Var*(h_0): extrapolated
Vst1 <- 0.5 * (Cst(0, 1) + Cst(1, 2))               # Var*(h_1): interpolated
Vst2 <- 2 * Cst(1, 2) - Cst(0, 2)                   # Var*(h_2): extrapolated
VdD  <- (Vst2 - 2 * Vst1 + Vst0) / 2
CHdD <- (Vst1 - Vst0 - VdD) / 2
VHD  <- Vst0
s2_1 <- V[2] - Vst1
s2_2 <- V[3] - 0.5 * (Cst(1, 2) + Cst(2, 3))

cat("\nDenoising (age-varying iid) - starred-profile reads:\n")
report("V_d  (starred curvature)", Vd, VdD)
report("C_Hd (starred slope)",     CHd, CHdD)
report("V_H  = Var*(h_0)",         VH, VHD)
report("s2_1 (neighbour average)", s2a[2], s2_1)
report("s2_2 (neighbour average)", s2a[3], s2_2)

## ---- Case 3 (main text in v5): AR(1) with age-specific variances v_a --------
rho <- 0.6
s2eta <- c(0.05, 0.09, 0.04, 0.07, 0.06)            # innovation variances, t = 1..5
v <- numeric(Tw); v[1] <- 0.10
u <- matrix(0, N, Tw)
u[, 1] <- rnorm(N, 0, sqrt(v[1]))
for (t in 2:Tw) {
  u[, t] <- rho * u[, t - 1] + rnorm(N, 0, sqrt(s2eta[t - 1]))
  v[t]   <- rho^2 * v[t - 1] + s2eta[t - 1]
}
h <- smooth_h() + u
V <- apply(h, 2, var)
Cst <- function(s, t) cov(h[, s + 1], h[, t + 1])

S1   <- Cst(0, 2) - 2 * Cst(0, 1) + V[1]
S2   <- Cst(0, 3) - 2 * Cst(0, 2) + Cst(0, 1)
rhoG <- S2 / S1
v0G  <- S1 / (1 - rhoG)^2
VHG  <- V[1] - v0G
CHdG <- Cst(0, 1) - V[1] + (1 - rhoG) * v0G
VdG  <- (Cst(1, 3) - rhoG * Cst(1, 2) - (1 - rhoG) * VHG - (4 - 3 * rhoG) * CHdG) /
        (3 - 2 * rhoG)
v1G  <- V[2] - VHG - 2 * CHdG - VdG
v2G  <- V[3] - VHG - 4 * CHdG - 4 * VdG

cat("\nCase 3 (AR(1), age-specific v_a) - four-wave recipe:\n")
report("rho  (row-0 2nd-diff ratio)", rho, rhoG)
report("v_0",  v[1], v0G)
report("V_H",  VH, VHG)
report("C_Hd", CHd, CHdG)
report("V_d",  Vd, VdG)
report("v_1",  v[2], v1G)
report("v_2",  v[3], v2G)

## ---- Result 6: targeting on initial health ----------------------------------
## P_i(s) = 2[A(s)(H-1)d + B(s)d^2];  under zero third central moments,
## G(s) = Cov(h_0, P_i) = 2A(s) dbar V_H + 2C_Hd [A(s)(Hbar-1) + 2B(s) dbar];
## linear-rule gain = kappa |G|/sd(h_0) = sqrt(lambda_0) x gain if H observed.
Hbar <- 0.85; dbar <- -0.03; s0sq <- 0.09
H2 <- H + Hbar; d2 <- d + dbar
h0 <- H2 + rnorm(N, 0, sqrt(s0sq))
abar <- 5; sA <- 2
aa <- sA:abar
A <- sum(aa - sA + 1); B <- sum(aa * (aa - sA + 1))
Pi <- 2 * (A * (H2 - 1) * d2 + B * d2^2)
G_form <- 2 * A * dbar * VH + 2 * CHd * (A * (Hbar - 1) + 2 * B * dbar)
lam0 <- VH / (VH + s0sq)

cat("\nResult 6 (targeting on initial health, per-person version):\n")
report("G(r) = Cov(h0, P_i)", G_form, cov(h0, Pi))
report("slope beta(r) = G/(V_H + v_0)", G_form / (VH + s0sq),
       coef(lm(Pi ~ h0))[2])
report("slope attenuation = lambda_0", lam0,
       (cov(h0, Pi) / var(h0)) / (cov(H2, Pi) / var(H2)))
report("per-sd gap ratio = sqrt(lambda0)",
       sqrt(lam0), (abs(cov(h0, Pi)) / sd(h0)) / (abs(cov(H2, Pi)) / sd(H2)))
report("E[P_i] check (mean channel + 2K)",
       2 * (A * ((Hbar - 1) * dbar + CHd) + B * (dbar^2 + Vd)), mean(Pi))

## ---------------------------------------------------------------------------
## Life-course channels vs the linear parameters (Section 3.4 bullet)
##   pathway      = (a-r+1) C_Hd
##   persistence  = (r-1)(a-r+1) V_d
##   accumulation = (a-r+1)^2 V_d
##   total        = (a-r+1)(C_Hd + a V_d), the summand of eq (24)
## ---------------------------------------------------------------------------
rr <- 3; aa1 <- 6
hb <- function(x) H2 + x * d2
gap_a  <- 1 - hb(aa1)
decl_a <- hb(rr - 1) - hb(aa1)

cat("\nLife-course channels in the linear case (r = 3, a = 6):\n")
report("pathway = (a-r+1) C_Hd", (aa1 - rr + 1) * CHd, cov(1 - H2, decl_a))
report("persistence = (r-1)(a-r+1) V_d", (rr - 1) * (aa1 - rr + 1) * Vd,
       cov(H2 - hb(rr - 1), decl_a))
report("accumulation = (a-r+1)^2 V_d", (aa1 - rr + 1)^2 * Vd, var(decl_a))
report("total = (a-r+1)(C_Hd + a V_d)",
       (aa1 - rr + 1) * (CHd + aa1 * Vd), cov(gap_a, decl_a))

## ---------------------------------------------------------------------------
## Result 6 in GENERAL form (Section 2.6): quadratic-in-age type paths, so the
## linear window does not hold. Checks eq (16) [two channels] and eq (15) [slope].
##   gap channel(a)         = -(mean decline_a) * Cov_j(hbar_0, hbar_a)
##   compounding channel(a) =  (mean gap_a)     * Cov_j(hbar_0, decline_a)
## ---------------------------------------------------------------------------
set.seed(42)
Ng <- 5e5; abar_g <- 8; rg <- 3; v0g <- 0.05
mug <- c(H = 0.80, d = -0.030, q = -0.002)
Sg  <- matrix(c( 0.030, -1.5e-3, -1e-4,
                -1.5e-3, 1.5e-3,  2e-5,
                -1e-4,   2e-5,    1e-5), 3, 3)
thg  <- MASS::mvrnorm(Ng, mug, Sg)
hbg  <- sapply(0:abar_g, function(a) thg[,1] + a*thg[,2] + a^2*thg[,3])
Pg   <- 2 * rowSums((1 - hbg[, (rg+1):(abar_g+1)]) *
                    (hbg[, rg] - hbg[, (rg+1):(abar_g+1)]))
h0g  <- hbg[,1] + rnorm(Ng, 0, sqrt(v0g))

chan <- sapply(rg:abar_g, function(a) {
  ha <- hbg[, a+1]; dk <- hbg[, rg] - ha
  c(gap  = -mean(dk)      * cov(hbg[,1], ha),
    comp =  mean(1 - ha)  * cov(hbg[,1], dk))
})
C0_form <- 2 * sum(chan)

cat("\nResult 6 in general form (quadratic type paths):\n")
report("Cov(h0, P_i) = Cov_j(hbar_0, P_j)", cov(hbg[,1], Pg), cov(h0g, Pg))
report("two-channel formula, eq (16)", C0_form, cov(hbg[,1], Pg))
report("slope beta(r), eq (15)", C0_form / (var(hbg[,1]) + v0g),
       coef(lm(Pg ~ h0g))[2])
report("slope attenuation = lambda_0",
       var(hbg[,1]) / (var(hbg[,1]) + v0g),
       (cov(h0g, Pg)/var(h0g)) / (cov(hbg[,1], Pg)/var(hbg[,1])))

## ---------------------------------------------------------------------------
## Growth-regression representation is general (Appendix C).
## (Hhat, dhat, e) = M h with M depending only on the wave design, so the
## coefficient second moments equal the congruence M Var(h) M' under ANY noise.
## ---------------------------------------------------------------------------
build_M <- function(ages) {
  X <- cbind(1, ages)
  A <- solve(t(X) %*% X) %*% t(X)                              # coefficient rows
  R <- qr.Q(qr(X), complete = TRUE)[, -(1:2), drop = FALSE]    # residual basis
  rbind(A, t(R))
}
congruence_gap <- function(ages, Sigma_u, n = 4e5) {
  th <- MASS::mvrnorm(n, c(0.85, -0.03),
                      matrix(c(VH, CHd, CHd, Vd), 2, 2))
  u  <- MASS::mvrnorm(n, rep(0, length(ages)), Sigma_u)
  h  <- th[,1] + outer(th[,2], ages) + u
  M  <- build_M(ages)
  max(abs(cov(h %*% t(M)) - M %*% cov(h) %*% t(M)))
}
ag3 <- 0:2; ag4 <- 0:3; rho_g <- 0.6
S3g <- outer(seq_along(ag4), seq_along(ag4), Vectorize(function(i, j) {
  s <- min(i, j); t <- max(i, j); rho_g^(t - s) * c(.50, .65, .40, .55)[s]
}))

cat("\nGrowth-regression congruence (should be 0 in every case):\n")
report("Case 1 (i.i.d.)", 0, congruence_gap(ag3, diag(0.30, 3)))
report("Case 2 (AR(1), stationary)", 0,
       congruence_gap(ag3, 0.30 * rho_g^abs(outer(ag3, ag3, "-"))))
report("Case 3 (AR(1), age-specific)", 0, congruence_gap(ag4, S3g))

## ---------------------------------------------------------------------------
## Local-reading identities (Section 3.4 "The local reading")
##   C_Hd(a) = Cov_j(hbar_a, d) = C_Hd + a V_d
##   Var_j(hbar_{a+1}) - Var_j(hbar_a) = C_Hd(a) + C_Hd(a+1)
##   in-window row slope at base s = C_Hd(s), constant in lag
## ---------------------------------------------------------------------------
cat("\nLocal-reading identities:\n")
report("C_Hd(3) = Cov(hbar_3, d)", CHd + 3*Vd, cov(hb(3), d2))
report("Var step 3->4 = C_Hd(3)+C_Hd(4)",
       (CHd + 3*Vd) + (CHd + 4*Vd), var(hb(4)) - var(hb(3)))
report("row slope at s=2 (lag 4 vs 3)", CHd + 2*Vd,
       cov(hb(2), hb(6)) - cov(hb(2), hb(5)))
report("Plin summand = (a-r+1) C_Hd(a), a=6, r=3",
       (6-3+1)*(CHd + 6*Vd), (6-3+1)*cov(hb(6), d2))

## ---------------------------------------------------------------------------
## Case 4: AR(1) persistent noise + one-period measurement error (4 waves)
##   diagonal reads unchanged (total noise variance stationary);
##   rho from OFF-diagonal drops of the detilted row (the spike never leaves
##   the diagonal); spike = anchor's excess over the fade.
## Also: the spike-blind Case 2 read is biased; and the 5-wave GENERAL member
## (free v_s AND free spike_s) for part (i) of the widened proposition.
## ---------------------------------------------------------------------------
set.seed(44)
Nc4 <- 1e6
VH4 <- 1.0; C4 <- 0.22; Vd4 <- 0.18; rho4 <- 0.6; s2w4 <- 0.32; s2e4 <- 0.18
th4 <- MASS::mvrnorm(Nc4, c(0, 0), matrix(c(VH4, C4, C4, Vd4), 2))
w4 <- matrix(rnorm(Nc4, 0, sqrt(s2w4)), Nc4, 1)
for (t in 2:4) w4 <- cbind(w4, rho4 * w4[, t-1] + rnorm(Nc4, 0, sqrt(s2w4 * (1 - rho4^2))))
h4 <- sapply(0:3, function(a) th4[,1] + a * th4[,2]) + w4 +
      matrix(rnorm(Nc4 * 4, 0, sqrt(s2e4)), Nc4, 4)
V4  <- apply(h4, 2, var)
C04 <- sapply(1:3, function(a) cov(h4[,1], h4[,1+a]))
Vd_h  <- 0.5 * (V4[3] - 2 * V4[2] + V4[1])
CHd_h <- 0.5 * (V4[2] - V4[1] - Vd_h)
Ct4 <- C04 - (1:3) * CHd_h
rho_h <- (Ct4[2] - Ct4[3]) / (Ct4[1] - Ct4[2])
s2w_h <- (Ct4[1] - Ct4[2]) / (rho_h * (1 - rho_h))
VH_h  <- Ct4[1] - rho_h * s2w_h
cat("\nCase 4 (AR(1) + spike, four waves):\n")
report("V_d from diagonal", Vd4, Vd_h)
report("C_Hd from diagonal", C4, CHd_h)
report("rho from off-diagonal drops", rho4, rho_h)
report("s2_w", s2w4, s2w_h)
report("V_H", VH4, VH_h)
report("s2_eps = V(0) - V_H - s2_w", s2e4, V4[1] - VH_h - s2w_h)
report("interior check Cov(h1,h2)", VH4 + 3*C4 + 2*Vd4 + rho4*s2w4, cov(h4[,2], h4[,3]))
report("spike-blind Case 2 rho (biased)", rho4,
       (Ct4[1] - Ct4[2]) / (V4[1] - Ct4[1]))

## ---- the 5-wave GENERAL member: free v_s and free spike_s ------------------
va5 <- c(0.30, 0.42, 0.26, 0.36, 0.32)          # persistent variances by age
se5 <- c(0.20, 0.12, 0.24, 0.16, 0.18)          # spike variances by age
w5 <- matrix(rnorm(Nc4, 0, sqrt(va5[1])), Nc4, 1)
for (t in 2:5) w5 <- cbind(w5, rho4 * w5[, t-1] +
  rnorm(Nc4, 0, sqrt(pmax(va5[t] - rho4^2 * va5[t-1], 1e-8))))
h5 <- sapply(0:4, function(a) th4[,1] + a * th4[,2]) + w5 +
      sapply(1:5, function(t) rnorm(Nc4, 0, sqrt(se5[t])))
V5  <- apply(h5, 2, var)
C05 <- sapply(1:4, function(a) cov(h5[,1], h5[,1+a]))
S2 <- C05[3] - 2 * C05[2] + C05[1]              # second diffs EXCLUDING anchor
S3 <- C05[4] - 2 * C05[3] + C05[2]
rho_g <- S3 / S2
v0_g  <- S2 / (rho_g * (1 - rho_g)^2)
CHd_g <- C05[2] - C05[1] + rho_g * (1 - rho_g) * v0_g
VH_g  <- C05[1] - CHd_g - rho_g * v0_g
Vd_g  <- (cov(h5[,2], h5[,4]) - rho_g * cov(h5[,2], h5[,3]) -
          (1 - rho_g) * VH_g - (4 - 3 * rho_g) * CHd_g) / (3 - 2 * rho_g)
v1_g  <- (cov(h5[,2], h5[,3]) - (VH_g + 3*CHd_g + 2*Vd_g)) / rho_g
se0_g <- V5[1] - VH_g - v0_g
se1_g <- V5[2] - (VH_g + 2*CHd_g + Vd_g) - v1_g
cat("\nGeneral member (free v_s + free spike_s, five waves):\n")
report("rho", rho4, rho_g)
report("v_0", va5[1], v0_g)
report("C_Hd", C4, CHd_g)
report("V_H", VH4, VH_g)
report("V_d", Vd4, Vd_g)
report("spike_0", se5[1], se0_g)
report("spike_1", se5[2], se1_g)

## ---- the same two reads on EXACT population moments (identification proper);
## the sampling wobble above is the known small-second-differences fragility
cell5 <- function(s, t) {         # 1-indexed ages 0..4; s <= t
  a <- s - 1; b <- t - 1
  det <- VH4 + (a + b) * C4 + a * b * Vd4
  det + rho4^(b - a) * va5[s] + (s == t) * se5[s]
}
C0x <- sapply(2:5, function(t) cell5(1, t)); V0x <- cell5(1, 1)
S2x <- C0x[3] - 2 * C0x[2] + C0x[1]; S3x <- C0x[4] - 2 * C0x[3] + C0x[2]
rx <- S3x / S2x; v0x <- S2x / (rx * (1 - rx)^2)
Cx <- C0x[2] - C0x[1] + rx * (1 - rx) * v0x
Vx <- C0x[1] - Cx - rx * v0x
Vdx <- (cell5(2, 4) - rx * cell5(2, 3) - (1 - rx) * Vx - (4 - 3 * rx) * Cx) / (3 - 2 * rx)
cat("\nGeneral member on exact population cells (should be exact):\n")
report("rho", rho4, rx); report("v_0", va5[1], v0x)
report("C_Hd", C4, Cx);  report("V_H", VH4, Vx); report("V_d", Vd4, Vdx)
report("spike_0", se5[1], V0x - Vx - v0x)


## ---------------------------------------------------------------------------
## Growth-regression representation: exact route equivalence (Appendix C).
## The denoising solve (3 waves, free noise variance per age: 6 moments, 6
## parameters, just-identified) and its coefficient-space twin invert the SAME
## map in two coordinate systems, so they agree to machine precision on any
## sample - the <= 4e-14 claim in the appendix.
## ---------------------------------------------------------------------------
set.seed(7)
Ne <- 2e5
the <- MASS::mvrnorm(Ne, c(0, 0), matrix(c(1.0, 0.05, 0.05, 0.04), 2))
he <- sapply(0:2, function(a) the[,1] + a * the[,2]) +
      sapply(1:3, function(t) rnorm(Ne, 0, sqrt(c(0.45, 0.55, 0.40)[t])))
S <- cov(he)
## level route: the denoising solve off the three off-diagonal cells
CHd_lvl <- S[1,3] - S[1,2]
Vd_lvl  <- 0.5 * (S[2,3] + S[1,2] - 2 * S[1,3])
VH_lvl  <- S[1,2] - CHd_lvl
## coefficient route: (Hhat, dhat, e) moments as quadratic forms in S, then
## the exact 6-parameter solve (noise variances from the e-moments first)
qf <- function(w1, w2) as.numeric(t(w1) %*% S %*% w2)
wd <- c(-1, 0, 1) / 2; wH <- c(5, 2, -1) / 6; we <- c(1, -2, 1)
A3 <- rbind(c(1, 4, 1),            # Var(e)      = s0 + 4 s1 + s2
            c(-1/2, 0, 1/2),       # Cov(dhat,e) = (s2 - s0)/2
            c(5/6, -4/6, -1/6))    # Cov(Hhat,e) = (5 s0 - 4 s1 - s2)/6
sv <- solve(A3, c(qf(we, we), qf(wd, we), qf(wH, we)))
Vd_cf  <- qf(wd, wd) - (sv[1] + sv[3]) / 4
CHd_cf <- qf(wH, wd) + (5 * sv[1] + sv[3]) / 12
VH_cf  <- qf(wH, wH) - (25 * sv[1] + 4 * sv[2] + sv[3]) / 36
cat("\nGrowth-regression route equivalence (same sample matrix; exact):\n")
report("V_d level vs coefficient route", Vd_lvl, Vd_cf)
report("C_Hd level vs coefficient route", CHd_lvl, CHd_cf)
report("V_H level vs coefficient route", VH_lvl, VH_cf)
cat(sprintf("  max abs discrepancy          %.2e (machine precision)\n",
    max(abs(c(Vd_lvl - Vd_cf, CHd_lvl - CHd_cf, VH_lvl - VH_cf)))))
