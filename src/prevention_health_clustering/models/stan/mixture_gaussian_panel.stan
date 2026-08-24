/**
 * Latent-class growth mixture for one or more aligned continuous channels.
 *
 * Covers the PCS-only and joint PCS+MCS model families, and the single-class
 * and iid baselines, in one program. Variants are selected by DATA, not by
 * separate files:
 *
 *   C            1 = univariate, 2 = bivariate (channels are row-aligned)
 *   K            number of latent classes; K = 1 gives the single-class baseline
 *   P            design width: 3 = quadratic [1, a, a^2], more = spline basis
 *   ar_mode      0 = conditionally independent, 1 = AR(1) on the class residual
 *   N_cohort     1 = no cohort effects (the parameter becomes length zero)
 *   homosigma    1 = one residual scale per channel, 0 = one per class
 *   n_hold[i]    0 = no held-out rows for person i
 *
 * Every "off" setting collapses a parameter to length zero, so it contributes
 * no sampled dimension and no prior term. Branches sit at the person-class
 * level, never inside the observation loop.
 *
 * IDENTIFICATION. Class labels are anchored by ordering the intercept of the
 * anchor channel, which `transformed data` requires to carry observations. The
 * predecessor project ordered a channel that could be masked out of the
 * likelihood, leaving labels exchangeable while diagnostics looked clean.
 *
 * The per-person class log-likelihood is defined ONCE, in `person_class_loglik`,
 * and called from both the model block and generated quantities.
 */

functions {
  /**
   * log( theta_k ) + log p(y_i | class k), for every k, over one row window.
   *
   * Passing the window explicitly is what makes held-out scoring free: the
   * fitted rows and the held-out rows use the same function.
   */
  vector person_class_loglik(
      int row_start, int row_end,
      int include_log_weight,
      int K, int C, int P,
      array[] vector y, matrix X,
      array[] int cohort_id,
      int ar_mode, vector age_gap,
      vector log_weight,
      array[] matrix coef,          // C matrices, each K x P
      matrix sigma,                 // K x C
      vector rho,                   // length K (or 0 when ar_mode == 0)
      array[] vector cohort_effect, // C vectors, each length N_cohort
      real obs_weight) {
    vector[K] class_lp =
        include_log_weight == 1 ? log_weight : rep_vector(0.0, K);
    if (row_start == 0 || row_end < row_start) {
      return class_lp;
    }
    for (k in 1:K) {
      real total = 0;
      for (c in 1:C) {
        if (ar_mode == 0) {
          for (n in row_start:row_end) {
            real mu = dot_product(X[n], coef[c][k])
                      + cohort_effect[c][cohort_id[n]];
            total += normal_lpdf(y[c][n] | mu, sigma[k, c]);
          }
        } else {
          // AR(1) on the deviation from the class mean. The first observation
          // uses the stationary variance; later ones use the gap-adjusted
          // recursion, which reduces to the plain AR(1) when gaps are 1.
          real mu_prev = 0;
          for (n in row_start:row_end) {
            real mu = dot_product(X[n], coef[c][k])
                      + cohort_effect[c][cohort_id[n]];
            if (n == row_start) {
              total += normal_lpdf(y[c][n] | mu,
                                   sigma[k, c] / sqrt(1 - square(rho[k])));
            } else {
              real gap = age_gap[n];
              real rho_gap = pow(rho[k], gap);
              real var_mult = (1 - pow(rho[k], 2 * gap)) / (1 - square(rho[k]));
              total += normal_lpdf(
                  y[c][n] | mu + rho_gap * (y[c][n - 1] - mu_prev),
                  sigma[k, c] * sqrt(fmax(var_mult, 1e-9)));
            }
            mu_prev = mu;
          }
        }
      }
      class_lp[k] += obs_weight * total;
    }
    return class_lp;
  }

  real partial_sum_lpmf(
      array[] int person_slice, int start, int end,
      int K, int C, int P,
      array[] vector y, matrix X,
      array[] int cohort_id,
      int ar_mode, vector age_gap,
      array[] int fit_start, array[] int fit_end,
      vector log_weight,
      array[] matrix coef, matrix sigma, vector rho,
      array[] vector cohort_effect,
      real person_weight_power) {
    real lp = 0;
    for (idx in 1:size(person_slice)) {
      int i = person_slice[idx];
      real n_obs = fit_end[i] - fit_start[i] + 1;
      real obs_weight = pow(n_obs, -person_weight_power);
      lp += log_sum_exp(person_class_loglik(
          fit_start[i], fit_end[i], 1, K, C, P, y, X, cohort_id,
          ar_mode, age_gap, log_weight, coef, sigma, rho,
          cohort_effect, obs_weight));
    }
    return lp;
  }
}

data {
  int<lower=1> N_obs;
  int<lower=1> N_person;
  int<lower=1> K;
  int<lower=1> C;
  int<lower=2> P;

  array[C] vector[N_obs] y;
  matrix[N_obs, P] X;

  array[N_person] int<lower=1, upper=N_obs> fit_start;
  array[N_person] int<lower=1, upper=N_obs> fit_end;
  array[N_person] int<lower=0, upper=N_obs> hold_start;
  array[N_person] int<lower=0, upper=N_obs> hold_end;

  int<lower=0, upper=1> ar_mode;
  vector<lower=0>[ar_mode == 0 ? 0 : N_obs] age_gap;

  int<lower=1> N_cohort;
  array[N_obs] int<lower=1, upper=N_cohort> cohort_id;
  int<lower=0, upper=1> cohort_by_class;

  int<lower=0, upper=1> homosigma;
  int<lower=1, upper=C> anchor_channel;
  real<lower=0, upper=1> person_weight_power;

  array[C] int<lower=0> channel_n_obs;

  real alpha_prior_scale;
  vector<lower=0>[P - 1] coef_prior_scale;
  real<lower=0> sigma_prior_location;
  real<lower=0> sigma_prior_scale;
  real<lower=0> theta_prior_concentration;
  real<lower=0> cohort_prior_scale;
  real<lower=0> rho_prior_alpha;
  real<lower=0> rho_prior_beta;

  // Person-level generated quantities are O(N_person * K) PER DRAW. On the
  // full roster that is 301,200 columns and ~9.6 GB of chain CSV for a model
  // whose scientific content is ~20 parameters. Off by default; turn on for a
  // short, thinned pass when class probabilities are actually needed.
  int<lower=0, upper=1> emit_person_quantities;

  int<lower=1> grainsize;
}

transformed data {
  array[N_person] int person_index;
  for (i in 1:N_person) {
    person_index[i] = i;
  }

  // The ordered anchor must carry likelihood weight, or class labels are
  // exchangeable and the reported diagnostics are meaningless.
  if (channel_n_obs[anchor_channel] == 0) {
    reject("Anchor channel ", anchor_channel, " has no observations: ",
           "class labels would be exchangeable.");
  }
}

parameters {
  simplex[K] theta;

  ordered[K] anchor_intercept;
  matrix[K, C - 1] other_intercept;
  array[C] matrix[K, P - 1] slope;

  matrix<lower=0.05>[homosigma == 1 ? 1 : K, C] sigma_raw;
  vector<lower=0, upper=0.99>[ar_mode == 0 ? 0 : K] rho;
  array[C] matrix[cohort_by_class == 1 ? K : 1, N_cohort - 1] cohort_free;
}

transformed parameters {
  array[C] matrix[K, P] coef;
  matrix[K, C] sigma;
  array[C] vector[N_cohort] cohort_effect;
  vector[K] log_weight = log(theta);

  for (c in 1:C) {
    // Column 1 of the design is the intercept; the anchor channel takes the
    // ordered vector, the others are free.
    for (k in 1:K) {
      coef[c][k, 1] = c == anchor_channel
          ? anchor_intercept[k]
          : other_intercept[k, c > anchor_channel ? c - 1 : c];
      for (p in 2:P) {
        coef[c][k, p] = slope[c][k, p - 1];
      }
      sigma[k, c] = homosigma == 1 ? sigma_raw[1, c] : sigma_raw[k, c];
    }
    // Cohort effects are class-independent by default and hoisted out of the
    // per-class loop. The reference cohort is pinned to zero.
    cohort_effect[c] = rep_vector(0.0, N_cohort);
    if (N_cohort > 1) {
      for (j in 2:N_cohort) {
        cohort_effect[c][j] = cohort_free[c][1, j - 1];
      }
    }
  }
}

model {
  theta ~ dirichlet(rep_vector(theta_prior_concentration, K));
  anchor_intercept ~ normal(0, alpha_prior_scale);
  to_vector(other_intercept) ~ normal(0, alpha_prior_scale);
  for (c in 1:C) {
    for (p in 1:(P - 1)) {
      col(slope[c], p) ~ normal(0, coef_prior_scale[p]);
    }
    if (N_cohort > 1) {
      to_vector(cohort_free[c]) ~ normal(0, cohort_prior_scale);
    }
  }
  to_vector(sigma_raw) ~ normal(sigma_prior_location, sigma_prior_scale);
  if (ar_mode != 0) {
    rho ~ beta(rho_prior_alpha, rho_prior_beta);
  }

  target += reduce_sum(
      partial_sum_lpmf, person_index, grainsize,
      K, C, P, y, X, cohort_id, ar_mode, age_gap,
      fit_start, fit_end, log_weight, coef, sigma, rho,
      cohort_effect, person_weight_power);
}

generated quantities {
  // Sized to zero unless requested, so the draws file stays small.
  matrix[emit_person_quantities ? N_person : 0, K] class_prob;
  array[emit_person_quantities ? N_person : 0] int<lower=1, upper=K> modal_class;
  vector[emit_person_quantities ? N_person : 0] log_lik;
  vector[emit_person_quantities ? N_person : 0] log_lik_heldout;

  if (emit_person_quantities == 1) {
    for (i in 1:N_person) {
      real n_obs = fit_end[i] - fit_start[i] + 1;
      real obs_weight = pow(n_obs, -person_weight_power);
      vector[K] fitted_lp = person_class_loglik(
          fit_start[i], fit_end[i], 1, K, C, P, y, X, cohort_id,
          ar_mode, age_gap, log_weight, coef, sigma, rho,
          cohort_effect, obs_weight);

      log_lik[i] = log_sum_exp(fitted_lp);
      class_prob[i] = softmax(fitted_lp)';
      modal_class[i] = sort_indices_desc(fitted_lp)[1];

      // Conditional predictive density of the held-out rows given the fitted
      // rows, marginalising the latent class. Zero when no rows are held out.
      if (hold_start[i] > 0) {
        vector[K] hold_lp = person_class_loglik(
            hold_start[i], hold_end[i], 0, K, C, P, y, X, cohort_id,
            ar_mode, age_gap, log_weight, coef, sigma, rho,
            cohort_effect, 1.0);
        log_lik_heldout[i] = log_sum_exp(fitted_lp + hold_lp) - log_lik[i];
      } else {
        log_lik_heldout[i] = 0;
      }
    }
  }
}
