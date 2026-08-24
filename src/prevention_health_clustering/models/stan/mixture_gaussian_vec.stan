/**
 * Vectorised form of the conditionally-independent Gaussian mixture.
 *
 * Same model as mixture_gaussian_panel.stan, same likelihood, but the inner
 * loop over a person's rows is replaced by two vectorised calls:
 *
 *   mu_seg = X_seg * b            one matrix-vector product (data * var)
 *   normal_lpdf(y_seg | mu_seg, sigma)   one vectorised density call
 *
 * The earlier sufficient-statistic attempt was 1.86x SLOWER because it routed
 * around Stan's hand-written analytic gradient for normal_lpdf. This variant
 * does the opposite: it hands normal_lpdf as much work per call as possible.
 *
 * Restricted to ar_mode == 0, which is the headline case. The AR recursion is
 * inherently sequential and cannot be vectorised this way.
 */

functions {
  vector person_class_loglik_vec(
      int row_start, int row_end,
      int K, int C, int P,
      array[] vector y, matrix X,
      vector cohort_row,
      int use_cohort,
      vector log_weight,
      array[] matrix coef,
      matrix sigma,
      real obs_weight) {
    vector[K] class_lp = log_weight;
    int n = row_end - row_start + 1;
    matrix[n, P] X_seg = block(X, row_start, 1, n, P);
    for (k in 1:K) {
      real total = 0;
      for (c in 1:C) {
        vector[n] mu_seg = X_seg * coef[c][k]';
        if (use_cohort == 1) {
          mu_seg += segment(cohort_row, row_start, n);
        }
        total += normal_lpdf(segment(y[c], row_start, n) | mu_seg, sigma[k, c]);
      }
      class_lp[k] += obs_weight * total;
    }
    return class_lp;
  }

  real partial_sum_lpmf(
      array[] int person_slice, int start, int end,
      int K, int C, int P,
      array[] vector y, matrix X,
      vector cohort_row, int use_cohort,
      array[] int fit_start, array[] int fit_end,
      vector log_weight,
      array[] matrix coef, matrix sigma,
      real person_weight_power) {
    real lp = 0;
    for (idx in 1:size(person_slice)) {
      int i = person_slice[idx];
      real n_obs = fit_end[i] - fit_start[i] + 1;
      real obs_weight = pow(n_obs, -person_weight_power);
      lp += log_sum_exp(person_class_loglik_vec(
          fit_start[i], fit_end[i], K, C, P, y, X, cohort_row, use_cohort,
          log_weight, coef, sigma, obs_weight));
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

  int<lower=1> N_cohort;
  array[N_obs] int<lower=1, upper=N_cohort> cohort_id;

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

  int<lower=0, upper=1> emit_person_quantities;

  int<lower=1> grainsize;
}

transformed data {
  array[N_person] int person_index;
  int use_cohort = N_cohort > 1 ? 1 : 0;
  for (i in 1:N_person) {
    person_index[i] = i;
  }
  if (channel_n_obs[anchor_channel] == 0) {
    reject("Anchor channel ", anchor_channel, " has no observations.");
  }
}

parameters {
  simplex[K] theta;
  ordered[K] anchor_intercept;
  matrix[K, C - 1] other_intercept;
  array[C] matrix[K, P - 1] slope;
  matrix<lower=0.05>[homosigma == 1 ? 1 : K, C] sigma_raw;
  vector[N_cohort - 1] cohort_free;
}

transformed parameters {
  array[C] matrix[K, P] coef;
  matrix[K, C] sigma;
  vector[K] log_weight = log(theta);
  // Class-independent additive shift, expanded to rows once per gradient
  // rather than K times inside the class loop.
  vector[N_obs] cohort_row = rep_vector(0.0, N_obs);

  for (c in 1:C) {
    for (k in 1:K) {
      coef[c][k, 1] = c == anchor_channel
          ? anchor_intercept[k]
          : other_intercept[k, c > anchor_channel ? c - 1 : c];
      for (p in 2:P) {
        coef[c][k, p] = slope[c][k, p - 1];
      }
      sigma[k, c] = homosigma == 1 ? sigma_raw[1, c] : sigma_raw[k, c];
    }
  }
  if (N_cohort > 1) {
    vector[N_cohort] cohort_level = append_row(0.0, cohort_free);
    for (n in 1:N_obs) {
      cohort_row[n] = cohort_level[cohort_id[n]];
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
  }
  to_vector(sigma_raw) ~ normal(sigma_prior_location, sigma_prior_scale);
  if (N_cohort > 1) {
    cohort_free ~ normal(0, cohort_prior_scale);
  }

  target += reduce_sum(
      partial_sum_lpmf, person_index, grainsize,
      K, C, P, y, X, cohort_row, use_cohort,
      fit_start, fit_end, log_weight, coef, sigma,
      person_weight_power);
}

generated quantities {
  matrix[emit_person_quantities ? N_person : 0, K] class_prob;
  vector[emit_person_quantities ? N_person : 0] log_lik;

  if (emit_person_quantities == 1) {
    for (i in 1:N_person) {
      real n_obs = fit_end[i] - fit_start[i] + 1;
      real obs_weight = pow(n_obs, -person_weight_power);
      vector[K] lp = person_class_loglik_vec(
          fit_start[i], fit_end[i], K, C, P, y, X, cohort_row, use_cohort,
          log_weight, coef, sigma, obs_weight);
      log_lik[i] = log_sum_exp(lp);
      class_prob[i] = softmax(lp)';
    }
  }
}
