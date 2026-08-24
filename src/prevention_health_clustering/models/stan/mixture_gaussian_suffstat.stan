/**
 * Sufficient-statistic form of the conditionally-independent Gaussian mixture.
 *
 * A benchmark variant of mixture_gaussian_panel.stan, restricted to the case
 * where the closed form is exact:
 *
 *   ar_mode == 0   (no sequential recursion)
 *   N_cohort == 1  (no per-row additive shift in the mean)
 *   sigma constant within a person
 *
 * Under those conditions
 *
 *   sum_n (y_n - X_n' b)^2  =  s_i - 2 b' v_i + b' M_i b
 *
 * with per-person statistics
 *
 *   n_i = count,  s_i = sum y_n^2,  v_i = sum X_n y_n,  M_i = sum X_n X_n'
 *
 * so the inner loop over a person's rows collapses to one quadratic form. The
 * statistics are computed once in transformed data and carry no autodiff cost.
 *
 * This file exists to measure the speedup. If it pays, the branch should be
 * folded into the main program as a fast path selected in transformed data.
 */

functions {
  vector person_class_loglik_ss(
      int i,
      int K, int C, int P,
      vector log_weight,
      array[] matrix coef,      // C matrices, each K x P
      matrix sigma,             // K x C
      array[] matrix M,         // N_person matrices, each P x P
      vector n_obs,             // N_person
      array[] matrix V,         // C matrices, each N_person x P
      matrix S,                 // N_person x C
      real obs_weight) {
    vector[K] class_lp = log_weight;
    for (k in 1:K) {
      real total = 0;
      for (c in 1:C) {
        vector[P] b = coef[c][k]';
        real quad = quad_form_sym(M[i], b) - 2 * dot_product(b, V[c][i]');
        total += -n_obs[i] * (0.5 * log(2 * pi()) + log(sigma[k, c]))
                 - (S[i, c] + quad) / (2 * square(sigma[k, c]));
      }
      class_lp[k] += obs_weight * total;
    }
    return class_lp;
  }

  real partial_sum_lpmf(
      array[] int person_slice, int start, int end,
      int K, int C, int P,
      vector log_weight,
      array[] matrix coef, matrix sigma,
      array[] matrix M, vector n_obs, array[] matrix V, matrix S,
      real person_weight_power) {
    real lp = 0;
    for (idx in 1:size(person_slice)) {
      int i = person_slice[idx];
      real obs_weight = pow(n_obs[i], -person_weight_power);
      lp += log_sum_exp(person_class_loglik_ss(
          i, K, C, P, log_weight, coef, sigma, M, n_obs, V, S, obs_weight));
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

  int<lower=1, upper=C> anchor_channel;
  int<lower=0, upper=1> homosigma;
  real<lower=0, upper=1> person_weight_power;
  array[C] int<lower=0> channel_n_obs;

  real alpha_prior_scale;
  vector<lower=0>[P - 1] coef_prior_scale;
  real<lower=0> sigma_prior_location;
  real<lower=0> sigma_prior_scale;
  real<lower=0> theta_prior_concentration;

  int<lower=1> grainsize;
}

transformed data {
  array[N_person] int person_index;
  array[N_person] matrix[P, P] M;
  vector[N_person] n_obs;
  array[C] matrix[N_person, P] V;
  matrix[N_person, C] S;

  if (channel_n_obs[anchor_channel] == 0) {
    reject("Anchor channel ", anchor_channel, " has no observations.");
  }

  for (i in 1:N_person) {
    person_index[i] = i;
    n_obs[i] = fit_end[i] - fit_start[i] + 1;

    matrix[P, P] moment = rep_matrix(0.0, P, P);
    for (n in fit_start[i]:fit_end[i]) {
      moment += X[n]' * X[n];
    }
    M[i] = moment;

    for (c in 1:C) {
      vector[P] cross = rep_vector(0.0, P);
      real sum_sq = 0;
      for (n in fit_start[i]:fit_end[i]) {
        cross += X[n]' * y[c][n];
        sum_sq += square(y[c][n]);
      }
      V[c][i] = cross';
      S[i, c] = sum_sq;
    }
  }
}

parameters {
  simplex[K] theta;
  ordered[K] anchor_intercept;
  matrix[K, C - 1] other_intercept;
  array[C] matrix[K, P - 1] slope;
  matrix<lower=0.05>[homosigma == 1 ? 1 : K, C] sigma_raw;
}

transformed parameters {
  array[C] matrix[K, P] coef;
  matrix[K, C] sigma;
  vector[K] log_weight = log(theta);

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

  target += reduce_sum(
      partial_sum_lpmf, person_index, grainsize,
      K, C, P, log_weight, coef, sigma, M, n_obs, V, S,
      person_weight_power);
}

generated quantities {
  matrix[N_person, K] class_prob;
  vector[N_person] log_lik;

  for (i in 1:N_person) {
    real obs_weight = pow(n_obs[i], -person_weight_power);
    vector[K] lp = person_class_loglik_ss(
        i, K, C, P, log_weight, coef, sigma, M, n_obs, V, S, obs_weight);
    log_lik[i] = log_sum_exp(lp);
    class_prob[i] = softmax(lp)';
  }
}
