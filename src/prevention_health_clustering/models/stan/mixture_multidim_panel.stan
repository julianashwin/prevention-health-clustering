/**
 * Multidimensional latent-class growth mixture over four health channels.
 *
 * Channels, all on one shared person-wave row set:
 *   1..C   Gaussian    the GRM thetas (physical, mental), standardised,
 *                      with optional AR(1) persistence in the class residual
 *                      (ar_mode 1) or an AR(1) latent state plus i.i.d.
 *                      measurement error (ar_mode 2, Kalman-filtered)
 *   chronic  count     cumulated chronic-condition count, negative binomial,
 *                      log link; masked, because the condition inventory is
 *                      not asked of the BHPS entrants
 *   mort     binary    discrete-time mortality hazard, logit link; masked.
 *                      Toggled by `use_mortality`; when off its parameters
 *                      are zero-sized and never sampled.
 *
 * WINDOWS. The Gaussian and count channels use the fitted window and are
 * scored on the held-out window. Mortality uses the person's FULL window and
 * is never held out: a decedent's single event sits by construction at their
 * last observed wave, so holding out the last rows would remove every event
 * from the likelihood rather than test the channel.
 *
 * IDENTIFICATION. As in the Gaussian model: an ordered intercept on the
 * anchor Gaussian channel, which must carry likelihood weight.
 *
 * AR_MODE 2 applies to the GAUSSIAN CHANNELS ONLY. The chronic count and the
 * mortality hazard have no measurement-error analogue here: a condition count
 * is a report of an accumulated stock rather than a noisy read of a
 * continuous state, and mortality is administrative. So sigma_meas is length
 * C, one scale per Gaussian channel, shared across classes as in the
 * univariate model.
 *
 * The per-person class log-likelihood is defined once in the functions block
 * and reused by the sampler and the gated generated quantities.
 */
functions {
  /**
   * Windowed channels: the Gaussian block and the chronic count.
   * `prev_row` anchors the ar_mode 1 recursion -- 0 opens the window at the
   * stationary variance, a row index carries that row's deviation forward.
   * `filter_start` does the same job for ar_mode 2, where one noisy last
   * value is not sufficient to condition on: rows from filter_start up to
   * row_start are filtered but not scored.
   */
  vector person_window_loglik(
      int row_start, int row_end, int include_log_weight, int prev_row,
      int filter_start,
      int K, int C, int P,
      array[] vector y, matrix X,
      int ar_mode, vector age_gap,
      vector log_weight,
      array[] matrix coef, matrix sigma, vector rho, vector sigma_meas,
      array[] int chronic_obs, array[] int chronic_y,
      matrix coef_chronic, real phi_chronic,
      vector channel_weight) {
    vector[K] class_lp =
        include_log_weight == 1 ? log_weight : rep_vector(0.0, K);
    if (row_start == 0 || row_end < row_start) {
      return class_lp;
    }
    for (k in 1:K) {
      real total = 0;
      for (c in 1:C) {
        real part = 0;
        if (ar_mode == 0) {
          for (n in row_start:row_end) {
            part += normal_lpdf(y[c][n] | dot_product(X[n], coef[c][k]),
                                sigma[k, c]);
          }
        } else if (ar_mode == 1) {
          real mu_prev = 0;
          for (n in row_start:row_end) {
            real mu = dot_product(X[n], coef[c][k]);
            if (n == row_start && prev_row == 0) {
              part += normal_lpdf(y[c][n] | mu,
                                  sigma[k, c] / sqrt(1 - square(rho[k])));
            } else {
              if (n == row_start) {
                mu_prev = dot_product(X[prev_row], coef[c][k]);
              }
              real gap = age_gap[n];
              real rho_gap = pow(rho[k], gap);
              real var_mult = (1 - pow(rho[k], 2 * gap)) / (1 - square(rho[k]));
              real y_prev = n == row_start ? y[c][prev_row] : y[c][n - 1];
              part += normal_lpdf(y[c][n] | mu + rho_gap * (y_prev - mu_prev),
                                  sigma[k, c] * sqrt(fmax(var_mult, 1e-9)));
            }
            mu_prev = mu;
          }
        } else {
          // AR(1) latent state plus i.i.d. measurement error, marginalised
          // by a Kalman filter; identical arithmetic to the univariate model.
          real rho_k = rho[k];
          real v_stat = square(sigma[k, c]) / (1 - square(rho_k));
          real var_meas = square(sigma_meas[c]);
          int f0 = filter_start == 0 ? row_start : filter_start;
          real a = 0;
          real Pv = v_stat;
          for (n in f0:row_end) {
            real mu = dot_product(X[n], coef[c][k]);
            if (n > f0) {
              real gap = age_gap[n];
              real rho_gap = pow(rho_k, gap);
              a = rho_gap * a;
              Pv = square(rho_gap) * Pv
                   + v_stat * fmax(1 - pow(rho_k, 2 * gap), 1e-9);
            }
            real F = Pv + var_meas;
            real v = y[c][n] - mu - a;
            if (n >= row_start) {
              part += -0.5 * (log(2 * pi() * F) + square(v) / F);
            }
            real Kg = Pv / F;
            a += Kg * v;
            Pv -= Kg * Pv;
          }
        }
        total += channel_weight[c] * part;
      }
      // chronic count, observed rows only
      real part_c = 0;
      for (n in row_start:row_end) {
        if (chronic_obs[n] == 1) {
          part_c += neg_binomial_2_log_lpmf(
              chronic_y[n] | dot_product(X[n], coef_chronic[k]), phi_chronic);
        }
      }
      total += channel_weight[C + 1] * part_c;
      class_lp[k] += total;
    }
    return class_lp;
  }

  /** Mortality hazard over the person's whole observed window. */
  vector person_mort_loglik(
      int row_start, int row_end, int K, matrix X,
      array[] int mort_obs, array[] int mort_y, matrix coef_mort,
      real weight) {
    vector[K] out = rep_vector(0.0, K);
    if (row_start == 0 || row_end < row_start) {
      return out;
    }
    for (k in 1:K) {
      real part = 0;
      for (n in row_start:row_end) {
        if (mort_obs[n] == 1) {
          part += bernoulli_logit_lpmf(
              mort_y[n] | dot_product(X[n], coef_mort[k]));
        }
      }
      out[k] = weight * part;
    }
    return out;
  }

  real partial_sum_lpmf(
      array[] int person_slice, int start, int end,
      int K, int C, int P,
      array[] vector y, matrix X,
      int ar_mode, vector age_gap,
      vector log_weight,
      array[] matrix coef, matrix sigma, vector rho, vector sigma_meas,
      array[] int chronic_obs, array[] int chronic_y,
      matrix coef_chronic, real phi_chronic,
      int use_mortality,
      array[] int mort_obs, array[] int mort_y, matrix coef_mort,
      vector channel_weight,
      array[] int fit_start, array[] int fit_end,
      array[] int full_end) {
    real lp = 0;
    for (i in 1:size(person_slice)) {
      int idx = person_slice[i];
      vector[K] class_lp = person_window_loglik(
          fit_start[idx], fit_end[idx], 1, 0, fit_start[idx], K, C, P, y, X,
          ar_mode, age_gap, log_weight, coef, sigma, rho, sigma_meas,
          chronic_obs, chronic_y, coef_chronic, phi_chronic, channel_weight);
      if (use_mortality == 1) {
        class_lp += person_mort_loglik(
            fit_start[idx], full_end[idx], K, X, mort_obs, mort_y, coef_mort,
            channel_weight[C + 2]);
      }
      lp += log_sum_exp(class_lp);
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
  array[N_person] int<lower=1, upper=N_obs> full_end;

  int<lower=0, upper=2> ar_mode;
  vector<lower=0>[ar_mode == 0 ? 0 : N_obs] age_gap;

  array[N_obs] int<lower=0, upper=1> chronic_obs;
  array[N_obs] int<lower=0> chronic_y;
  int<lower=0, upper=1> use_mortality;
  array[N_obs] int<lower=0, upper=1> mort_obs;
  array[N_obs] int<lower=0, upper=1> mort_y;

  int<lower=0, upper=1> homosigma;
  int<lower=1, upper=C> anchor_channel;
  array[C] int<lower=0> channel_n_obs;
  vector<lower=0>[C + 2] channel_weight;

  real alpha_prior_scale;
  vector<lower=0>[P - 1] coef_prior_scale;
  real<lower=0> sigma_prior_location;
  real<lower=0> sigma_prior_scale;
  real<lower=0> theta_prior_concentration;
  real<lower=0> rho_prior_alpha;
  real<lower=0> rho_prior_beta;
  // Half-normal on the measurement-error scale; read only when ar_mode == 2.
  real<lower=0> sigma_meas_prior_location;
  real<lower=0> sigma_meas_prior_scale;
  real chronic_log_mean;
  real mort_logit_mean;

  int<lower=0, upper=1> emit_person_quantities;
  int<lower=1> grainsize;
}

transformed data {
  array[N_person] int person_index;
  for (i in 1:N_person) {
    person_index[i] = i;
  }
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
  // One measurement-error scale per Gaussian channel, shared across classes.
  vector<lower=0.01>[ar_mode == 2 ? C : 0] sigma_meas;

  matrix[K, P] coef_chronic_raw;
  real<lower=0> phi_chronic;
  matrix[use_mortality == 1 ? K : 0, P] coef_mort_raw;
}

transformed parameters {
  array[C] matrix[K, P] coef;
  matrix[K, C] sigma;
  matrix[K, P] coef_chronic = coef_chronic_raw;
  matrix[use_mortality == 1 ? K : 0, P] coef_mort = coef_mort_raw;
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
  for (k in 1:K) {
    coef_chronic[k, 1] = coef_chronic_raw[k, 1] + chronic_log_mean;
  }
  if (use_mortality == 1) {
    for (k in 1:K) {
      coef_mort[k, 1] = coef_mort_raw[k, 1] + mort_logit_mean;
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
  to_vector(sigma_raw) ~ lognormal(log(sigma_prior_location), sigma_prior_scale);
  if (ar_mode != 0) {
    rho ~ beta(rho_prior_alpha, rho_prior_beta);
  }
  if (ar_mode == 2) {
    sigma_meas ~ normal(sigma_meas_prior_location, sigma_meas_prior_scale);
  }
  col(coef_chronic_raw, 1) ~ normal(0, 1);
  for (p in 2:P) {
    col(coef_chronic_raw, p) ~ normal(0, 1);
  }
  if (use_mortality == 1) {
    col(coef_mort_raw, 1) ~ normal(0, 2);
    for (p in 2:P) {
      col(coef_mort_raw, p) ~ normal(0, 1.5);
    }
  }
  phi_chronic ~ gamma(2, 0.1);

  target += reduce_sum(
      partial_sum_lpmf, person_index, grainsize,
      K, C, P, y, X, ar_mode, age_gap, log_weight, coef, sigma, rho,
      sigma_meas, chronic_obs, chronic_y, coef_chronic, phi_chronic,
      use_mortality, mort_obs, mort_y, coef_mort, channel_weight,
      fit_start, fit_end, full_end);
}

generated quantities {
  matrix[emit_person_quantities ? N_person : 0, K] class_prob;
  array[emit_person_quantities ? N_person : 0] int<lower=1, upper=K> modal_class;
  vector[emit_person_quantities ? N_person : 0] log_lik;
  vector[emit_person_quantities ? N_person : 0] log_lik_heldout;
  vector[emit_person_quantities ? N_person : 0] log_lik_heldout_marginal;

  if (emit_person_quantities == 1) {
    for (i in 1:N_person) {
      vector[K] fitted_lp = person_window_loglik(
          fit_start[i], fit_end[i], 1, 0, fit_start[i], K, C, P, y, X,
          ar_mode, age_gap, log_weight, coef, sigma, rho, sigma_meas,
          chronic_obs, chronic_y, coef_chronic, phi_chronic, channel_weight);
      if (use_mortality == 1) {
        fitted_lp += person_mort_loglik(
            fit_start[i], full_end[i], K, X, mort_obs, mort_y, coef_mort,
            channel_weight[C + 2]);
      }

      log_lik[i] = log_sum_exp(fitted_lp);
      class_prob[i] = softmax(fitted_lp)';
      modal_class[i] = sort_indices_desc(fitted_lp)[1];

      if (hold_start[i] > 0) {
        vector[K] hold_lp = person_window_loglik(
            hold_start[i], hold_end[i], 0, fit_end[i], fit_start[i], K, C, P,
            y, X, ar_mode, age_gap, log_weight, coef, sigma, rho, sigma_meas,
            chronic_obs, chronic_y, coef_chronic, phi_chronic, channel_weight);
        vector[K] hold_lp_marg = person_window_loglik(
            hold_start[i], hold_end[i], 0, 0, 0, K, C, P, y, X, ar_mode,
            age_gap, log_weight, coef, sigma, rho, sigma_meas,
            chronic_obs, chronic_y, coef_chronic, phi_chronic, channel_weight);
        log_lik_heldout[i] = log_sum_exp(fitted_lp + hold_lp) - log_lik[i];
        log_lik_heldout_marginal[i] =
            log_sum_exp(fitted_lp + hold_lp_marg) - log_lik[i];
      } else {
        log_lik_heldout[i] = 0;
        log_lik_heldout_marginal[i] = 0;
      }
    }
  }
}
