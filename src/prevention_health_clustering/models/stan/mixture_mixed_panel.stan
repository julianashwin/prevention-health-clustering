/**
 * Multidimensional latent-class model: five health channels, one shared class.
 *
 *   pcs      Gaussian on the standardised score (higher = better health)
 *   mcs      Gaussian on the standardised score
 *   srh      self-rated health 1..5, ordered logit (higher = worse)
 *   chronic  chronic-condition count, negative binomial
 *   adl      ADL/IADL limitation count, hurdle negative binomial
 *            (observed on ~5% of rows - a scheduled sub-module - so the
 *            ragged per-channel indexing below is essential, not cosmetic)
 *
 * Channels are INDEPENDENTLY ragged: each has its own value/design arrays and
 * per-person [start, end] windows into them, with start = 0 meaning the person
 * has no rows for that channel. Setting a channel's N to zero disables it
 * entirely; `transformed data` rejects a configuration whose anchor channel
 * carries no observations, so a masked fit can never sample with exchangeable
 * labels (the defect behind the predecessor's mcs_only special case).
 *
 * Design choices carried from the rebuild manifest:
 *  - No tanh caps and no bounded_mean clamp. The predecessor's quarantined
 *    fit shows alpha_pcs[3] = 1.4994 against a 1.5 cap and phi_chronic = 19.99
 *    against an upper bound of 20 - two silently binding bounds. Soft priors
 *    plus wide hard guards replace them.
 *  - SRH cutpoints are positive gaps, mean-centred (the eight-domain fix), so
 *    the latent location ridge between eta and the cutpoints is removed
 *    structurally rather than pinned by a tight prior.
 *  - Channel weights are DATA with a default of one. The predecessor applied
 *    a bare 0.5 to the ADL term in two of four programs; downstream scoring
 *    hard-coded it regardless of which program produced the draws.
 *  - The per-person class log-likelihood is written once and used by both the
 *    sampler and the (gated) generated quantities.
 *
 * Orientation: class 1 has the lowest anchor-channel intercept. With PCS as
 * the anchor on the raw standardised scale, class 1 = worst physical health,
 * matching mixture_gaussian_panel.stan. This is deliberately the opposite of
 * the predecessor's mixed-health fits, which sign-flipped the scores so that
 * class 1 was best; the repo carries ONE convention, recorded in metadata.
 *
 * The survival/mortality channel is deliberately absent from this version;
 * the five-channel and six-outcome families were separate lanes in the study
 * design, and the six-outcome variant needs the discrete-time interval file.
 */

functions {
  real hurdle_neg_binomial_2_log_lpmf(int y, real zero_logit,
                                      real log_mu, real phi) {
    // zero_logit is the log-odds of ANY limitation (y > 0).
    if (y == 0) {
      return bernoulli_logit_lpmf(0 | zero_logit);
    }
    return bernoulli_logit_lpmf(1 | zero_logit)
           + neg_binomial_2_log_lpmf(y | log_mu, phi)
           - log1m_exp(neg_binomial_2_log_lpmf(0 | log_mu, phi));
  }

  vector person_class_loglik_mixed(
      int i, int K,
      // gaussian channels: values, age design, windows
      int C_gauss,
      array[] vector y_gauss, array[] matrix X_gauss,
      array[,] int gauss_start, array[,] int gauss_end,
      array[] matrix coef_gauss,     // C_gauss items, each K x P
      vector sigma_gauss,            // C_gauss
      // srh
      array[] int srh_y, matrix X_srh,
      array[] int srh_start, array[] int srh_end,
      matrix coef_srh,               // K x P
      vector cut_srh,
      // chronic
      array[] int chronic_y, matrix X_chronic,
      array[] int chronic_start, array[] int chronic_end,
      matrix coef_chronic, real phi_chronic,
      // adl
      array[] int adl_y, matrix X_adl,
      array[] int adl_start, array[] int adl_end,
      matrix coef_adl_zero, matrix coef_adl_count, real phi_adl,
      vector log_weight, vector channel_weight, int P) {
    vector[K] class_lp = log_weight;
    for (k in 1:K) {
      real total = 0;
      for (c in 1:C_gauss) {
        if (gauss_start[c, i] > 0) {
          real part = 0;
          for (n in gauss_start[c, i]:gauss_end[c, i]) {
            part += normal_lpdf(y_gauss[c][n]
                                | dot_product(X_gauss[c][n], coef_gauss[c][k]),
                                sigma_gauss[c]);
          }
          total += channel_weight[c] * part;
        }
      }
      if (srh_start[i] > 0) {
        real part = 0;
        for (n in srh_start[i]:srh_end[i]) {
          part += ordered_logistic_lpmf(srh_y[n]
                                        | dot_product(X_srh[n], coef_srh[k]),
                                        cut_srh);
        }
        total += channel_weight[C_gauss + 1] * part;
      }
      if (chronic_start[i] > 0) {
        real part = 0;
        for (n in chronic_start[i]:chronic_end[i]) {
          part += neg_binomial_2_log_lpmf(chronic_y[n]
                                          | dot_product(X_chronic[n], coef_chronic[k]),
                                          phi_chronic);
        }
        total += channel_weight[C_gauss + 2] * part;
      }
      if (adl_start[i] > 0) {
        real part = 0;
        for (n in adl_start[i]:adl_end[i]) {
          part += hurdle_neg_binomial_2_log_lpmf(adl_y[n]
                    | dot_product(X_adl[n], coef_adl_zero[k]),
                      dot_product(X_adl[n], coef_adl_count[k]),
                      phi_adl);
        }
        total += channel_weight[C_gauss + 3] * part;
      }
      class_lp[k] += total;
    }
    return class_lp;
  }

  real partial_sum_lpmf(
      array[] int person_slice, int start, int end,
      int K, int C_gauss,
      array[] vector y_gauss, array[] matrix X_gauss,
      array[,] int gauss_start, array[,] int gauss_end,
      array[] matrix coef_gauss, vector sigma_gauss,
      array[] int srh_y, matrix X_srh,
      array[] int srh_start, array[] int srh_end,
      matrix coef_srh, vector cut_srh,
      array[] int chronic_y, matrix X_chronic,
      array[] int chronic_start, array[] int chronic_end,
      matrix coef_chronic, real phi_chronic,
      array[] int adl_y, matrix X_adl,
      array[] int adl_start, array[] int adl_end,
      matrix coef_adl_zero, matrix coef_adl_count, real phi_adl,
      vector log_weight, vector channel_weight, int P) {
    real lp = 0;
    for (idx in 1:size(person_slice)) {
      lp += log_sum_exp(person_class_loglik_mixed(
          person_slice[idx], K, C_gauss,
          y_gauss, X_gauss, gauss_start, gauss_end, coef_gauss, sigma_gauss,
          srh_y, X_srh, srh_start, srh_end, coef_srh, cut_srh,
          chronic_y, X_chronic, chronic_start, chronic_end,
          coef_chronic, phi_chronic,
          adl_y, X_adl, adl_start, adl_end,
          coef_adl_zero, coef_adl_count, phi_adl,
          log_weight, channel_weight, P));
    }
    return lp;
  }
}

data {
  int<lower=1> N_person;
  int<lower=1> K;
  int<lower=2> P;
  int<lower=1> C_gauss;                       // continuous channels (pcs, mcs)
  int<lower=1, upper=C_gauss> anchor_channel; // index into the gaussian block

  // gaussian channels, each independently ragged
  array[C_gauss] int<lower=0> N_gauss;
  array[C_gauss] vector[max(N_gauss)] y_gauss_pad;
  array[C_gauss] matrix[max(N_gauss), P] X_gauss_pad;
  array[C_gauss, N_person] int<lower=0> gauss_start;
  array[C_gauss, N_person] int<lower=0> gauss_end;

  int<lower=0> N_srh;
  array[N_srh] int<lower=1, upper=5> srh_y;
  matrix[N_srh, P] X_srh;
  array[N_person] int<lower=0> srh_start;
  array[N_person] int<lower=0> srh_end;

  int<lower=0> N_chronic;
  array[N_chronic] int<lower=0> chronic_y;
  matrix[N_chronic, P] X_chronic;
  array[N_person] int<lower=0> chronic_start;
  array[N_person] int<lower=0> chronic_end;

  int<lower=0> N_adl;
  array[N_adl] int<lower=0> adl_y;
  matrix[N_adl, P] X_adl;
  array[N_person] int<lower=0> adl_start;
  array[N_person] int<lower=0> adl_end;

  // one weight per channel block: C_gauss gaussians, then srh, chronic, adl
  vector<lower=0>[C_gauss + 3] channel_weight;

  // priors
  real<lower=0> alpha_prior_scale;
  vector<lower=0>[P - 1] coef_prior_scale;
  real<lower=0> sigma_prior_location;
  real<lower=0> sigma_prior_scale;
  real<lower=0> theta_prior_concentration;
  real chronic_log_mean;                      // empirical offset, from data
  real adl_zero_logit;                        // empirical any-limitation logit
  real adl_log_positive_mean;

  int<lower=0, upper=1> emit_person_quantities;
  int<lower=1> grainsize;
}

transformed data {
  array[N_person] int person_index;
  for (i in 1:N_person) {
    person_index[i] = i;
  }
  if (N_gauss[anchor_channel] == 0) {
    reject("Anchor channel ", anchor_channel,
           " has no observations: class labels would be exchangeable.");
  }
}

parameters {
  simplex[K] theta;
  ordered[K] anchor_intercept;
  matrix[K, C_gauss - 1] other_intercept;
  array[C_gauss] matrix[K, P - 1] slope_gauss;
  vector<lower=0.05>[C_gauss] sigma_gauss;

  matrix[K, P] coef_srh_raw;
  vector<lower=0>[3] cut_srh_gap;             // 4 centred cutpoints from 3 gaps

  matrix[K, P] coef_chronic_raw;
  real<lower=0.1> phi_chronic;

  matrix[K, P] coef_adl_zero_raw;
  matrix[K, P] coef_adl_count_raw;
  real<lower=0.1> phi_adl;
}

transformed parameters {
  array[C_gauss] matrix[K, P] coef_gauss;
  matrix[K, P] coef_srh = coef_srh_raw;
  matrix[K, P] coef_chronic = coef_chronic_raw;
  matrix[K, P] coef_adl_zero = coef_adl_zero_raw;
  matrix[K, P] coef_adl_count = coef_adl_count_raw;
  vector[4] cut_srh;
  vector[K] log_weight = log(theta);

  for (c in 1:C_gauss) {
    for (k in 1:K) {
      coef_gauss[c][k, 1] = c == anchor_channel
          ? anchor_intercept[k]
          : other_intercept[k, c > anchor_channel ? c - 1 : c];
      for (p in 2:P) {
        coef_gauss[c][k, p] = slope_gauss[c][k, p - 1];
      }
    }
  }
  // Count-channel intercepts sit on empirical offsets so their priors are
  // centred where the data live rather than at zero.
  for (k in 1:K) {
    coef_chronic[k, 1] += chronic_log_mean;
    coef_adl_zero[k, 1] += adl_zero_logit;
    coef_adl_count[k, 1] += adl_log_positive_mean;
  }
  // Positive gaps + mean-centring: ordering is automatic and the location
  // ridge between eta and the cutpoints is structurally removed.
  {
    vector[4] cumulative;
    cumulative[1] = 0;
    for (g in 1:3) {
      cumulative[g + 1] = cumulative[g] + cut_srh_gap[g];
    }
    cut_srh = cumulative - mean(cumulative);
  }
}

model {
  theta ~ dirichlet(rep_vector(theta_prior_concentration, K));
  anchor_intercept ~ normal(0, alpha_prior_scale);
  to_vector(other_intercept) ~ normal(0, alpha_prior_scale);
  for (c in 1:C_gauss) {
    for (p in 1:(P - 1)) {
      col(slope_gauss[c], p) ~ normal(0, coef_prior_scale[p]);
    }
  }
  sigma_gauss ~ normal(sigma_prior_location, sigma_prior_scale);

  col(coef_srh_raw, 1) ~ normal(0, 2.5);
  for (p in 2:P) {
    col(coef_srh_raw, p) ~ normal(0, coef_prior_scale[p - 1]);
  }
  cut_srh_gap ~ lognormal(log(2.0), 0.5);

  col(coef_chronic_raw, 1) ~ normal(0, 1.5);
  col(coef_adl_zero_raw, 1) ~ normal(0, 2.0);
  col(coef_adl_count_raw, 1) ~ normal(0, 1.5);
  for (p in 2:P) {
    col(coef_chronic_raw, p) ~ normal(0, coef_prior_scale[p - 1]);
    col(coef_adl_zero_raw, p) ~ normal(0, coef_prior_scale[p - 1]);
    col(coef_adl_count_raw, p) ~ normal(0, coef_prior_scale[p - 1]);
  }
  phi_chronic ~ lognormal(log(10), 1);
  phi_adl ~ lognormal(log(5), 1);

  target += reduce_sum(
      partial_sum_lpmf, person_index, grainsize,
      K, C_gauss,
      y_gauss_pad, X_gauss_pad, gauss_start, gauss_end, coef_gauss, sigma_gauss,
      srh_y, X_srh, srh_start, srh_end, coef_srh, cut_srh,
      chronic_y, X_chronic, chronic_start, chronic_end,
      coef_chronic, phi_chronic,
      adl_y, X_adl, adl_start, adl_end,
      coef_adl_zero, coef_adl_count, phi_adl,
      log_weight, channel_weight, P);
}

generated quantities {
  matrix[emit_person_quantities ? N_person : 0, K] class_prob;
  vector[emit_person_quantities ? N_person : 0] log_lik;

  if (emit_person_quantities == 1) {
    for (i in 1:N_person) {
      vector[K] lp = person_class_loglik_mixed(
          i, K, C_gauss,
          y_gauss_pad, X_gauss_pad, gauss_start, gauss_end,
          coef_gauss, sigma_gauss,
          srh_y, X_srh, srh_start, srh_end, coef_srh, cut_srh,
          chronic_y, X_chronic, chronic_start, chronic_end,
          coef_chronic, phi_chronic,
          adl_y, X_adl, adl_start, adl_end,
          coef_adl_zero, coef_adl_count, phi_adl,
          log_weight, channel_weight, P);
      log_lik[i] = log_sum_exp(lp);
      class_prob[i] = softmax(lp)';
    }
  }
}
