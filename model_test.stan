data {
  int<lower=2> K;
  int<lower=0> N_test;
  int<lower=0> N_train;
  int<lower=1> D;
  matrix[N_train, K] y_train;
  matrix[N_test, D] x_test;
  matrix[N_train, D] x_train;
  int do_prior_predictive;
}

parameters {
  vector[D] beta;
  ordered[K-1] c;
}

model {
  c ~ normal(0, 1.8);
  beta ~ normal(0, 1);
  if (do_prior_predictive != 1) {
    vector[N_train] eta_1 = x_train * beta;
    matrix[N_train, K - 1] eta_2;
    for (k in 1:K-1) {
      eta_2[, k] = eta_1 - c[k];
    }
    matrix[N_train, K - 1] cumprob = inv_logit(eta_2);
    matrix[N_train, K] vec_1 = append_col(cumprob, rep_matrix(0, N_train, 1));
    matrix[N_train, K] vec_2 = append_col(rep_matrix(1, N_train, 1), cumprob);
    matrix[N_train, K] lpmfs = log(vec_2 - vec_1);
    matrix[N_train, K] likelihood = lpmfs .* y_train;
    real total_p = sum(likelihood);
    target += total_p;
  }
}

generated quantities {
  array[N_train] int<lower=0,upper=100> y_train_tilde;
  for (n in 1:N_train) {
    real eta_train_tilde = x_train[n] * beta;
    y_train_tilde[n] = ordered_logistic_rng(eta_train_tilde, c);
  }
  array[N_test] int<lower=0,upper=100> y_test;
  for (n in 1:N_test) {
    real eta_test = x_test[n] * beta;
    y_test[n] = ordered_logistic_rng(eta_test, c);
    }
}