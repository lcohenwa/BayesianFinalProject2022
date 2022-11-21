data {
  int<lower=2> K;
  int<lower=0> N_test;
  int<lower=0> N_train;
  int<lower=1> D;
  array[N_train, K] int<lower=0,upper=100> y_train;
  array[N_test] row_vector[D] x_test;
  array[N_train] row_vector[D] x_train;
  int do_prior_predictive;
}

parameters {
  vector[D] beta;
  ordered[K-1] c;
}

model {
  c ~ normal(0, 2.3);
  beta ~ normal(0, 1);
  if (do_prior_predictive != 1) {
    for (n in 1:N_train) {
      int index = 0;
      real eta_train = x_train[n] * beta;
      for (m in y_train[n]) {
        index += 1;
        target += (ordered_logistic_lupmf(index | eta_train, c) * m);
      }
    }
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