data {
  int<lower=0> N_test;
  int<lower=0> N_train;
  int<lower=0> D;   // number of predictors

  matrix[N_train, D] x_train;   // predictor matrix
  matrix[N_test, D] x_test;   // predictor matrix

  vector[N_train] y_train;      // outcome vector
  vector[N_test] y_test;      // outcome vector
}

parameters {
  real alpha;           // intercept
  vector[D] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}

model {
  y_train ~ normal(alpha + x_train * beta, sigma);  // likelihood
  // the following doesn't work: { alpha, beta, sigma } ~ normal(0, 1);
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ lognormal(0, 1);
}

generated quantities {
  vector[N_train + N_test] y_test_hat = normal_rng(alpha + x_test * beta, sigma);
  vector[N_train + N_test] err = y_test_sim - y_hat; // not sure if this will work
}