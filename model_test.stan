data {
  int<lower=2> K;
  int<lower=0> N_test;
  int<lower=0> N_train;
  int<lower=1> D;
  array[N_train, K] int<lower=0,upper=100> y_train;
  matrix[N_train, K] y_train_2;
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
    vector[N_train] eta_train_2 = x_train * beta;
    matrix[N_train, K - 1] eta_2;
    for (n in 1:N_train) {
      eta_2[n] = to_row_vector(eta_train_2[n] - c);
    }
    matrix[N_train, K - 1] cumprob_2 = inv_logit(eta_2);
    matrix[N_train, K] v1_2 = append_col(cumprob_2, rep_matrix(0, N_train, 1));
    matrix[N_train, K] v2_2 = append_col(rep_matrix(1, N_train, 1), cumprob_2);
    matrix[N_train, K] lpmfs = log(v2_2 - v1_2);
    matrix[N_train, K] likelihood_2 = lpmfs .* y_train_2;
    real total_p_2 = sum(likelihood_2);
    target += total_p_2;

    //print(eta_train_2);
    //print(v1_2);

    //for (n in 1:N_train) {
    //  int index = 0;
    //  real eta_train = x_train[n] * beta;

    //  vector[K - 1] cumprob = inv_logit(eta_train - c);
    //  vector[K] v1;
    //  vector[K] v2;
    //  v1[1:K-1] = cumprob;
    //  v1[K] = 0;
    //  v2[2:K] = cumprob;
    //  v2[1] = 1;
    //  vector[K] pdf = v2 - v1;
    //  vector[K] lpdf = log(pdf);
    //  vector[K] test = to_vector(y_train_2[n]);
    //  vector[K] likelihood = lpdf .* test;
    //  real total_p = sum(likelihood);

    //  target += total_p;
      //print(c);
      //print(eta_train);
      //print(v1);
      //print(v2);
      //print(cumprob);
      //print(lpdf);
      //print(likelihood);

      //for (m in y_train[n]) {
      //  index += 1;
      //  real inc_val = (ordered_logistic_lupmf(index | eta_train, c) * m);
      //  print(inc_val);
      //  target += inc_val;
      //}
    //}
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