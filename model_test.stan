
data {
  int<lower=2> K;
  int<lower=0> N;
  int<lower=1> D;
  array[N, K] int<lower=0,upper=100> y;
  array[N] row_vector[D] x;
}

parameters {
  vector[D] beta;
  ordered[K-1] c;
}

model {
  for (n in 1:N) {
    int index = 0;
    real eta = x[n] * beta;
    for (m in y[n]) {
      index += 1;
      target += (ordered_logistic_lupmf(index | eta, c) * m);
    }
  }
  c ~ normal(0, 1);
  beta ~ normal(0, 1);
}

