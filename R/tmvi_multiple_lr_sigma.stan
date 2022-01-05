data {
  int<lower=0> N;
  int<lower=1> P;
  vector[N] y;
  matrix[N,P] x;
}

parameters {
  vector[P] w;
  real b;
  real<lower=0> sigma;
}

model {
  y ~ normal(x * w + b, sigma);
  b ~ normal(0,10);
  w ~ normal(0, 10);
  sigma ~ lognormal(0.5,1);
}



