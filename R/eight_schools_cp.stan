//https://github.com/betanalpha/knitr_case_studies/blob/master/rstan_workflow/eight_schools_cp.stan
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real theta[J];
}

model {
  //Priors for p(mu, tau, theta)
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  //Likelihood
  y ~ normal(theta, sigma);
}