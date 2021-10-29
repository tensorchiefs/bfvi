x_r = c(-5.41239, -4.142973,-5.100401,-4.588446,-2.0570369,-2.0035906,-3.7404475,
        -5.344997,4.506781,5.9761415,4.073539,5.168227,4.1196156,2.4791312,2.0348845, 2.7495284)
y_r = c(0.973122  ,  0.96644104,  1.2311585 ,  0.5193988 , -1.2059958 ,
        -0.9434611 ,  0.8041748 ,  0.82996416, -1.3704962 , -0.3733918 ,
        -0.98566836, -1.1550032 , -1.0276004 ,  0.539029  ,  1.5336514 ,
        0.34641847)

x_r = x_r[1:9]
y_r = y_r[1:9]


plot(x_r, y_r)
N = length(x_r)
P = 1L
D = 10L
num_hidden_units = 5
x = matrix(x_r, ncol=1)
sigma = 0.2
num_xx=500
xx = matrix(seq(-11, 11, length.out = num_xx), ncol=1)


my_dat = list(
  N = N, #'N': num,
  d = 1, #'d': 1,
  num_nodes = 6, #'num_nodes': num_hidden_units,
  num_middle_layers = 1, #'num_middle_layers':1,
  X = x,#'X': x,
  sigma = sigma,#'sigma': sigma,
  Nt = num_xx, #'Nt': num_xx,
  Xt = xx, #'Xt': xx,
  y = y_r#'y': y}
)

library(rstan)
options(mc.cores = parallel::detectCores())
model = stan_model(file="network.stan")
mcmc_samples = sampling(model, data=my_dat, iter = 8000)

preds = extract(mcmc_samples, 'predictions')$predictions
y_med = apply(preds, 2, quantile, probs=0.5)
y_l = apply(preds, 2, quantile, probs=0.05)
y_u = apply(preds, 2, quantile, probs=1-0.05)

plot(xx, y_med, ylim=c(-2,3), type='l', main='Num nodes 6')
lines(xx, y_l)
lines(xx, y_u)
points(x_r,y_r)

save(mcmc_samples, file='~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/NETWORK_only9_Data/mcmc_samples.rda')

