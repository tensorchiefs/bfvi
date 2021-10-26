#############################################################
# The center parametrisation (harder for MCMC)
# --- HARDER FOR MCMC -----

library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
library(rstan)
#source('eval_utils.R')

#Loading of cached data
dir_name = 'runs/8SCHOOLS_CP_AS_Paper'
PDF = TRUE
load(file.path(dir_name, 'loss_hist.rda'))
load(file.path(dir_name, 'samples.rda'))
if (PDF) pdf(file.path(dir_name, 'plots.pdf'))


load('runs/8SCHOOLS_CP_AS_Paper/mcmc_samples.rda')
#pdf("/runs/8Schools_CP_run1/plots.pdf")
data <- read_rdump("runs/8SCHOOLS_CP_AS_Paper/eight_schools.data.R")
x_r = data$sigma
y_r = data$y
N = 8L
D = 10L #mu, tau, and 8 Theta_Tilde
P = 1L  #Input dimension of "x" here sigma


mcmc_samples = data.frame(posts_mcmc$mu, posts_mcmc$tau, posts_mcmc$theta)
if (FALSE){
  #For the MCMC we use the easier parametrisation
  fit <- stan(file='eight_schools_ncp.stan', data=data, iter = 10000, seed=194838,control=list(adapt_delta=0.9))
  posts_mcmc = extract(fit)
  save(posts_mcmc, file = 'runs/8SCHOOLS_CP_AS_Paper/mcmc_samples.rda')
}

library(loo)
log_joint = samples$LLs + samples$L_priors
log_ratios = log_joint - samples$log_qs
res_psis = psis(log_ratios = log_ratios)
k_hat = res_psis$diagnostics$pareto_k 
k_hat

#plot(res_psis)
ww = samples$w
#Note here one could use the log-sum-exp trick to get a better estimate
evidence_est = log(mean( exp(log_joint) / exp(samples$log_qs)  ))
evidence_est = log(mean( exp(log_ratios)  )) #Alternative Version


log_ratios_psis = weights(res_psis)
evidence_est_psis = log(mean( exp(log_ratios_psis)  )) #Alternative Version
plot(log_ratios, log_ratios_psis)
#evidence_est_psis = log(mean(exp(weights(res_psis))))

# error_mean_sigma(mcmc_samples, ww)

plot(loss_hist, type='l', main='Loss function', sub = paste0('k_hat ', round(k_hat,3), ' ADVI ~ 1.0 '), ylim=c(30,50), col='skyblue')
abline(h=-evidence_est,col='darkgreen')

tau_vi = as.numeric(tf$math$exp(0.1*ww[,1]))   #Needs to be positive
hist(log(tau_vi), freq = FALSE, 100)
sam = rcauchy(10000, 0, 5)
sam = sam[sam >= 0]
lines(density(log(sam)), col='green')
lines(density(log(posts_mcmc$tau)), col='red')

hist(ww[,2], freq = FALSE, 100, xlab='mu')
lines(density(posts_mcmc$mu), col='red')

theta_vi = ww[,3:D]
par(mfrow=(c(1,2)))
boxplot(posts_mcmc$theta, ylim=c(-20.5,20.5), main='MCMC')
boxplot(as.matrix(theta_vi), main='VI', ylim=c(-20.5,20.5))
par(mfrow=(c(1,1)))

DF = data.frame(log_tau = log(posts_mcmc$tau), theta_1 = posts_mcmc$theta[,1]) 
DF$method = 'MCMC'
DF[1:6000,] %>% 
  ggplot(aes(x=log_tau, y=theta_1)) + 
  geom_point(size=0.01) + 
  geom_hex(alpha=0.5)+ ggtitle('MCMC')

DF2 = data.frame(log_tau = log(tau_vi), theta_1=theta_vi[,1], method='VI')
DF2[1:6000,] %>% 
  ggplot(aes(x=log_tau, y=theta_1)) + 
  geom_point(size=0.01) + 
  geom_hex(alpha=0.5) + ggtitle('VI')


if (PDF) dev.off()

