#############################################################
# The non-central parametrisation (easier for MCMC)
# --- EASIER FOR MCMC -----

library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
library(rstan)
library(loo)

#Loading of cached data

dir_name = 'runs/8SCHOOLS_NCP_AS_PAPER/'
PDF = TRUE
load(file.path(dir_name, 'loss_hist.rda'))
load(file.path(dir_name, 'samples.rda'))
if (PDF) pdf(file.path(dir_name, 'plots.pdf'))
load(file.path(dir_name, 'mcmc_samples.rda'))

data <- read_rdump("runs/8SCHOOLS_CP_AS_Paper/eight_schools.data.R")


x_r = data$sigma
y_r = data$y
N = 8L
D = 10L #mu, tau, and 8 Theta_Tilde
P = 1L  #Input dimension of "x" here sigma

mcmc_samples = data.frame(posts_mcmc$mu, posts_mcmc$tau, posts_mcmc$theta)
if (FALSE){
  #For the MCMC we use the easier parametrisation
  m = stan_model(file='eight_schools_ncp.stan')
  fit <- sampling(m, data=data, iter = 10000, seed=194838,control=list(adapt_delta=0.9))
  posts_mcmc = extract(fit)
  save(posts_mcmc, file = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/8SCHOOLS_NCP_Run1/mcmc_samples.rda')

  #Using Stan AUVI
  fit_vb=vb(m,  data=data, iter=1e6,output_samples=1e5,tol_rel_obj=0.001)
  vb_sample=extract(fit_vb)
  vb_samples = data.frame(vb_sample$mu, vb_sample$tau, vb_sample$theta)
  error_mean_sigma(mcmc_samples, vb_samples)  
  #L2_mu   L2_sig 
  #1.892609 1.207052 

  #Second run
  #L2_mu   L2_sig 
  #1.985089 1.227350 
  psis(vb_sample)
}

log_joint = samples$LLs + samples$L_priors
log_ratios = log_joint - samples$log_qs
res_psis = psis(log_ratios = log_ratios)
k_hat = res_psis$diagnostics$pareto_k 
k_hat

#Note here one could use the log-sum-exp trick to get a better estimate
evidence_est = log(mean( exp(log_joint) / exp(samples$log_qs)  ))
evidence_est = log(mean( exp(log_ratios)  )) #Alternative Version

plot(loss_hist, type='l', main='Loss function', sub = paste0('k_hat ', round(k_hat,3), ' ADVI ~ 0.64 '), ylim=c(28,35), col='skyblue')
abline(h=-evidence_est,col='darkgreen')

ww = samples$w
tau_vi = as.numeric(tf$math$exp(0.1*ww[,1]))   #Needs to be positive
hist(log(tau_vi), freq = FALSE, 100)
sam = rcauchy(10000, 0, 5)
sam = sam[sam >= 0]
lines(density(log(sam)), col='green')
lines(density(log(posts_mcmc$tau)), col='red')

mu_vi = ww[,2]
hist(mu_vi, freq = FALSE, 100, xlab='mu')
lines(density(posts_mcmc$mu), col='red')

theta_tilde_vi = ww[,3:10]
par(mfrow=(c(1,2)))
boxplot(as.matrix(posts_mcmc$theta_tilde), main='MCMC Theta_Tilde',ylim=c(-4.5,4.5))
boxplot(as.matrix(theta_tilde_vi), main='VI', ylim=c(-4.5,4.5))
par(mfrow=(c(1,1)))

par(mfrow=(c(1,2)))
boxplot(as.matrix(posts_mcmc$theta), ylim=c(-20,20))
theta_vi = mu_vi + tau_vi * theta_tilde_vi
boxplot(as.matrix(theta_vi), main='VI', ylim=c(-20,20))
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

