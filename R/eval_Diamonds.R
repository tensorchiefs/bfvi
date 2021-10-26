#############################################################
library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
library(rstan)
library(loo)
library(rstan)

#Loading of cached data
dir_name = 'runs/Diamonds'
PDF = FALSE
load(file.path(dir_name, 'loss_hist.rda'))
load(file.path(dir_name, 'samples.rda'))
if (PDF) pdf(file.path(dir_name, 'plots.pdf'))

if (FALSE){
  # we used provided MCMC samples
  library(posteriordb)
  library(posterior)
  my_pdb <- pdb_local(path='~/Documents/workspace/posteriordb')
  po <- posteriordb::posterior("diamonds-diamonds", my_pdb)
  draws = get_data(po)
  reference_posterior_draws_info(po)  
  posteriordb::stan_code(po)
  rpd <- reference_posterior_draws(po)
  posterior::summarize_draws(rpd)
  #https://mc-stan.org/cmdstanr/reference/fit-method-draws.html
  chef_mcmc = as.data.frame(as_draws_df(rpd))[,1:26]
}
load(file="runs/Diamonds/chef_mcmc.rda")

load('runs/Diamonds/diamonds.rda')

N = data$N # number data points
P = as.integer(data$K - 1L) #K contains the intercept
Xc = scale(data$X[,-1], center = TRUE, scale = FALSE)
Y = data$Y
ml = lm(Y ~ Xc, data.frame(Y = data$Y, Xc = Xc))
summary(ml)

#Estimation of k_hat
log_joint = samples$LLs + samples$L_priors
log_ratios = log_joint - samples$log_qs
res_psis = psis(log_ratios = log_ratios)
k_hat = res_psis$diagnostics$pareto_k 
k_hat

mean_and_check_mc_error = function(a, atol=0.01, rtol=0.0){
  m = mean(a)
  s = sd(a)/sqrt(length(a))
  if (s > rtol*abs(m) + atol){
    print('There is something foule in the state of denmark')
  }
  return (m)
}

#Note here one could use the log-sum-exp trick to get a better estimate
evidence_est = log(mean( exp(log_joint) / exp(samples$log_qs)  ))
evidence_est = log(mean( exp(log_ratios)  )) #Alternative Version
evidence_est = pomp::logmeanexp(log_ratios)

### Loss
plot(loss_hist, type='l', main='Loss function', 
     sub = paste0('k_hat ', round(k_hat,3)), 
     col='skyblue', ylim=c(-1e4,1e4))
d = lowess(loss_hist, delta = 20)
lines(d)
abline(h=-evidence_est,col='darkgreen')
legend('topleft', legend = c('-ELBO', '-Evidence (numerical integration)', '-Evidence (importance sampled)'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')

### Intercept
w = samples$w
hist(w[,1], freq = FALSE, 50, xlab='b', col='skyblue', main='VI-Samples for intercept b')
lines(density(posts_mcmc$b), col='red', lwd=2)
lines(density(chef_mcmc$Intercept), col='pink', lwd=2)
#hist(chef_mcmc$Intercept, xlim=c(5, 10))
#lines(density(w[,1]), col='skyblue', lwd=2)
abline(v=ml$coefficients[1], col='darkgreen', lwd=3)
legend('topleft', legend = c('VI', 'MCMC', 'ML'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')

car::qqPlot(chef_mcmc$Intercept, main='MCMC Intercept')
car::qqPlot(chef_mcmc$sigma, main='MCMC Sigma')
for (i in 1:14){
  car::qqPlot(chef_mcmc[,i], main=paste0('MCMC b ', i))
}
cor(chef_mcmc)

par(mfrow = c(2,1))
boxplot(w[,1:26], ylim=c(-10,15), main='VI') 
points(ml$coefficients[1:25], col='red')
d = cbind(chef_mcmc$Intercept, chef_mcmc[1:24], chef_mcmc$sigma)
boxplot(d, ylim=c(-10,15), main='MCMC') 
points(ml$coefficients[1:25], col='red')
par(mfrow = c(1,1))


sigma = tf$math$softplus(w[,P+2])$numpy()
hist(sigma, freq = FALSE, 50, xlab='sigma', col='skyblue', main='Samples for sigma')
lines(density(posts_mcmc$sigma), col='red', lwd=2)
lines(density(sigma), col='skyblue', lwd=2)
lines(density(chef_mcmc$sigma), col='pink', lwd=2)
abline(v=sd_ml, col='darkgreen', lwd=3)
legend('topright', legend = c('VI', 'MCMC', 'ML'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')







if (PDF) dev.off()

