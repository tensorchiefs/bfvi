#############################################################
library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
library(rstan)
library(loo)

#Loading of cached data
dir_name = 'runs/regression_2D/'
PDF = FALSE
load(file.path(dir_name, 'loss_hist.rda'))
load(file.path(dir_name, 'samples.rda'))
if (PDF) pdf(file.path(dir_name, 'plots.pdf'))
load(file.path(dir_name, 'mcmc_samples.rda'))

evidence = -12.62 # from mathematica (numerical integration)

N = 6L # number data points
P = 2L #number of predictors
D = P + 2L  #Dimension (number of weights) predictors + intercept + sigma
load(file.path(dir_name, '2D.rdata'))

x_r
y_r
if (FALSE){ #Time consuming MCMC
  dl = list(N=N,x=x_r,y=y_r, P = P)
  samples = stan(file="tmvi_multiple_lr_sigma.stan", data=dl, iter = 24000)
  samples
  #traceplot(samples)
  posts_mcmc = extract(samples)
  save(posts_mcmc, file = 'runs/regression_2D/mcmc_samples.rda')
}

#Estimation of k_hat
log_joint = samples$LLs + samples$L_priors
log_ratios = as.vector(log_joint - samples$log_qs)
res_psis = psis(log_ratios)
psis(rnorm(1000, 2, 0.00001))$diagnostics$pareto_k
psis(rnorm(1000, 2, 1e-5))$diagnostics$pareto_k
k_hat = res_psis$diagnostics$pareto_k 
k_hat

#Note here one could use the log-sum-exp trick to get a better estimate
#evidence_est = log(mean( exp(log_joint) / exp(samples$log_qs)  ))
evidence_est = log(mean( exp(log_ratios)  )) #Alternative Version
#evidence_est = pomp::logmeanexp(log_ratios)

plot(loss_hist, type='l', main='Loss function', sub = paste0('k_hat ', round(k_hat,3)), ylim=c(10,30), col='skyblue')
abline(h=-evidence, col='red')
abline(h=-evidence_est,col='darkgreen')
legend('topleft', legend = c('-ELBO', '-Evidence (numerical integration)', '-Evidence (importance sampled)'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')





l = lm(y_r ~ x_r)
summary(l)
w_ml = coef(l)  #Intercept
sd_ml = sqrt(sum(l$residuals^2)/(length(y_r)-2))
logLik(l)

w = samples$w
hist(w[,1], freq = FALSE, 50, xlab='b', col='skyblue', main='VI-Samples for intercept b')
lines(density(posts_mcmc$b), col='red', lwd=2)
lines(density(w[,1]), col='skyblue', lwd=2)
abline(v=w_ml[1], col='darkgreen', lwd=3)
legend('topleft', legend = c('VI', 'MCMC', 'ML'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')

####### 
# For paper
apatheme=theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        legend.title=element_blank(),
        legend.position = c(0.15,0.85),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))


sigma = tf$math$softplus(w[,P+2])$numpy()
TT = 1000
df = data.frame(beta1 = posts_mcmc$w[1:TT,1], 
                beta2=posts_mcmc$w[1:TT,2], 
                sigma=posts_mcmc$sigma[1:TT],
                intercept = posts_mcmc$b[1:TT],
                type='MCMC')
df = rbind(df, data.frame(
  beta1 = w[1:TT,2], 
  beta2=w[1:TT,3], 
  sigma=sigma[1:TT],
  intercept  =w[1:TT,1],
  type='BF-VI'))
library(ggplot2)
library(GGally)
p = ggpairs(df, 
            aes(color = type, alpha=0.5), 
            columns = c(1,2,3,4),
            columnLabels = c('beta1', "beta2", "sigma", 'intercept')
  ) + 
  scale_color_manual(values = c('red', 'blue')) + 
  scale_fill_manual(values = c('red', 'blue'))
p = p + theme_bw() + theme(
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank()
)
p
ggsave(p, filename = 'runs/regression_2D/2D_pairs.pdf', width = 7, height = 7)
#Attention Manual relabeling has been done
#ggsave(p, filename = '~/Dropbox/Apps/Overleaf/bernvi/images/2D_pairs.pdf',width = 7, height=7)

#########  Mean Field Gaussian ########
# Mean field gaussian
# https://colab.research.google.com/drive/1FKnZ-0x5yBlM3-4p1GzQgW35JhfUwxix
b = read_csv("runs/regression_2D/MF_GAUSS_VI/2D_MFG_intercept_samples.csv.gz")
w = read_csv("runs/regression_2D/MF_GAUSS_VI/2D_MFG_w_samples.csv.gz")
s = read_csv("runs/regression_2D/MF_GAUSS_VI/2D_MFG_sigma_samples.csv.gz")
TT = 1000
df = data.frame(beta1 = posts_mcmc$w[1:TT,1], 
                beta2=posts_mcmc$w[1:TT,2], 
                sigma=posts_mcmc$sigma[1:TT],
                intercept = posts_mcmc$b[1:TT],
                type='MCMC')

df = rbind(df, data.frame(
  beta1 = w$w1[1:TT], 
  beta2 = w$w2[1:TT], 
  sigma= s$b[1:TT],
  intercept  = b$b[1:TT],
  type='Gauss-MF'))
library(ggplot2)
library(GGally)
p = ggpairs(df, 
            aes(color = type, alpha=0.5), 
            columns = c(1,2,3,4),
            columnLabels = c('beta1', "beta2", "sigma", 'intercept')
) + 
  scale_color_manual(values = c('red', 'blue')) + 
  scale_fill_manual(values = c('red', 'blue'))
p = p + theme_bw() + theme(
  panel.grid.major=element_blank(),
  panel.grid.minor=element_blank()
)
p
ggsave(p, filename = 'runs/regression_2D/MF_GAUSS_VI/2D.pairs.gauss_vs.mcmc.pdf', width = 7, height = 7)
ggsave(p, filename = '~/Dropbox/Apps/Overleaf/bernvi/images/2D.pairs.gauss_vs.mcmc.pdf',width = 7, height=7)





for (k in 1:P){
  hist(w[,k+1], freq = FALSE, 30, xlab='w', col='skyblue', main=paste0('VI Weight k=', k))
  lines(density(posts_mcmc$w[,k]), col='red', lwd=2)
  abline(v=w_ml[k+1], col='green')
  legend('topleft', legend = c('VI', 'MCMC', 'ML'), lty=c(1,1), 
         col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')
}


hist(sigma, freq = FALSE, 50, xlab='sigma', col='skyblue', main='Samples for sigma')
lines(density(posts_mcmc$sigma), col='red', lwd=2)
lines(density(sigma), col='skyblue', lwd=2)
abline(v=sd_ml, col='darkgreen', lwd=3)
legend('topright', legend = c('VI', 'MCMC', 'ML'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')





plot(density(posts_mcmc$b), main='Intercept MCMC vs ML')
abline(v=w_ml[1], col='green')

for (k in 1:P){
  plot(density(posts_mcmc$w[,k]), main='MCMC vs ML', sub=paste0('Weight of Predictor k=', k))
  abline(v=w_ml[k+1], col='green')
}

plot(density(posts_mcmc$sigma), main='MCMC vs ML sigma')
abline(v=sd_ml, col='green')
if (PDF) dev.off()

