#############################################################
# Creating the 1-D NN Regression figure

library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
library(rstan)
library(loo)
library(rstan)
#Loading of cached data
dir_name = 'runs/network'
PDF = FALSE
load(file.path(dir_name, 'loss_hist.rda'))
load(file.path(dir_name, 'samples.rda'))
if (PDF) pdf(file.path(dir_name, 'plots.pdf'))

#Reading of the MCMC samples from the python 
post_mcmc = read.csv(file.path(dir_name,'mcmc_y_quantiles.csv'))
post_mcmc$X = NULL
names(post_mcmc) = c('X','Mean', 'qlow', 'qhigh', 'med')

y_preds_MFGauss = read.csv(file.path(dir_name,'y_preds_VIGAUSS.csv'), header=FALSE)
y_preds_MFTM = read.csv(file.path(dir_name,'y_preds_VIMLTS.csv'), header=FALSE)
xx = read.csv(file.path(dir_name,'xx.csv'), header=FALSE)
dim(xx)
xx = xx[,1]

P = 1L
x_r = c(-5.41239, -4.142973,-5.100401,-4.588446,-2.0570369,-2.0035906,-3.7404475,
        -5.344997,4.506781,5.9761415,4.073539,5.168227,4.1196156,2.4791312,2.0348845, 2.7495284)
y_r = c(0.973122  ,  0.96644104,  1.2311585 ,  0.5193988 , -1.2059958 ,
        -0.9434611 ,  0.8041748 ,  0.82996416, -1.3704962 , -0.3733918 ,
        -0.98566836, -1.1550032 , -1.0276004 ,  0.539029  ,  1.5336514 ,
        0.34641847)
x_r = x_r[1:9]
y_r = y_r[1:9]
N = length(x_r) # number data points
plot(x_r, y_r)
N = length(x_r)
P = 1L
D = 10L
sigma=0.2

#Estimation of k_hat
log_joint = samples$LLs + samples$L_priors
log_ratios = log_joint - samples$log_qs
library(loo)
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
     col='skyblue', ylim=c(-100, 300))
abline(h=-evidence_est,col='darkgreen')
legend('topleft', legend = c('-ELBO', '-Evidence (numerical integration)', '-Evidence (importance sampled)'), lty=c(1,1), 
       col=c('skyblue', 'red', 'darkgreen'), lwd=3, bty='n')

w = samples$w

# get the log-likelihood and the prior
get_post_predictive <- function(w, x) {
  w = tf$Variable(w, dtype='float32')
  x = tf$Variable(x, dtype='float32')
  #Determination of the likelihood
  w_11 =  tf$slice(w,c(0L,0L),c(T,1L))  
  b_11 =  tf$slice(w,c(0L,1L),c(T,1L))   
  w_12 =  tf$slice(w,c(0L,2L),c(T,1L))  
  b_12 =  tf$slice(w,c(0L,3L),c(T,1L))
  w_13 =  tf$slice(w,c(0L,4L),c(T,1L))  
  b_13 =  tf$slice(w,c(0L,5L),c(T,1L))
  #See https://stackoverflow.com/questions/33858021/outer-product-in-tensorflow
  x_rep = tf$reshape(x, shape=c(N,1L))
  
  w_11_rep = tf$reshape(w_11, shape=c(1L, T))
  h_1 = tf$sigmoid(x_rep * w_11_rep + tf$reshape(b_11, shape=c(1L,T)))
  w_12_rep = tf$reshape(w_12, shape=c(1L, T))
  h_2 = tf$sigmoid(x_rep * w_12_rep + tf$reshape(b_12, shape=c(1L,T)))
  w_13_rep = tf$reshape(w_13, shape=c(1L, T))
  h_3 = tf$sigmoid(x_rep * w_13_rep + tf$reshape(b_13, shape=c(1L,T)))
  
  w_21 =  tf$slice(w,c(0L,6L),c(T,1L))  
  w_22 =  tf$slice(w,c(0L,7L),c(T,1L))  
  w_23 =  tf$slice(w,c(0L,8L),c(T,1L))  
  b_2 =  tf$slice(w,c(0L,9L),c(T,1L))
  
  mu = h_1 * tf$squeeze(w_21)  + h_2 * tf$squeeze(w_22) + h_3 * tf$squeeze(w_23) + tf$squeeze(b_2)
  mu = tf$transpose(mu) #oups did it wrong way
  return (tfp$distributions$Normal(loc=mu, scale=sigma)$sample())
}

# get the log-likelihood and the prior
get_post_predictive_6 <- function(w, x) {
  w = tf$Variable(w, dtype='float32')
  x = tf$Variable(x, dtype='float32')
  #Determination of the likelihood
  w_11 =  tf$slice(w,c(0L,0L),c(T,1L))  
  b_11 =  tf$slice(w,c(0L,1L),c(T,1L))   
  w_12 =  tf$slice(w,c(0L,2L),c(T,1L))  
  b_12 =  tf$slice(w,c(0L,3L),c(T,1L))
  w_13 =  tf$slice(w,c(0L,4L),c(T,1L))  
  b_13 =  tf$slice(w,c(0L,5L),c(T,1L))
  w_14 =  tf$slice(w,c(0L,6L),c(T,1L))  
  b_14 =  tf$slice(w,c(0L,7L),c(T,1L))   
  w_15 =  tf$slice(w,c(0L,8L),c(T,1L))  
  b_15 =  tf$slice(w,c(0L,9L),c(T,1L))
  w_16 =  tf$slice(w,c(0L,10L),c(T,1L))  
  b_16 =  tf$slice(w,c(0L,11L),c(T,1L))
  
  #See https://stackoverflow.com/questions/33858021/outer-product-in-tensorflow
  x_rep = tf$reshape(x, shape=c(N,1L))
  
  w_11_rep = tf$reshape(w_11, shape=c(1L, T))
  w_12_rep = tf$reshape(w_12, shape=c(1L, T))
  w_13_rep = tf$reshape(w_13, shape=c(1L, T))
  w_14_rep = tf$reshape(w_14, shape=c(1L, T))
  w_15_rep = tf$reshape(w_15, shape=c(1L, T))
  w_16_rep = tf$reshape(w_16, shape=c(1L, T))
  
  h_1 = tf$sigmoid(x_rep * w_11_rep + tf$reshape(b_11, shape=c(1L,T)))
  h_2 = tf$sigmoid(x_rep * w_12_rep + tf$reshape(b_12, shape=c(1L,T)))
  h_3 = tf$sigmoid(x_rep * w_13_rep + tf$reshape(b_13, shape=c(1L,T)))
  h_4 = tf$sigmoid(x_rep * w_14_rep + tf$reshape(b_14, shape=c(1L,T)))
  h_5 = tf$sigmoid(x_rep * w_15_rep + tf$reshape(b_15, shape=c(1L,T)))
  h_6 = tf$sigmoid(x_rep * w_16_rep + tf$reshape(b_16, shape=c(1L,T)))
  
  
  w_21 =  tf$slice(w,c(0L,12L),c(T,1L))  
  w_22 =  tf$slice(w,c(0L,13L),c(T,1L))  
  w_23 =  tf$slice(w,c(0L,14L),c(T,1L))  
  w_24 =  tf$slice(w,c(0L,15L),c(T,1L))  
  w_25 =  tf$slice(w,c(0L,16L),c(T,1L))  
  w_26 =  tf$slice(w,c(0L,17L),c(T,1L))  
  b_2 =  tf$slice(w,c(0L,18L),c(T,1L))
  
  mu = h_1 * tf$squeeze(w_21)  + h_2 * tf$squeeze(w_22) + h_3 * tf$squeeze(w_23) +
    h_4 * tf$squeeze(w_24)  + h_5 * tf$squeeze(w_25) + h_6 * tf$squeeze(w_26) +
    tf$squeeze(b_2)
  
  mu = tf$transpose(mu) #oups did it wrong way
  return (tfp$distributions$Normal(loc=mu, scale=sigma)$sample())
}




### Intercept
N = 100L
T = 50000L
x = seq(-10,10, length.out=N)
y = get_post_predictive(w, x)
#y = get_post_predictive_6(w, x)
y_med = apply(y, 2, quantile, probs=0.5)
y_l = apply(y, 2, quantile, probs=0.05)
y_u = apply(y, 2, quantile, probs=1-0.05)

y_med_g = apply(y_preds_MFGauss, 1, quantile, probs=0.5)
y_l_g = apply(y_preds_MFGauss, 1, quantile, probs=0.05)
y_u_g = apply(y_preds_MFGauss, 1, quantile, probs=1-0.05)

y_med_tmmf = apply(y_preds_MFTM, 1, quantile, probs=0.5)
y_l_tmmf = apply(y_preds_MFTM, 1, quantile, probs=0.05)
y_u_tmmf = apply(y_preds_MFTM, 1, quantile, probs=1-0.05)

#VI
pdf(file.path(dir_name, 'network_ppd.pdf'), width = 7, height = 5)
plot(x, y_med, ylim=c(-2.5,3), ylab='y', type='l', col='skyblue', lwd=2, las=1)
for (t in 1:20) {
  lines(x, y[t,], col='lightgray', lwd=0.5)
}
#MCMC
lines(post_mcmc$X, post_mcmc$qlow, col='darkolivegreen4', lty=3, lwd=2)
lines(post_mcmc$X, post_mcmc$qhigh, col='darkolivegreen4', lty=3, lwd=2)
lines(post_mcmc$X, post_mcmc$med, col='darkolivegreen4', lty=3, lwd=2)

lines(x, y_med, ylim=c(-2.5,3), type='l', col='deepskyblue3', lwd=2)
points(x_r,y_r, col='black', cex=0.75)
lines(x, y_l, col='deepskyblue3', lwd=1)
lines(x, y_u, col='deepskyblue3', lwd=1)

lines(xx, y_u_g, col='orange', lty=2)
lines(xx, y_l_g, col='orange', lty=2)
lines(xx, y_med_g, col='orange', lty=2)
dev.off()

df_samples = data.frame(x = x, y = y[1,]$numpy(), t = 1L)
for (t in 2:20) {
  df_samples = rbind(df_samples, data.frame(x = x, y = y[t,]$numpy(), t = t))
}
df_samples$t = as.factor(df_samples$t)
ggplot(df_samples) + geom_line(aes(x=x,y=y, group=t))

apatheme=theme_bw(base_size = 22)+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        legend.title=element_blank(),
        legend.position = c(0.75,0.85),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))

df_quantiles = data.frame(x = post_mcmc$X, y = post_mcmc$qlow, method = 'MCMC', d=1)
df_quantiles = rbind(df_quantiles, data.frame(x = post_mcmc$X, y = post_mcmc$qhigh, method = 'MCMC', d=2))
df_quantiles = rbind(df_quantiles, data.frame(x = post_mcmc$X, y = post_mcmc$med, method = 'MCMC', d=3))

df_quantiles = rbind(df_quantiles, data.frame(x = x, y = y_med, method = 'BF-VI',d=4))
df_quantiles = rbind(df_quantiles, data.frame(x = x, y = y_l, method = 'BF-VI',d=5))
df_quantiles = rbind(df_quantiles, data.frame(x = x, y = y_u, method = 'BF-VI',d=6))

df_quantiles = rbind(df_quantiles, data.frame(x = xx, y = y_u_g, method = 'Gauss-MF',d=7))
df_quantiles = rbind(df_quantiles, data.frame(x = xx, y = y_l_g, method = 'Gauss-MF',d=8))
df_quantiles = rbind(df_quantiles, data.frame(x = xx, y = y_med_g, method = 'Gauss-MF',d=9))

ggplot(df_quantiles) + 
  geom_line(data = df_samples, aes(x=x,y=y,group=t), col='lightgray', alpha=0.5)+
  geom_point(data = data.frame(x_r, y_r), aes(x=x_r, y=y_r), col='black', size=2) + 
  geom_line(data = df_quantiles, aes(x=x,y=y, col=method, group = d)) +
  scale_color_manual(values = c('blue', 'orange', 'red')) + 
  apatheme 

ggsave(file.path(dir_name,'network_ppd.pdf'), width = 7, height=3.8)
# ggsave('~/Dropbox/Apps/Overleaf/bernvi/images/network_ppd.pdf', width = 7, height=3.8)

### Weights
boxplot(w, ylim=c(-10,15), main='VI') 
for (i in 1:10) hist(w[,i],100, main=i)
cor(w)
TT = 3000
plot(w[1:TT,1],w[1:TT,7])

if (PDF) dev.off()

