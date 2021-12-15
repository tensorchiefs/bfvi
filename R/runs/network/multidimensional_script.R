###### Notes #####
# Fitting a lr-model with D=3 a,b, and sigma Parameters using Berstein Flows and MAF
# The weights are determined by transforming z from the base distribution via a Bernstein polynomial of degree M
# For the first M weights theta_1 the M coefficients of the Bernstein are independed of z. 
# To guaranty the monotonic increase the are deterimend from the unresticted trainable variable theta_1_prime   
# The others 2-D unrestricted parameters theta_2_D_prime are determined by a masked autoregresive network (maf) 
# For estimating the ELBO, we draw T samples zfrom the base distribution and transform it to w 
# w and z have the dimensions (T,D,M)
rm(T)
library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
d = tf$version
d$VERSION #2.1.0
source("bern_utils.R")

### User Specific Settings 
RUN_DIR = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs'
SCRIPT_NAME = '~/Documents/workspace/tmvi/R/multidimensional_script.R'
SAVE = TRUE #Save the experiment
setwd("~/Documents/workspace/tmvi/R")

# Choose experiment ---------------
EXP_NAME = 'NETWORK_only9_Data'
#DATA = '2D'   #2-D regression on simulated P=2 predictors
#DATA = 'BOOK' #1-D regression on N=4 data points 
#DATA = '8SCHOOLS'    # Bessere Parametrisierung NCP
#DATA = '8SCHOOLS_CP'  # So wie man denkt, hierarchisch etwas schwerer
#DATA = '1SCHOOLS_CP' # So wie man denkt, hierarchisch
#DATA = 'DIAMONDS'
DATA = 'NETWORK'
#DATA = 'NETWORK6'
RELU = FALSE #Relu in the hidden layer

# Parameters
#optimizer = tf$keras$optimizers$Adam() #default lr=1E-3
#optimizer = tf$keras$optimizers$Adam(lr=1e-3, clipvalue=0.2, clipnorm=0.1) #8 Schools better than SGD
#optimizer = tf$keras$optimizers$SGD(lr=1e-3, clipvalue=0.2)
optimizer = tf$keras$optimizers$RMSprop()
number_epochs = 20000

#### General Parameters for TMVI ####
ML_INIT = FALSE #Initialise the paramters with ML Solution
T_INIT = 5000

M = 50L #Degree of the Bernstein Polynomials (due to technical reason minimum is 2)
T = 600L # number samples of w from q for ELBO-estimation during optimization 
T_SAMPLES = 50000L #sampels after training
w_min =  -10 #Minima valuel of the start variational distibution
w_max =  10

# Creating the run directories
if (SAVE){
  EXP_DIR = file.path(RUN_DIR, EXP_NAME)
  if (file.exists(EXP_DIR)){
    quit(save="ask")
  } else {
    dir.create(file.path(EXP_DIR))
  }
  #setwd(EXP_DIR)
  file.copy(SCRIPT_NAME, EXP_DIR, overwrite = TRUE)
}


# Experiment-specific code -----------------
# Network ======
if(DATA == 'NETWORK') {
  x_r = c(-5.41239, -4.142973,-5.100401,-4.588446,-2.0570369,-2.0035906,-3.7404475,
    -5.344997,4.506781,5.9761415,4.073539,5.168227,4.1196156,2.4791312,2.0348845, 2.7495284)
  y_r = c(0.973122  ,  0.96644104,  1.2311585 ,  0.5193988 , -1.2059958 ,
          -0.9434611 ,  0.8041748 ,  0.82996416, -1.3704962 , -0.3733918 ,
          -0.98566836, -1.1550032 , -1.0276004 ,  0.539029  ,  1.5336514 ,
          0.34641847)
  
  #Hack 
  x_r = x_r[1:9]
  y_r = y_r[1:9]
  
  plot(x_r, y_r)
  N = length(x_r)
  P = 1L
  D = 10L
  y = tf$reshape(y_r, shape=c(N, 1L))
  x = tf$reshape(tf$Variable(x_r, dtype='float32'), shape=c(N,P))
  sigma = 0.2
  
  # get the log-likelihood and the prior
  get_LL <- function(w, x, y, T) {
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
    w_12_rep = tf$reshape(w_12, shape=c(1L, T))
    w_13_rep = tf$reshape(w_13, shape=c(1L, T))
    if (RELU){
      h_1 = tf$nn$relu(x_rep * w_11_rep + tf$reshape(b_11, shape=c(1L,T)))
      h_2 = tf$nn$relu(x_rep * w_12_rep + tf$reshape(b_12, shape=c(1L,T)))
      h_3 = tf$nn$relu(x_rep * w_13_rep + tf$reshape(b_13, shape=c(1L,T)))
    } else{
      h_1 = tf$sigmoid(x_rep * w_11_rep + tf$reshape(b_11, shape=c(1L,T)))
      h_2 = tf$sigmoid(x_rep * w_12_rep + tf$reshape(b_12, shape=c(1L,T)))
      h_3 = tf$sigmoid(x_rep * w_13_rep + tf$reshape(b_13, shape=c(1L,T)))
    }
    w_21 =  tf$slice(w,c(0L,6L),c(T,1L))  
    w_22 =  tf$slice(w,c(0L,7L),c(T,1L))  
    w_23 =  tf$slice(w,c(0L,8L),c(T,1L))  
    b_2 =  tf$slice(w,c(0L,9L),c(T,1L))
    
    mu = h_1 * tf$squeeze(w_21)  + h_2 * tf$squeeze(w_22) + h_3 * tf$squeeze(w_23) + tf$squeeze(b_2)
    mu = tf$transpose(mu) #oups did it wrong way
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    return (tfp$distributions$Normal(loc=mu, scale=sigma)$log_prob(y_rep))
  }
  
  get_log_prior = function(w){
    tf$reshape(tf$reduce_sum(tfd_normal(0.,1.)$log_prob(w), axis=1L), shape=c(T,1L))
  }
  
}

# Experiment-specific code -----------------
# Network ======
if(DATA == 'NETWORK6') {
  x_r = c(-5.41239, -4.142973,-5.100401,-4.588446,-2.0570369,-2.0035906,-3.7404475,
          -5.344997,4.506781,5.9761415,4.073539,5.168227,4.1196156,2.4791312,2.0348845, 2.7495284)
  y_r = c(0.973122  ,  0.96644104,  1.2311585 ,  0.5193988 , -1.2059958 ,
          -0.9434611 ,  0.8041748 ,  0.82996416, -1.3704962 , -0.3733918 ,
          -0.98566836, -1.1550032 , -1.0276004 ,  0.539029  ,  1.5336514 ,
          0.34641847)
  plot(x_r, y_r)
  N = length(x_r)
  P = 1L
  D = 19L 
  y = tf$reshape(y_r, shape=c(N, 1L))
  x = tf$reshape(tf$Variable(x_r, dtype='float32'), shape=c(N,P))
  sigma = 0.2
  
  # get the log-likelihood and the prior
  get_LL <- function(w, x, y, T) {
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
    if (RELU){
      h_1 = tf$nn$relu(x_rep * w_11_rep + tf$reshape(b_11, shape=c(1L,T)))
      h_2 = tf$nn$relu(x_rep * w_12_rep + tf$reshape(b_12, shape=c(1L,T)))
      h_3 = tf$nn$relu(x_rep * w_13_rep + tf$reshape(b_13, shape=c(1L,T)))
      h_4 = tf$nn$relu(x_rep * w_14_rep + tf$reshape(b_14, shape=c(1L,T)))
      h_5 = tf$nn$relu(x_rep * w_15_rep + tf$reshape(b_15, shape=c(1L,T)))
      h_6 = tf$nn$relu(x_rep * w_16_rep + tf$reshape(b_16, shape=c(1L,T)))
    } else{
      h_1 = tf$sigmoid(x_rep * w_11_rep + tf$reshape(b_11, shape=c(1L,T)))
      h_2 = tf$sigmoid(x_rep * w_12_rep + tf$reshape(b_12, shape=c(1L,T)))
      h_3 = tf$sigmoid(x_rep * w_13_rep + tf$reshape(b_13, shape=c(1L,T)))
      h_4 = tf$sigmoid(x_rep * w_14_rep + tf$reshape(b_14, shape=c(1L,T)))
      h_5 = tf$sigmoid(x_rep * w_15_rep + tf$reshape(b_15, shape=c(1L,T)))
      h_6 = tf$sigmoid(x_rep * w_16_rep + tf$reshape(b_16, shape=c(1L,T)))
    }
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
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    return (tfp$distributions$Normal(loc=mu, scale=sigma)$log_prob(y_rep))
  }
  
  get_log_prior = function(w){
    tf$reshape(tf$reduce_sum(tfd_normal(0.,1.)$log_prob(w), axis=1L), shape=c(T,1L))
  }
  
}


# Diamonds ======
if(DATA == 'DIAMONDS') {
  if (FALSE){
    library(posteriordb)
    library(posterior)
    my_pdb <- pdb_local(path='~/Documents/workspace/posteriordb')
    po <- posterior("diamonds-diamonds", my_pdb)
    dat =  pdb_data(po)
    data = list(N = dat$N, Y = dat$Y, X = dat$X, K=dat$K, prior_only = dat$prior_only)
    save(data, file='diamonds.rda')
  }
  load('diamonds.rda')
  #Centering (we also have an itercept)
  Xc = scale(data$X[,-1], center = TRUE, scale = FALSE)
  N = data$N
  P = as.integer(data$K - 1L) #K contains the intercept
  y = tf$reshape(data$Y, shape=c(N, 1L))
  x = tf$reshape(tf$Variable(Xc, dtype='float32'), shape=c(N,P))
  D = P + 2L
  
  
  ml = lm(Y ~ Xc, data.frame(Y = data$Y, Xc = Xc))
  summary(ml)
  w_mu_init_r = coef(ml)
  sigma = sqrt(sum(ml$res^2)/(N-1-P))
  w_sigma = tfp$math$softplus_inverse(sigma)$numpy()
  #tf$math$softplus(w_sigma)
  w_mu_init_r = append(w_mu_init_r, w_sigma)
  w_mu_init = tf$reshape(tf$Variable(w_mu_init_r, dtype='float32'), shape=c(26L,1L))
  
  w_sd_init_r = 5*summary(ml)[["coefficients"]][,2]
  w_sd_init_r = append(w_sd_init_r, w_sigma*0.2)
  w_sd_init = tf$reshape(tf$Variable(w_sd_init_r, dtype='float32'), shape=c(26L,1L))
  
  
  
  get_LL <- function(w, x, y, T) {
    w0 = tf$slice(w,c(0L,0L),c(T,1L))  #First is intercept 
    wx = tf$slice(w,c(0L,1L),c(T,P))   #Slopes
    ws = tf$slice(w,c(0L,P+1L),c(T,1L))  #for SD
    mu = tf$matmul(wx, tf$transpose(x)) + w0 #T,X
    sigma = tf$math$softplus(ws)
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    return (tfp$distributions$Normal(loc=mu, scale=sigma)$log_prob(y_rep)) 
  }
  
  get_log_prior = function(w){
    w0 = tf$slice(w,c(0L,0L),c(T,1L))    #First is intercept 
    wx = tf$slice(w,c(0L,1L),c(T,P))     #Slopes
    ws = tf$slice(w,c(0L,P+1L),c(T,1L))  #for SD
    sigma = tf$math$softplus(ws)
    return(
        tfd_student_t(3L,8, 10)$log_prob(w0) + #Stan student_t_lpdf(Intercept | 3, 8, 10);
        tf$reshape(tf$reduce_sum(tfd_normal(0.,1.)$log_prob(wx), axis=1L), shape=c(T,1L)) + # normal_lpdf(b | 0, 1);
        tfd_student_t(3L, 0, 10)$log_prob(sigma) + 0.6931472  #student_t_lpdf(sigma | 3, 0, 10) + 0.6931472
    )
  }
}
# 8SCHOOLS CP ============
if(DATA == '8SCHOOLS_CP' || DATA == '1SCHOOLS_CP') {
  library(rstan)
  data <- read_rdump("eight_schools.data.R")
  
  x_r = data$sigma
  y_r = data$y
  N = 8L
  D = 10L #mu, tau, and 8 Theta_Tilde
  P = 1L  #Input dimension of "x" here sigma
  
  if (DATA == '1SCHOOLS_CP') {
    x_r = data$sigma[1]
    y_r = data$y[1]
    N = 1L
    D = 3L #mu, tau, and 8 
    data$y = data$y[1,drop=FALSE]
    data$sigma = data$sigma[1,drop=FALSE]
    data$J = 1L
  }
  
  y = tf$reshape(y_r, shape=c(N, 1L))
  x = tf$reshape(tf$Variable(x_r), shape=c(N,P))
  
  
  get_LL <- function(w, x, y, T) {
    #The w are (up to transformations) the mu, tau, and theta_tilde[1-8]  
    sigma = x
    #Determination of the likelihood (see stan file)
    # mu ~ normal(0, 5);
    # tau ~ cauchy(0, 5);
    # theta ~ normal(mu, tau);
    # y ~ normal(theta, sigma);
    
    # Not needed in this parametrization
    # mu = tf$slice(w,c(0L,0L),c(T,1L)) 
    # tau_prime = tf$slice(w,c(0L,1L),c(T,1L)) 
    # tau = tf$math$softplus(tau_prime)   #Needs to be positive
    theta_j = tf$slice(w,c(0L,2L),c(T,N))
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    sigma_rep = tf$tile(tf$transpose(sigma), c(T,1L))
    return (tfp$distributions$Normal(loc=theta_j, scale=sigma_rep)$log_prob(y_rep))
  }
  
  get_log_prior = function(w){
    #mu ~ normal(0, 5);
    #tau ~ cauchy(0, 5);
    #theta_j|mu,tau ~ normal(mu, tau);
    #p(mu,tau, theta_j) = p(theta_j|mu, tau) p(mu) p(tau)
    
    #mu = tf$slice(w,c(0L,0L),c(T,1L)) 
    #tau_prime = tf$slice(w,c(0L,1L),c(T,1L)) 
    mu = tf$slice(w,c(0L,1L),c(T,1L)) 
    tau_prime = tf$slice(w,c(0L,0L),c(T,1L)) 
    
    tau = tf$math$exp(0.1*tau_prime)   #Needs to be positive
    theta_j = tf$slice(w,c(0L,2L),c(T,N))
    return(
      tfd_normal(0,5.)$log_prob(mu) + #mu 
        tfd_cauchy(0.,5.)$log_prob(tau) + #tau
        tf$reshape(tf$reduce_sum(tfd_normal(mu, tau)$log_prob(theta_j), axis=1L), shape=c(T,1L)) #theta_tilde
    )
  }  
  
}
# 8SCHOOLS Non Centered ============
if(DATA == '8SCHOOLS') {
  # Data of the Eight Schools Model
  #See https://github.com/betanalpha/knitr_case_studies/blob/master/rstan_workflow/eight_schools.data.R
  library(rstan)
  data <- read_rdump("eight_schools.data.R")
  
  x_r = data$sigma
  y_r = data$y
  N = 8L
  D = 10L #mu, tau, and 8 Theta_Tilde
  P = 1L  #Input dimension of "x" here sigma
  y = tf$reshape(y_r, shape=c(N, 1L))
  x = tf$reshape(tf$Variable(x_r), shape=c(N,P))
  
  
  get_LL <- function(w, x, y, T) {
    #The w are (up to transformations) the mu, tau, and theta_tilde[1-8]  
    sigma = x
    #Determination of the likelihood (see stan file)
    # real mu;
    # real<lower=0> tau;
    # real theta_tilde[J];
    
    #mu = tf$slice(w,c(0L,0L),c(T,1L)) 
    #tau_prime = tf$slice(w,c(0L,1L),c(T,1L)) 
    mu = tf$slice(w,c(0L,1L),c(T,1L)) 
    tau_prime = tf$slice(w,c(0L,0L),c(T,1L)) 
    tau = tf$math$exp(0.1*tau_prime)
    theta_tilde = tf$slice(w,c(0L,2L),c(T,8L))
    
    #real theta[J];
    theta_j = mu + tau * theta_tilde
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    sigma_rep = tf$tile(tf$transpose(sigma), c(T,1L))
    return (tfp$distributions$Normal(loc=theta_j, scale=sigma_rep)$log_prob(y_rep))
  }
  
  get_log_prior = function(w){
    #mu ~ normal(0, 5);
    #tau ~ cauchy(0, 5);
    #theta_tilde ~ normal(0, 1);
    
    #mu = tf$slice(w,c(0L,0L),c(T,1L)) 
    #tau_prime = tf$slice(w,c(0L,1L),c(T,1L)) 
    mu = tf$slice(w,c(0L,1L),c(T,1L)) 
    tau_prime = tf$slice(w,c(0L,0L),c(T,1L))
    tau = tf$math$exp(0.1*tau_prime)
    theta_tilde = tf$slice(w,c(0L,2L),c(T,8L))
    return(
        tfd_normal(0,5.)$log_prob(mu) + #mu 
        tfd_cauchy(0.,5.)$log_prob(tau) + log(2.0) + #tau
        tf$reshape(tf$reduce_sum(tfd_normal(0., 1.)$log_prob(theta_tilde), axis=1L), shape=c(T,1L)) #theta_tilde
    )
  }  
 

}
# BOOK --------
if(DATA == 'BOOK') {  
  
  # lineare regression with 1 predictor and 4 data points
  N = 4L
  P = 1L #number of predictors
  D = P + 2L  #Dimension (number of weights) predictors + intercept + sigma
  
  evidence = -16.696 #Numerical Integration with Mathematica NIntegrate
  
  # data, Example from the "book" chapter 7
  x_r = c(-2.0000e+00, -6.6667e-01, 6.6667e-01, 2.0000e+00)
  y_r = c(-6.2503e+00, -2.5021e+00, -6.0753e+00, 7.9208e+00)
  
  y = tf$reshape(y_r, shape=c(N, 1L))
  x = tf$reshape(tf$Variable(x_r), shape=c(N,P))
  plot(x_r, y_r)
  
  #The prior and likelihood for the specific problem
  get_LL <- function(w, x, y, T) {
    #Determination of the likelihood
    w1 = tf$slice(w,c(0L,0L),c(T,1L)) 
    w2 = tf$slice(w,c(0L,1L),c(T,1L)) 
    w3 = tf$slice(w,c(0L,2L),c(T,1L))
    mu = w1*tf$transpose(x) + w2 #T,X
    sigma = tf$math$softplus(w3)
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    return (tfp$distributions$Normal(loc=mu, scale=sigma)$log_prob(y_rep))
  }
  
  get_log_prior = function(w){
    w1 = tf$slice(w,c(0L,0L),c(T,1L)) 
    w2 = tf$slice(w,c(0L,1L),c(T,1L)) 
    w3 = tf$slice(w,c(0L,2L),c(T,1L))
    sigma = tf$math$softplus(w3)
    return(
      tfd_normal(0,10.)$log_prob(w1) + 
        tfd_normal(0,10.)$log_prob(w2) +
        tfd_log_normal(0., 1.)$log_prob(sigma)
    )
  } #'Book' 
}
## End Book, 1-D (simple) linear regression 


####### 2-D experiment ############
# (multiple) linear regression with 2 predictors 
if (DATA == '2D') {
  
  N = 6L # number data points
  P = 2L #number of predictors
  D = P + 2L  #Dimension (number of weights) predictors + intercept + sigma
  
  evidence = -12.62 # from mathematica (numerical integration)
  
  if (FALSE){
    set.seed(42)
    z_dg = rnorm(N, mean=0, sd=1)
    x_dg = scale(rnorm(N, mean = 1.42*z_dg, sd = 0.1), scale = FALSE)
    y_r = rnorm(N, mean = 0.42*x_dg - 1.1 * z_dg, sd=0.42)
    #loss (average over 30k-50k iterations) 11.80437
    x_r = matrix(c(z_dg, x_dg), ncol=2,byrow = FALSE)
    save(x_r,y_r, file='2D.rdata')
    evidence = -12.62
  }
  load('2D.rdata')  
  x_r
  y_r
  #### ML estimations 
  l = lm(y_r ~ x_r)
  summary(l)
  w_ml = coef(l)  #Intercept
  sd_ml = sqrt(sum(l$residuals^2)/(length(y_r)-2))
  logLik(l)
  
  #going to tf
  y = tf$reshape(tf$Variable(y_r, dtype='float32'), shape=c(N, 1L))
  x = tf$reshape(tf$Variable(x_r, dtype='float32'), shape=c(N,P))
  
  # get the log-likelihood and the prior
  get_LL <- function(w, x, y, T) {
    #Determination of the likelihood
    w0 = tf$slice(w,c(0L,0L),c(T,1L))  #First is intercept 
    wx = tf$slice(w,c(0L,1L),c(T,P))   #P Slopes
    ws = tf$slice(w,c(0L,P+1L),c(T,1L))  #for SD
    mu = tf$matmul(wx, tf$transpose(x)) + w0 #T,X
    sigma = tf$math$softplus(ws)
    y_rep = tf$transpose(tf$tile(y, c(1L,T)))
    return (tfp$distributions$Normal(loc=mu, scale=sigma)$log_prob(y_rep))
  }
  
  get_log_prior = function(w){
    w0 = tf$slice(w,c(0L,0L),c(T,1L))    #First is intercept 
    #TODO make w1 work for general dimension e.g. with unrolling rolling trick
    w1 = tf$slice(w,c(0L,1L),c(T,1L))      
    w2 = tf$slice(w,c(0L,2L),c(T,1L))     
    ws = tf$slice(w,c(0L,P+1L),c(T,1L))  #for SD
    sigma = tf$math$softplus(ws)
    return(
      tfd_normal(0,10.)$log_prob(w0) + 
        tfd_normal(0,10.)$log_prob(w1) + 
        tfd_normal(0,10.)$log_prob(w2) +
        tfd_log_normal(0.5, 1.)$log_prob(sigma)
      #tfd_truncated_normal(loc=0.7,scale=5., low = 0.1, high = 50)$log_prob(sigma)
    )
    # plot priors 
    sigs = seq(0,3,length.out=1000)
    plot(sigs, dlnorm(sigs, meanlog = 0.5, sdlog = 1), type='l', 
         col='red', main="prior for sigma")
  }
}
##### End of 2-D  
#### Definition of MAF ####
#z = tf$reshape(base_dist$sample(T*D), c(T, D)) #Dim (T,D)
maf = tfp$bijectors$AutoregressiveNetwork(
  params=M, 
  event_shape=c(NULL, D),
  hidden_units = c(10L,10L)
  #hidden_units = c(50L,20L)
)
d = maf$weights


###### Base Distribution ######
# The base distribution (with support [0,1] as needed for the Berstein-Polynomials)
base_dist=tfp$distributions$TruncatedNormal(loc=0.5,scale=0.15,low=0.,high=1.)
beta_dists_for_h = init_beta_dist_for_h(M)
beta_dists_for_h_dash = init_beta_dist_for_h_dash(M)
zs = seq(0,1, length.out=500)
plot(zs, as.numeric(base_dist$prob(zs)), type='l', ylab='density', main='Base distribution')

# The folling is for getting theta_1, needed to transform 1. dim z to w

# Determines  w-samples for first dimension form corresponding z-samples
#Returns w = h(z) each t z-samples
# see formula page 12 mlt-paper https://arxiv.org/abs/1508.06749
#h(z) = 1/M \sum_m theta_1m * BE_tm
#NOTE: theta_1m is restricted 
eval_h1 = function(theta_1m, z_t){
  z_t = tf$clip_by_value(z_t,1E-5, 1.0-1E-5) #Dim (T,1)
  BE_tm = beta_dists_for_h$prob(z_t) #dim (T,M)
  #BE_tm * theta_1m is Dim(T,M) 
  return (tf$reduce_mean(BE_tm * theta_1m, axis=1L)) #Dim (T,)
}

# Initialising theta_1_prime 
#This trivaly ensures a roughly uniform variational distribution of w before training
d = base_dist$cdf(seq(0,1,length.out=M))*(w_max - w_min)+w_min
delta = diff(d$numpy())
t=c(d$numpy()[1], tfp$math$softplus_inverse(delta)$numpy())
theta_1_prime = tf$Variable(matrix(t,nrow=1), dtype='float32')
theta_1 = to_theta(theta_1_prime)
# for plotting the start w-dist from q before training
z1 = tf$reshape(base_dist$sample(10000L), c(10000L,1L))
w1 = as.numeric(eval_h1(theta_1, z1))
hist(w1, freq = FALSE)

# The folling is for getting theta_2_D, needed to transform 2.-p. dim z to w
#theta_prime = maf(z) #Yields a T,D,M Tensor

# Determines  w-samples for dimensions 2 to D form corresponding z-samples 
# depending on variational parameters theta 
eval_hD = function(theta_1, z1, theta_2_D, z_2_D) {
  w1 = eval_h1(theta_1, z1)
  z_2_D = tf$clip_by_value(z_2_D,1E-5, 1.0-1E-5) #Avoid exact zeros and ones
  beta_dist = tfd_beta(1:M, M:1)
  #unrolling and rerolling 
  Bes_2_D = tf$reshape(
    beta_dist$prob(tf$reshape(z_2_D, c(T*(D-1L),1L))), 
    shape = c(T, (D-1L), M))
  
  w_2_D = tf$reduce_mean(theta_2_D * Bes_2_D, axis=2L)
  w = tf$concat(c(tf$reshape(w1,c(T,1L)), w_2_D), axis=1L)
  return(w)
}

# for change-of-variable formula we need h'(z)

# see formula page 12 mlt-paper https://arxiv.org/abs/1508.06749, but drop M/(M+1)
#Returns h'(z) each t z-samples
#NOTE: input theta_1m is restricted 
eval_h1_dash = function(theta_1m, z_t){
  z_t = tf$clip_by_value(z_t,1E-5, 1.0-1E-5)
  BE = beta_dists_for_h_dash$prob(z_t) 
  dtheta = (theta_1m[,2:(ncol(theta_1m))]-theta_1m[,1:(ncol(theta_1m)-1)])
  return (tf$reduce_sum(BE * dtheta, axis=1L))
}

# the following is needed for stabalize cudo computiation
#For numerical stability Stolen from https://github.com/jhuggins/viabel/blob/master/viabel/diagnostics.py 
mean_and_check_mc_error = function(a, atol=0.01, rtol=0.0){
  m = mean(a)
  s = sd(a)/sqrt(length(a))
  if (s > rtol*abs(m) + atol){
    print('There is something foule in the state of denmark')
  }
  return (m)
}

#### Initialization of the Parameters #####
if (ML_INIT){
  optimizer_maf = tf$keras$optimizers$Adam(lr=1e-2, clipvalue=0.5, clipnorm=0.1) #8 Schools better than SGD
  
  loss_maf = function(theta_1_prime, maf, return_w = FALSE){
    theta_1 = to_theta(theta_1_prime)
    z1 = tf$reshape(base_dist$sample(T), c(T,1L)) #The first dimension
    z_2_D = tf$reshape(base_dist$sample(T*(D-1L)), c(T, D-1L))#Second until last dimension 
    z = tf$concat(c(z1, z_2_D), axis=1L) 
    
    theta_2_D_prime_with_zero = maf(z) 
    #Removing the zeros in the first entry in D of T,D,M
    theta_2_D_prime = tf$slice(theta_2_D_prime_with_zero, c(0L,1L,0L), c(T, D-1L, M)) 
    #To get to_theta working we unroll T,D-1,M to T*D-1,M and re-roll back
    theta_2_D = tf$reshape(
      to_theta(tf$reshape(theta_2_D_prime, shape = c(T*(D-1L),M))) 
      , shape=c(T,D-1L,M))
    w = eval_hD(theta_1, z1 = z1, theta_2_D = theta_2_D, z_2_D = z_2_D)
    w_mu_init_rep = tf$transpose(tf$tile(w_mu_init, c(1L,T)))
    loss_mu = tf$reduce_mean((w - w_mu_init_rep)^2)
    
    w_mean = tf$reduce_mean(w, axis=0L)
    w_sd = tf$reshape(tf$reduce_mean(tf$math$abs(w - w_mean), axis=0L), shape=c(D,1L))
    loss_sd = tf$reduce_mean((w_sd - w_sd_init)^2)
    loss_maf = loss_mu + loss_sd
    if (return_w){
      return (list(w=w, loss_maf=loss_maf))
    } else{
      return (loss_maf)
    }
  }
  loss_maf(theta_1_prime, maf)
  
  LF_maf = tf_function(loss_maf)
  loss_hist = NULL
  for (i in 1:T_INIT){
    if(i %% 2 == 0) {
      # d = loss(theta_1_prime, maf, return_w = TRUE) #slow!
      # cubo_1_hist[i] = d$cubo_1
      print(paste0(i, " ", loss_$numpy()))#, " cubo_1 ", d$cubo_1, " cubo num stable", d$cubo_stable))
    }
    with(tf$GradientTape() %as% t, {
      loss_ = LF_maf(theta_1_prime, maf)
    })
    w_maf = maf$trainable_variables
    pars = c(theta_1_prime, w_maf)
    grads = t$gradient(loss_, pars)
    optimizer_maf$apply_gradients(purrr::transpose(list(grads, pars)))
    loss_hist[i] = loss_$numpy()
  }
  
  d = loss_maf(theta_1_prime, maf, return_w = TRUE)
  boxplot(d$w$numpy())
  points(ml$coefficients, col='red', size=2)
}

##  Loss ####
# Caluclation of the ELBO-Loss (return a value which needs to be minimized)
loss = function(theta_1_prime, maf,  return_w = FALSE){
  theta_1 = to_theta(theta_1_prime)
  
  z1 = tf$reshape(base_dist$sample(T), c(T,1L)) #The first dimension
  z_2_D = tf$reshape(base_dist$sample(T*(D-1L)), c(T, D-1L))#Second until last dimension 
  z = tf$concat(c(z1, z_2_D), axis=1L) 
  
  theta_2_D_prime_with_zero = maf(z) 
  #Removing the zeros in the first entry in D of T,D,M
  theta_2_D_prime = tf$slice(theta_2_D_prime_with_zero, c(0L,1L,0L), c(T, D-1L, M)) 
  #To get to_theta working we unroll T,D-1,M to T*D-1,M and re-roll back
  theta_2_D = tf$reshape(
    to_theta(tf$reshape(theta_2_D_prime, shape = c(T*(D-1L),M))) 
    , shape=c(T,D-1L,M))
  
  w = eval_hD(theta_1, z1 = z1, theta_2_D = theta_2_D, z_2_D = z_2_D)
  # Compute terms needed for n_ELBO = KL_q_prior - LL
  #log-likelihood
  LLt = tf$reshape(tf$reduce_sum(get_LL(w,x,y,T), axis=1L), shape=c(T,1L))
  
  # penalty KL(q(w) || prior) needs several steps
  log_pior = get_log_prior(w)
  
  #next determine q(w) = p(z) / prod_i(Jii)
  
  #log(p(z)) for the samples (z's are no correlations between dimensions)
  log_pz = tf$reduce_sum(
    tf$reshape(
      base_dist$log_prob(tf$reshape(z, shape = c(T*D,1L))),
      shape=c(T,D)
    ), axis=1L)
  
  # Jacobi diagonal elements
  J11 = eval_h1_dash(to_theta(theta_1_prime), z1)
  #Jii = d h_i /d z_i = \sum_m (??_(m+1) -  ??_(m)) * Be_
  BE = tf$reshape(beta_dists_for_h_dash$prob(
    tf$reshape(z_2_D, shape = c(T*(D-1L),1L))
  ), c(T,D-1L,M-1L))
  dtheta = tf$slice(theta_2_D,c(0L,0L,1L), c(T,D-1L,M-1L)) - tf$slice(theta_2_D,c(0L,0L,0L), c(T,D-1L,M-1L))
  J_ii = tf$reduce_sum(BE * dtheta, axis=2L) #Jaccobi Diagonals for i>1 a T, D-1 Tensor
  
  log_det = log(J11) + tf$reduce_sum(log(J_ii),axis=1L) 
  
  log_q = tf$reshape(log_pz - log_det,c(T, 1L)) #eq 5 (change of variable)
  
  KL_q_prior_t = -log_pior + log_q
  loss_t = (KL_q_prior_t - LLt)

  #Using the REINFORCE aka score gradient trick
  #loss_t = log_q * tf$stop_gradient(-LLt - log_pior + log_q)
  ##### Upperbound Approach (Dieng, Blei)
  #log_liks = get_LL(w,x,y,T)
  #log_prios = tf$exp(log_pior)
  #qs = tf$reshape(exp(log_q), shape=c(T,1L))
  #EXP_CUBO_2 = tf$reduce_mean((tf$multiply(liks,prios)/qs)**2L)
  if (return_w == FALSE){
    return(tf$reduce_mean(loss_t))
  } else {
    #Here we provide the output dianostics
    return(list(
      LLs = apply(get_LL(w,x,y,T)$numpy(),1,sum),   # The log-likelihoods for the samples
      L_priors = as.numeric(log_pior), # The log-prios for the samples
      log_qs = log_q$numpy(),
      w = w$numpy()                   # The samples drawn
    ))
  }
}

##### Gradient Descent ######
LF = tf_function(loss)
loss_hist = NULL
cubo_1_hist = NULL
for (i in 1:number_epochs){
  if(i %% 2 == 0) {
    # d = loss(theta_1_prime, maf, return_w = TRUE) #slow!
    # cubo_1_hist[i] = d$cubo_1
    print(paste0(i, " ", loss_$numpy()))#, " cubo_1 ", d$cubo_1, " cubo num stable", d$cubo_stable))
  }
  with(tf$GradientTape() %as% t, {
    loss_ = LF(theta_1_prime, maf)
  })
  w_maf = maf$trainable_variables
  pars = c(theta_1_prime, w_maf)
  grads = t$gradient(loss_, pars)
  optimizer$apply_gradients(purrr::transpose(list(grads, pars)))
  loss_hist[i] = loss_$numpy()
}

T_OLD = T
T = T_SAMPLES
samples = loss(theta_1_prime, maf, return_w = TRUE)

if(SAVE){
  save(loss_hist, file = file.path(EXP_DIR, 'loss_hist.rda'))
  save(samples, file = file.path(EXP_DIR, 'samples.rda'))
}

T = T_OLD
log_joint = samples$LLs + samples$L_priors
log_weights =  + as.numeric(samples$L_priors) - samples$log_qs
library(loo)
res_psis = psis(log_ratios = log_joint - samples$log_qs)
k_hat = res_psis$diagnostics$pareto_k 
plot(loss_hist, type='l', main='Loss function', sub = paste0('k_hat ', k_hat))



