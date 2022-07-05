### Calculation of KL(q || post) ###
eval_kl_cauchy = function(w_samples, log_qw) {
  y = c(1.2083935, -2.7329216, 4.1769943, 1.9710574, -4.2004027, -2.384988)
  sigma = 0.5
  w1_m = as.matrix(w_samples)
  y_m = t(as.matrix(y))
  f = function(y,m,s){
    return (dcauchy(y, location = w1_m, scale = sigma))
  }
  log_liks = rowSums(apply(y_m, 2, FUN=dcauchy, location = w1_m, scale = sigma, log=TRUE))
  #log_liks = np.sum(cauchy.logpdf(y, loc=w1), scale=sigma), axis=1)
  priors = dnorm(w1_m, mean = 0, sd = 1, log = TRUE)
  #priors = prior_dist.log_prob(samples).numpy()
  post_un = log_liks + priors
  # sort = np.argsort(samples)
  # samples_s = samples[sort]
  # post_un_s = post_un[sort]
  return(mean(log_qw - post_un) + (-21.43069))
}  

eval_h1 = function(theta_1m, z_t){
  z_t = tf$clip_by_value(z_t,1E-5, 1.0-1E-5) #Dim (T,1)
  BE_tm = beta_dists_for_h$prob(z_t) #dim (T,M)
  #BE_tm * theta_1m is Dim(T,M) 
  return (tf$reduce_mean(BE_tm * theta_1m, axis=1L)) #Dim (T,)
}


get_z2_samples = function(params, line = 2, num_samples = 1e4L, trafo = 'F1F2'){
  if (trafo == 'F1F2') {
    z = rnorm(num_samples)
    a = tf$math$softplus(params$az[line])$numpy()
    z2 = a * z - params$bz[line]
    return (1/(1+exp(-z2)))
  } else if (trafo == 'TruncF2') {
    shape = c(as.integer(num_samples), 1L)
    z_ = tfp$distributions$TruncatedNormal(loc=0.5*tf$ones(shape), scale=0.15*tf$ones(shape), low=0.0001+tf$zeros(shape), high=tf$ones(shape)-0.0001)
    z = z_$sample()$numpy()
    return (z)
  } else if (trafo == 'SigmoidF2') {
    z = rnorm(num_samples)
    return (1/(1+exp(-z)))
  }
}

get_w_logqw = function(a_np, b_np, theta, num_samples = 1e6L, trafo = 'F1F2'){
  shape = c(as.integer(num_samples), 1L)
  if (trafo == 'F1F2') {
    with(tf$GradientTape() %as% tape, {
      dist = tfp$distributions$Normal(0,1)
      zz = dist$sample(num_samples)
      tape$watch(zz)
      a = tf$math$softplus(tf$Variable(a_np))
      z2 =  a * zz - b_np
      z2s = tf$reshape(tf$math$sigmoid(z2), c(-1L,1L))
      w = eval_h1(theta, z2s)
      dw_dz = tape$gradient(w, zz)
    })
    log_p_z = dist$log_prob(zz)
    log_q_w = log_p_z - tf$math$log(tf$math$abs(dw_dz))
  } else if (trafo == 'SigmoidF2') {
    with(tf$GradientTape() %as% tape, {
      dist = tfp$distributions$Normal(0,1)
      zz = dist$sample(num_samples)
      tape$watch(zz)
      z2s = tf$reshape(tf$math$sigmoid(zz), c(-1L,1L))
      w = eval_h1(theta, z2s)
      dw_dz = tape$gradient(w, zz)
    }) 
    log_p_z = dist$log_prob(zz)
    log_q_w = log_p_z - tf$math$log(tf$math$abs(dw_dz))
  } else if (trafo == 'TruncF2') {
    with(tf$GradientTape() %as% tape, {
      dist = tfp$distributions$TruncatedNormal(loc=0.5, scale=0.15, low=0.0001, high=1.-0.0001)
      zz = dist$sample(num_samples)
      tape$watch(zz)
      zz = tf$reshape(zz, c(-1L,1L))
      w = eval_h1(theta, zz)
      dw_dz = tape$gradient(w, zz)
    }) 
    log_p_z = dist$log_prob(zz)
    log_q_w = log_p_z - tf$math$log(tf$math$abs(dw_dz))
  }
  return(data.frame(
    w = w$numpy(),
    log_q_w = log_q_w$numpy()
  ))
}
