# utils_scale = function (y){
#   min_y = min(y) 
#   max_y = max(y) 
#   return ( (y-min_y)/(max_y-min_y) )
# }
# 
# utils_back_scale = function(y_scale, y){
#   min_y = min(y)
#   max_y = max(y)
#   return (y_scale * (max_y - min_y) + min_y)
# }

############################################################
# utils for bernstein
############################################################

# Construct h according to MLT paper (Hothorn, Möst, Bühlmann, p12)
init_beta_dist_for_h = function(len_theta){
  beta_dist = tfd_beta(1:len_theta, len_theta:1) 
  return (beta_dist)
}

# Construct h_dash according to MLT (Hothorn, Möst, Bühlmann, p12) correcting the factor M/(M+1) with 1 
init_beta_dist_for_h_dash = function(len_theta){
  M = len_theta - 1
  beta_dist = tfd_beta(1:M, M:1)
  return (beta_dist)
}

.eval_h = function(theta_im, y_i, beta_dist_h){
  y_i = tf$clip_by_value(y_i,1E-5, 1.0-1E-5)
  f_im = beta_dist_h$prob(y_i) 
  return (tf$reduce_mean(f_im * theta_im, axis=1L))
}

.eval_h_dash = function(theta, y, beta_dist_h_dash){
  y = tf$clip_by_value(y,1E-5, 1.0-1E-5)
  by = beta_dist_h_dash$prob(y) 
  dtheta = (theta[,2:(ncol(theta))]-theta[,1:(ncol(theta)-1)])
  return (tf$reduce_sum(by * dtheta, axis=1L))
}

# ######## function that turns pre_theta from NN ouput to theta
to_theta = function(pre_theta){
  #Trivially calclutes theta_1 = h_1, theta_k = theta_k-1 + exp(h_k)
  d = tf$concat( c(tf$zeros(c(nrow(pre_theta),1L)),
                   tf$slice(pre_theta, begin = c(0L,0L), size = c(nrow(pre_theta),1L)),
                   #tf$math$exp(pre_theta[,2:ncol(pre_theta)])),axis=1L)
                   tf$math$softplus(pre_theta[,2:ncol(pre_theta)])),axis=1L)
  #python d = tf.concat( (tf.zeros((h.shape[0],1)), h[:,0:1], tf.math.softplus(h[:,1:h.shape[1]])),axis=1)
  return (tf$cumsum(d[,2L:ncol(d)], axis=1L))
}

########################
# Class bernp
# S3 Class build according to https://adv-r.hadley.nz/s3.html
# My first class in R, hence some comments

#"Private" and complete constructor. This constructor verifies the input.
new_bernp = function(len_theta = integer(), base_dist){
  stopifnot(is.integer(len_theta))
  stopifnot(len_theta > 0)
  structure( #strcutur is a bunch of data with 
    list( #bunch of data
      len_theta=len_theta,  
      beta_dist_h = init_beta_dist_for_h(len_theta),
      beta_dist_h_dash = init_beta_dist_for_h_dash(len_theta),
      base_dist = base_dist
    ),
    class = "bernp" #class attribute set so it's a class
  )  
}

#"Public" Constructor to create an instance of class bernp
bernp = function(len_theta=integer(), base_dist){
  new_bernp(as.integer(len_theta), base_dist)
}

# Computes the trafo h_y(y|x) out_bern is the (unconstrained) output of the NN 
bernp.eval_h = function(bernp, out_bern, y){
  theta_im = to_theta(out_bern)
  .eval_h(theta_im, y_i = y, beta_dist_h = bernp$beta_dist_h)
}

# Computes the trafo h'_y(y|x) out_bern is the (unconstrained) output of the NN 
bernp.eval_h_dash = function(bernp, out_bern, y){
  theta_im = to_theta(out_bern)
  .eval_h_dash(theta_im, y, beta_dist_h = bernp$beta_dist_h_dash)
}


# Computs NLL out_bern is the (unconstrained ) output of the NN 
bernp.nll = function(bernp, out_bern, y, y_range=1, out_eta = NULL) {
  theta_im = to_theta(out_bern)
  if (is.null(out_eta)){
    z = eval_h(theta_im, y_i = y, beta_dist_h = bernp$beta_dist_h)
  }else{
    hy = eval_h(theta_im, y_i = y, beta_dist_h = bernp$beta_dist_h)
    z = hy - out_eta[,1]
  }
  h_y_dash = eval_h_dash(theta_im, y, beta_dist_h_dash = bernp$beta_dist_h_dash)
  return(-tf$math$reduce_mean(bernp$stdnorm$log_prob(z) + tf$math$log(h_y_dash)) + log(y_range) )
}


