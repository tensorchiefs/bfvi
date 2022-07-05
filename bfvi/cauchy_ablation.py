print("Hallo")
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import initializers
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import stan
import scipy.stats as stats
import seaborn as sns
import tqdm
import sys
import pandas as pd
from bfvi.vimlts_fast import VimltsLinear

###### Global Vars, komme nicht in der Programmierer Himmel ############
y = np.array((1.2083935, -2.7329216, 4.1769943, 1.9710574, -4.2004027, -2.384988))
ytensor = y.reshape([len(y),1])
# Prior
prior_dist=tfd.Normal(loc=0.,scale=1.)
num = len(y)
sigma = 1.
stan_code = """
        data{
          int<lower=0> N;
          real<lower=0> sigma;
          vector[N] y;
        }
        parameters{
          real w;
        }
        model{
          y ~ cauchy(w, sigma);
          w ~ normal(0, 1);
        }
"""

class LogKL(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epochs'] = epoch
        # kl = [layer.losses[0] for layer in self.model.layers]
        # # logs['kl'] = tf.reduce_sum(kl)
        # logs['kl'] = kl

def scheduler(epoch, lr, lr_start, lr_stop, epochs):
    if epoch > epochs:
        return lr_stop
    else:
        return lr_start + (lr_stop-lr_start)*(epoch/epochs)

@tf.function
def sample_nll(y_obs, y_pred, scale=sigma):
    """
    Args:
        y_obs: true labels. Expected shape (#batch, 1) or (#batch)
        y_pred: model prediction. Expected shape (#samples, #batch, 1) or (#samples, #batch)

    Returns: sum of Nll
    """
    if len(y_pred.shape) == 2:  # Bug tf?! If we have a single output it squeezes y_pred. I did not want this behaviour.
        y_pred = y_pred[...,None]
    tf.debugging.check_numerics(y_pred, "Prediction for nll computation contains NaNs or Infs")
    error_str = f"Expected one of the above defined shapes. Got shapes: y_obs: {y_obs.shape}; y_pred: {y_pred.shape}"
    assert y_pred.shape[-1] == y_obs.shape[-1] or ((len(y_pred.shape) == 3) and y_pred.shape[-1] == 1), error_str

    # dist = tfp.distributions.Normal(loc=y_pred, scale=scale)
    dist = tfd.Cauchy(loc=y_pred, scale=scale)
    nll_per_sample = -dist.log_prob(y_obs)
    nlls = tf.reduce_mean(nll_per_sample, axis=0)
    tf.debugging.check_numerics(nlls, "NLL contains NaNs or Infs")
    return tf.reduce_sum(nlls)

def theta_prime(M, w_min, w_max, base_dist):
    #M = 10
    #w_min = -2
    #w_max = 5
    #base_dist = tfp.distributions.TruncatedNormal(loc=0.5,scale=0.15,low=0.,high=1.)
    #np.linspace(0,1,num=M)
    d = base_dist.cdf(np.linspace(0,1,num=M))*(w_max - w_min)+w_min
    delta = np.diff(d)
    theta_prime = np.hstack((d[0].numpy(), tfp.math.softplus_inverse(delta)))
    return theta_prime

from scipy.stats import cauchy, norm
# Compute posterior from samples
def posterior_unnormalized(samples, data, prior_dist=tfd.Normal(loc=0, scale=1.)):
    assert len(samples.shape) == 1, "Expected samples to be one dimensional"
    log_liks = np.sum(cauchy.logpdf(data.reshape(1,-1), loc=samples.reshape(-1,1), scale=sigma), axis=1)
    priors = prior_dist.log_prob(samples).numpy()
    post_un = log_liks + priors
    sort = np.argsort(samples)
    samples_s = samples[sort]
    post_un_s = post_un[sort]
    return post_un_s

def posterior_unnormalized_single(xi, data=ytensor, prior_dist=tfd.Normal(loc=0, scale=1.)):
    log_liks = np.sum(cauchy.logpdf(data.reshape(1,-1), loc=xi, scale=sigma), axis=1)
    priors = prior_dist.log_prob(xi).numpy()
    post_un = log_liks + priors
    return np.exp(post_un)

from scipy import integrate
A, e = integrate.quad(posterior_unnormalized_single, -np.inf, np.inf, epsabs=1e-40)
log_p_D = np.log(A)
print(f"log_p_D with quad: {log_p_D}, log(error):{np.log(e)}")

class My_VimltsLinear(VimltsLinear):

    def f_1(self, z):
        return z

def run_mcmc():
    mcmc_data = {'N': num,
                 'sigma': sigma,
                 'y': y}
    model=stan.build(stan_code, data=mcmc_data)
    d = model.sample(num_chains  = 4,num_samples=1000)
    print("has")

def createModels():
    # Prior
    prior_dist=tfd.Normal(loc=0.,scale=1.)

    ytensor = y.reshape([len(y),1])

    def softplus_inv(y):
        return np.log(np.exp(y) - 1)

    models = {}
    theta_start = -5
    theta_stop = 5
    # Number of samples psi to approximate the expected value
    num_samples=10000
    Ms = [30]
    for M in Ms:
        # init params
        theta_starts = theta_prime(M, theta_start, theta_stop, prior_dist)
        kernel_initializers=dict(kernel_init_alpha_w = initializers.Constant(1.),
                                 kernel_init_beta_w = initializers.Constant(0.),
                                 kernel_init_alpha_z = initializers.Constant(1.),
                                 kernel_init_beta_z = initializers.Constant(0.),
                                 kernel_init_thetas = [initializers.Constant(theta_start)] + [initializers.Constant(softplus_inv((theta_stop-theta_start)/(M))) for i in range(M)])
        # define model
        tf.random.set_seed(2)
        layer = VimltsLinear(1,
                             activation=lambda x: x,
                             **kernel_initializers,
                             num_samples=num_samples,
                             prior_dist=prior_dist,
                             input_shape=(1,))
        model = tf.keras.Sequential([layer], name=f"VIMLTS-degree{M}")
        model.build(input_shape=(None, 1))
        model.summary()
        models[f"TM-VI, M={M}"] = model
    return models

def fit_models(models):
    # Learning rate
    epochs = 10000
    epoch_lr_end = epochs // 2
    for name, model in models.items():
        print(f"Start experiment with model {name}")
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(partial(scheduler, lr_start=lr_start, lr_stop=lr_end, epochs=epoch_lr_end))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=sample_nll, run_eagerly=False)
        model.fit(tf.ones(ytensor.shape), ytensor, epochs=epochs, verbose=False, callbacks=[LogKL()])
    return models

def save_models(models):
    for name, model in models.items():
        print(f"Saveing experiment with model {name}")
        model.save(f'models{name}')

def load_models(models):
    for name, model in models.items():
        print(f"loading experiment with model {name}")
        model = tf.keras.models.load_model(f'models{name}')
    return models

def show_plot(models):
    plt.figure(figsize=(13,6))
    #log_p_w_D_unnorm = posterior_unnormalized(w_mcmc.to_numpy(), ytensor, prior_dist=prior_dist)
    #kl_unnorm = np.mean(log_mcmc_p_w_D - log_mcmc_p_D - log_p_w_D_unnorm)
    #kl = log_p_D + kl_unnorm
    #Kls = dict(MCMC=kl)
    Kls = dict()
    #plt.plot(w_mcmc, np.exp(log_mcmc_p_w_D - log_mcmc_p_D), label=f"MCMC; KL{kl:.2e}", linewidth=8., color='g', linestyle=(0, (1, 1.5)))
    linestyles = [(0, (2, 2)), (0, (1, 1)), '-']
    ls = 0
    for name, model in models.items():
        print(f"ploting experiment with model {name}")
        layer = model.layers[0]
        w, log_qw = layer.sample(num=100000)
        w = w.numpy().squeeze()
        sort = np.argsort(w)
        w = w[sort]
        log_qw = log_qw.numpy().squeeze()[sort]
        # compute KL
        log_p_w_D_unnorm = posterior_unnormalized(w, ytensor, prior_dist=prior_dist)
        kl_unnorm = np.mean(log_qw - log_p_w_D_unnorm)
        # 1/T \sum_{w_i \sim q(w_i}} log(q(w_i)/p(w_i|D)) => log(p(D)) + 1/T \sum_{w_i \sim q(w_i}} log(q(w_i)) - log(p(D|w_i)*p(w_i))
        Kls[name] = 42.42 #log_p_D + kl_unnorm
        if "TM-VI" in name:
            plt.plot(w, np.exp(log_qw), label=f"{name}; KL:{Kls[name]:.2e}", linewidth=3., color='c', linestyle=linestyles[ls%3])
            ls += 1
        else:
            plt.plot(w, np.exp(log_qw), label=f"{name}; KL:{Kls[name]:.2e}", linewidth=3., color='peru')
    plt.legend(fontsize='small')
    # plt.title("Posterior distribution $p(w|D)$")
    plt.xlim((-4,4.))
    plt.xlabel("$\\xi$")
    plt.ylabel("$p(\\xi|D)$")
    plt.tight_layout()
    #display(pd.DataFrame.from_dict(Kls, orient='index', columns=["$KL(q(\\xi), p(\\xi|D)$"]).sort_values("$KL(q(\\xi), p(\\xi|D)$", ascending=False))
    plt.savefig("02_cauchy_posterior.pdf")
    plt.show()

def main():
    print(tf.__version__)
    print(tfp.__version__)



if __name__ == '__main__':
    main()
    #run_mcmc()
    models = createModels()
    models = fit_models(models)
    #models = save_models(models)
    #models = load_models(models)
    show_plot(models)





