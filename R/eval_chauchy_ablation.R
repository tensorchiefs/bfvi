#setwd("~/Documents/workspace/bfvi/R")

library(readr)
library(ggplot2)
library(data.table)
library(keras)
library(tensorflow)
library(tfprobability)
source('bern_utils.R')
source('eval_utils.R')
library(readr)
library(dplyr)

mcmc_df =  fread("runs/Cauchy_1D/mcmc_densities.csv.gz")[-1,]
mcmc_df = mcmc_df %>%  arrange(w) #Sorting

Ms = c(2,3,6,10,20,30,50,60)#,100,200,300)
#Ms = c(50)
#M = 1
method = 'F1F2'
#dir = "~/Documents/workspace/bfvi/bfvi/runs/cauchy_ablation/"
dir = "../bfvi/runs/cauchy_ablation/"
#Checking if all files exists 
for (M in Ms){
  param_fn = paste0(dir,"cauchy_eval_ablation_M_", M, "_", method, "_params.csv")
  print(file.exists(param_fn))
}

########## Main Loop #######
df_plot = data.frame(M=0, method='MCMC', x = mcmc_df$w, density = mcmc_df$lp__, seed=1)
df_kl = NULL
for (M in Ms){
  param_fn = paste0(dir,"cauchy_eval_ablation_M_", M, "_", method, "_params.csv")
  print(paste0("Starting with M ", M, "file", param_fn))
  params <- read_csv(param_fn, col_names = TRUE)
  params = params[,-1] 
  M1 = ncol(params) - 5 + 1 #M+1
  colnames(params) = c('seed','epoch','az','bz', paste0('theta_p', 1:M1))
  seeds = unique(params$seed)
  # theta = to_theta(tf$reshape(tf$Variable(as.numeric(params[line,5:ncol(params)])),c(1L,-1L)))
  beta_dists_for_h = init_beta_dist_for_h(M1)
  
  for (seed in seeds){
    line = which(params$seed == seed & params$epoch!='initial')
    theta = to_theta(tf$reshape(tf$Variable(as.numeric(params[line,5:ncol(params)])),c(1L,-1L)))
    num_samples = 5000L
    df_w_logqw = get_w_logqw(theta = theta, num_samples = num_samples,
                             a_np = params$az[line], 
                             b_np = params$bz[line], 
                             trafo = method) %>% 
      arrange(w) #Sorting
    
    
    df_plot = rbind(df_plot, data.frame(M = M1 - 1,
      method=method, x = df_w_logqw$w, density = exp(df_w_logqw$log_q_w), seed=seed))
  }
  
  #Calculation of KL Divergence (might take some time)
  if(FALSE){
    for (seed in seeds){
      line = which(params$seed == seed & params$epoch!='initial')
      num_samples = 1e5L
      df_w_logqw = get_w_logqw(theta = theta, num_samples = num_samples,
                               a_np = params$az[line], 
                               b_np = params$bz[line], 
                               trafo = method)
      kl = eval_kl_cauchy(df_w_logqw$w, df_w_logqw$log_q_w)
      df_kl_c = data.frame(seed, kl, M=M1-1, method, num_samples)
      if (is.null(df_kl)) {
        df_kl = df_kl_c
      } else{
        df_kl = rbind(df_kl, df_kl_c)
      }
    }
  }
  
}#M in Ms
str(df_kl$M)
if (FALSE){
  df_kl_SigmoidF2 = df_kl 
  write_csv(df_kl_SigmoidF2, file='df_kl_SigmoidF2.csv')
}
if (FALSE){
  df_kl_F1F2 = df_kl 
  write_csv(df_kl_F1F2, file='df_kl_F1F2.csv')
}
if (FALSE){
  df_kl_TruncF2 = df_kl 
  write_csv(df_kl_TruncF2, file='df_kl_TruncF2.csv')
}



########## produce kl_vs_M_methods_chauchy plot #######

df_kl = read_csv('df_kl_F1F2.csv')
df_kl = rbind(df_kl, read_csv('df_kl_SigmoidF2.csv'))
df_kl = rbind(df_kl, read_csv('df_kl_TruncF2.csv'))
#df_kl$M = as.numeric(as.numeric(as.character(df_kl$M)))
if (FALSE){
  write_csv(df_plot, file='df_plot_M50.csv')
  df_plot = read_csv(file='df_plot_M50.csv')
  df_plot$M = NULL
}
nrow(df_kl)


####### KL - Dependence on M #####
apatheme=theme_bw(base_size = 22)+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        legend.title=element_blank(),
        legend.position = c(0.8,0.8),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))


x = 3:60
df2 = data.frame(
  x = x,
  y = 1.2/x
)

eval_kl_cauchy(mcmc_df$w, log(mcmc_df$lp__))
## Adding Gaus #####
runs_M <- read_csv("runs/Cauchy_1D/01_single_weight_kl_eval_kl.csv")
df_kl = rbind(df_kl,
              data.frame(seed = runs_M$seed, kl=runs_M$`Gauss-VI`, M=0.5, method='Gauss-VI', num_samples=1000))

dat = df_kl %>% 
  filter(M < 61009) 
  #filter(method == 'F1F2' | method == 'Gauss-VI')

gg_chauchy = ggplot(dat) + 
  geom_jitter(aes(x=M, y=kl, col=method),alpha=0.9) +
  #geom_boxplot(aes(x=M, y=kl, fill=method, group=cut_interval(x=M, length=0.5))) + 
  #geom_line(data=df2, aes(x=x,y=y), linetype=2, size=1.2) +
  scale_x_log10() + 
  ylab(expression(paste('KL(q(',theta,') || p(',theta,'|D))')))+
  #scale_y_log10(name=expression(paste('KL(q(',theta,') || p(',theta,'|D))'))) +
  #annotate("text", x = 20, y = 2/20, label = "~1/M", size=6) +
  scale_color_manual(
    name = 'Method',
    values = c(
      "Gauss-VI" = "green",
      "F1F2" = "darkblue",
      "SigmoidF2" = "turquoise",
      "TruncF2" = "brown"),
    labels = c('Gauss-VI', 'N-F1F2','N-SigmoidF2','TruncN-F2')
  )  +
  apatheme
gg_chauchy
ggsave('kl_vs_M_methods_chauchy.pdf',gg_chauchy)
ggsave(gg_chauchy, width = 7, height=4.8, filename = '~/Dropbox/Apps/Overleaf/bernvi/images/kl_vs_M_methods_chauchy.pdf')

#pkl_k10 = pkl 

####### q_w Dependence on M #####
apatheme=theme_bw(base_size = 22)+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        legend.title=element_blank(),
        legend.position = c(0.2,0.85),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))

df_plot$seed = as.factor(df_plot$seed)
library(dplyr)
library(magrittr)
#idx = sort(sample(1:nrow(df_plot), 5e4))
idx = 1:nrow(df_plot)
df_p = df_plot[idx,]
dd = df_p %>% filter(seed %in% c(1,2)) 


p = ggplot(dd) + 
  geom_line(aes(x=x, y=density, col=factor(method)),size=1) + 
  #geom_line(data=mcmc_df, aes(x=w, y = lp__)) +
  xlab(expression(xi)) +
  scale_color_manual(values = c("F1F2" = "darkgreen",
                                "SigmoidF2" = "steelblue",
                                "TruncF2" = "pink",
                                "MCMC" = "black"
                                )) +
  apatheme
p
ggsave('cauchy_M50_methods.pdf', p)

###########  produce cauchy_F1F2_M_comparison plot ###########
apatheme=theme_bw(base_size = 22)+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        #legend.title=element_blank(),
        legend.position = c(0.2,0.75),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))

d =  fread("runs/Cauchy_1D/Gauss-VI_densities.csv.gz")[-1,]
df_p = rbind(df_plot, data.frame(M=0.5, method='Gauss-VI', x=d$V1, density=d$V2, seed=1))
dd = df_p %>% filter(method %in% c('F1F2', 'Gauss-VI', 'MCMC')) %>% 
  filter(M %in% c(0.5,2,6,10,30,50,0)) %>% 
  #sample_n(1e4) %>% 
  filter(as.numeric(seed) < 11) %>% arrange(method, factor(M), x) %>% subset(method != 'MCMC')

dfmcmc = subset(dd, method == 'MCMC') %>% sample_n(1e3)

library(dplyr)

p = ggplot(dd) + 
  geom_line(aes(x=x, y=density, col=factor(M)),size=0.75) + 
  geom_line(data = dfmcmc, aes(x=x, y=density, col=factor(M)), linetype=2, size=1) +
  #geom_line(data=mcmc_df, aes(x=w, y = lp__)) +
  xlab(expression(xi)) + 
  scale_color_manual(
    name = 'Method',
    values = c(
      "0.5" = "green",
      "2" = "darkgreen",
      "6" = "steelblue",
      "10" = "pink",
      "30" = "grey",
      "50" = "darkblue",
      "0" = "red"),
    labels = c('Gauss-VI', 'M=2','M=6','M=10','M=30','M=50', 'MCMC')
  ) +
  scale_linetype(guide="none") +
  apatheme
p
ggsave('cauchy_F1F2_M_comparison.pdf', p,width = 7, height=4.8)
ggsave(p, width = 7, height=4.8,filename = '~/Dropbox/Apps/Overleaf/bernvi/images/cauchy_F1F2_M_comparison.pdf')

