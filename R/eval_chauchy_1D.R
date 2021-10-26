#############################################################
# Creating the Chauchy Example Figure
library(ggplot2)
#From https://bookdown.org/content/2015/figures.html
apatheme=theme_bw(base_size = 22)+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        legend.title=element_blank(),
        legend.position = c(0.15,0.75),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))

library(data.table)
mcmc_densities = fread("~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/mcmc_densities.csv.gz")
#mcmc_densities = fread("~/Google Drive/tmp/transfer/mcmc_densities.csv.gz")
df = data.frame(method='MCMC', pi = mcmc_densities$w, density = mcmc_densities$lp__)
d =  fread("~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/TM-VI, M=1_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M1', pi = d$V1, density = d$V2))
d =  fread("~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/TM-VI, M=10_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M10', pi = d$V1, density = d$V2))
d =  fread("~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/TM-VI, M=30_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M30', pi = d$V1, density = d$V2))
d =  fread("~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/Gauss-VI_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='Gauss', pi = d$V1, density = d$V2))
d =  fread("~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/TM-VI, M=50_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M50', pi = d$V1, density = d$V2))

ggplot(df) + geom_line(aes(x=pi, y=density, col=method, linetype=method),size=1.2) + 
  xlab(expression(xi)) +
  scale_color_manual(values = c('orange', 'lightblue', 'darkblue', 'blue', 'darkblue', 'red')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'dashed', 'dotted', 'solid')) +
  apatheme
#Saving
ggsave('~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Chauchy_Example/02_cauchy_posterior.pdf',width = 7, height=3.8)
ggsave('~/Dropbox/Apps/Overleaf/bernvi/images/02_cauchy_posterior.pdf',width = 7, height=3.8)



library(loo)
d =  fread("~/Google Drive/tmp/transfer/TM-VI, M=1_samples.csv.gz")
df = data.frame(M=1, k=loo::psis(d$ll + d$lp - log(d$lq))$diagnostics$pareto_k)
d =  fread("~/Google Drive/tmp/transfer/TM-VI, M=5_samples.csv.gz")
df = rbind(df, data.frame(M=5, k=loo::psis(d$ll + d$lp - log(d$lq))$diagnostics$pareto_k))
d =  fread("~/Google Drive/tmp/transfer/TM-VI, M=10_samples.csv.gz")
df = rbind(df, data.frame(M=10, k=loo::psis(d$ll + d$lp - log(d$lq))$diagnostics$pareto_k))
d =  fread("~/Google Drive/tmp/transfer/TM-VI, M=40_samples.csv.gz")
df = rbind(df, data.frame(M=40, k=loo::psis(d$ll + d$lp - log(d$lq))$diagnostics$pareto_k))
d =  fread("~/Google Drive/tmp/transfer/TM-VI, M=50_samples.csv.gz")
df = rbind(df, data.frame(M=50, k=loo::psis(d$ll + d$lp - log(d$lq))$diagnostics$pareto_k))


df 

plot(df$M, df$k)


