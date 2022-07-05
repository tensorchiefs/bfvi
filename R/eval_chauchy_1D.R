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
mcmc_densities = fread('runs/Cauchy_1D/mcmc_densities.csv.gz')
#mcmc_densities = fread("~/Google Drive/tmp/transfer/mcmc_densities.csv.gz")
df = data.frame(method='MCMC', pi = mcmc_densities$w, density = mcmc_densities$lp__)
d =  fread("runs/Cauchy_1D/TM-VI, M=1_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M1', pi = d$V1, density = d$V2))
d =  fread("runs/Cauchy_1D/TM-VI, M=10_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M10', pi = d$V1, density = d$V2))
d =  fread("runs/Cauchy_1D/TM-VI, M=30_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M30', pi = d$V1, density = d$V2))
d =  fread("runs/Cauchy_1D/Gauss-VI_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='Gauss', pi = d$V1, density = d$V2))
d =  fread("runs/Cauchy_1D/TM-VI, M=50_densities.csv.gz")[-1,]
df = rbind(df, data.frame(method='M50', pi = d$V1, density = d$V2))

ggplot(df) + geom_line(aes(x=pi, y=density, col=method, linetype=method),size=1.2) + 
  xlab(expression(xi)) +
  scale_color_manual(values = c('orange', 'lightblue', 'darkblue', 'blue', 'darkblue', 'red')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'dashed', 'dotted', 'solid')) +
  apatheme
#Saving
ggsave('runs/Cauchy_1D/02_cauchy_posterior.pdf',width = 7, height=3.8)
#ggsave('~/Dropbox/Apps/Overleaf/bernvi/images/02_cauchy_posterior.pdf',width = 7, height=3.8)


# Plot for the supplemenatary region
library(readr)
runs_M <- read_csv("runs/Cauchy_1D/01_single_weight_kl_eval_kl.csv")
runs_M$X1 = NULL
colnames(runs_M) = c('rep', 0.5, 1,2,3,6,10,20,30,60,100,300,200)
library(reshape2)
df = melt(runs_M, id="rep")
df$M = as.numeric(as.numeric(as.character(df$variable)))
df$variable = NULL
df$KL = df$value
df$value = NULL
df$type = 'BF-VI'
df$type[df$M == 0.5] = 'Gauss'


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

x = c(6,10,20,30,40)
df2 = data.frame(
  x = x,
  y = 0.5/x
)

gg_chauchy = ggplot(df) + 
  #geom_point(aes(x=M, y=KL, col=type)) + 
  geom_boxplot(aes(x=M, y=KL, fill=type, group=cut_interval(x=M, length=0.25)))+
  geom_jitter(aes(x=M, y=KL, col=type),alpha=0.9) +
  geom_line(data=df2, aes(x=x,y=y), linetype=2, size=1.2) +
  #scale_x_log10() + 
  scale_y_log10(name=expression(paste('KL(q(',theta,') || p(',theta,'|D))'))) +
  annotate("text", x = 20, y = 0.9/20, label = "~1/M", size=6) +
  annotate("text", x = 3,  y = 0.8, label = "Cauchy example", size=6) +
  apatheme
gg_chauchy
ggsave('kl_vs_m_chauchy.pdf',gg_chauchy)
ggsave(gg_chauchy, filename = '~/Dropbox/Apps/Overleaf/bernvi/images/kl.vs.m.Chauchy.pdf')



x = c(1,2,3,6,10,20,30,60,100,300,200)
y = 200/(x+10)
y = 200/x - 200/x^2 
plot(x, y, log='xy')
l = lm(log(y) ~ log(x))
lines(x,y=exp(5.2)*x^(-1))

