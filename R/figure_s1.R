#### Additional Figure
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
  scale_x_log10() + 
  scale_y_log10(name=expression(paste('KL(q(',theta,') || p(',theta,'|D))'))) +
  annotate("text", x = 20, y = 0.9/20, label = "~1/M", size=6) +
  annotate("text", x = 3,  y = 0.8, label = "Cauchy example", size=6) +
  apatheme
gg_chauchy


library(readr)
runs_M <- read_csv("runs/Bernoulli_1D/00_conj_prior_kl_eval_kl_all_random.csv")[1:20,]
runs_M$X1 = NULL
colnames(runs_M) = c('rep', 1,2,3,4,5,6,7,8,9,10,15,30,100,300,0.5)
#c('rep', 0.5, 1,2,3,6,10,20,30,60,100,300,200)
#runs_M$`4`=NULL
runs_M$`5`=NULL
#runs_M$`6`=NULL
runs_M$`7`=NULL
runs_M$`8`=NULL
runs_M$`9`=NULL
runs_M$`15`=NULL
#runs_M$`30`=NULL
#runs_M2 <- read_csv("runs/Bernoulli_1D/00_conj_prior_kl_eval_kl.csv")[1:20,]

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

x = c(6,10,20)
df2 = data.frame(
  x = x,
  y = 0.05/x
)

gg_bernoulli = ggplot(df) + 
  geom_boxplot(aes(x=M, y=KL, fill=type, group=cut_interval(x=M, length=0.25)))+
  #geom_point(aes(x=M, y=KL, col=type)) + 
  geom_jitter(aes(x=M, y=KL, col=type),alpha=0.9) +
  geom_line(data=df2, aes(x=x,y=y), linetype=2, size=1.2) +
  scale_x_log10() + 
  scale_y_log10(name=expression(paste('KL(q(',theta,') || p(',theta,'|D))'))) +
  annotate("text", x = 20, y = 0.1/20, label = "~1/M", size=6) +
  #annotate("text", x = 3,  y = 0.05, label = "Bernoulli example", size=6) +
  apatheme 
gg_bernoulli
ggsave('kl_vs_m_bern.pdf',gg_bernoulli)
ggsave(gg_bernoulli, filename = '~/Dropbox/Apps/Overleaf/bernvi/images/kl.vs.m.bern.pdf')


library(ggpubr)
library(gridExtra)
comb = grid.arrange(
  gg_bernoulli, gg_chauchy,
  widths=c(0.5, 0.5), ncol=2, nrow=1)
ggsave('KlMChauchy.pdf',width = 21, height = 6,comb)
ggsave(comb, filename = '~/Dropbox/Apps/Overleaf/bernvi/images/KlMChauchy.pdf',width = 21, height=6)





