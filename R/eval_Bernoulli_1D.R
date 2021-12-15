#############################################################
# Creating the 1-D Conjugate Prior Figure
library(ggplot2)
#From https://bookdown.org/content/2015/figures.html
apatheme=theme_bw(base_size = 22)+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        text=element_text(family='Times'),
        legend.title=element_blank(),
        legend.position = c(0.15,0.85),
        axis.text.y=element_text(size = 20),
        axis.text.x=element_text(size = 20))


d = read.csv('runs/Bernoulli_1D/Bernoulli_1D.csv')
d$X = NULL
dat = data.frame(t(d))
colnames(dat) = c('pia', 'Analytical',  'M1','pi1', 'M10','pi10',  'M30', 'pi30', 'Gauss','piG')
plot(dat$pia, dat$Analytical)
lines(dat$pi1, dat$M1)
points(dat$pi10, dat$M10)
lines(dat$pi30, dat$M30)
lines(dat$piG, dat$Gauss)

df = data.frame(method='Analytical', pi = c(dat$pia,1), density = c(dat$Analytical,0))
df = rbind(df, data.frame(method='M1', pi = dat$pi1, density = dat$M1))
df = rbind(df, data.frame(method='M10', pi = dat$pi10, density = dat$M10))
df = rbind(df, data.frame(method='M30', pi = dat$pi30, density = dat$M30))
df = rbind(df, data.frame(method='Gauss', pi = dat$piG, density = dat$Gauss))

ggplot(df) + geom_line(aes(x=pi, y=density, col=method, linetype=method),size=1.2) + 
  xlab(expression(pi)) +
  scale_color_manual(values = c('red', 'orange', 'lightblue', 'blue', 'darkblue')) +
  apatheme
  
#Saving
ggsave('runs/Bernoulli_1D/conjugate_prior.pdf',width = 7, height=7/sqrt(2))
# ggsave('~/Dropbox/Apps/Overleaf/bernvi/images/conjugate_prior.pdf',width = 7, height=3.8)


#### Additional Figure
library(readr)
runs_M <- read_csv("runs/Bernoulli_1D/00_conj_prior_kl_eval_kl.csv")
runs_M$X1 = NULL
colnames(runs_M) = c('rep', 1,2,3,4,5,6,7,8,9,10,15,30,100,300,0.5)
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

ggplot(df) + 
  geom_boxplot(aes(x=M, y=KL, fill=type, group=cut_interval(x=M, length=0.25)))+
  #geom_point(aes(x=M, y=KL, col=type)) + 
  scale_x_log10() + 
  scale_y_log10() +
  apatheme

