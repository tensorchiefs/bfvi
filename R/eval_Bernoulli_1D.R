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


d = read.csv('~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Conjugate_Prior/Conjugate_Prior_Exp.csv')
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
ggsave('~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/tmvi/runs/Conjugate_Prior/conjugate_prior.pdf',width = 7, height=7/sqrt(2))
ggsave('~/Dropbox/Apps/Overleaf/bernvi/images/conjugate_prior.pdf',width = 7, height=3.8)