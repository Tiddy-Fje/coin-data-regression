## ----setup, include=FALSE------------------------------------------------
library(knitr)
opts_chunk$set(fig.path='figure/latex-', cache.path='cache/latex-')
options(width=70)

## ----echo=FALSE----------------------------------------------------------
 # Read in data and make time series
 maunaloa <- read.table("MaunaLoa.txt")
 co2 <- ts(maunaloa[,5],frequency=12,start=c(1958,3))

## ----figinit, echo=FALSE, fig.width=6, fig.height=5----------------------
# Plot data
plot(co2)

## ----STLdecomp, echo=FALSE, fig.width=6, fig.height=7--------------------
# STL decomposition
 par(mar=c(3.1,3.1,1.1,0.2),mgp=c(1.5,0.4,0),font.main=1,cex.main=0.8)
 plot(stl(co2,s.window=13))

## ----spectrum, echo=FALSE------------------------------------------------
# spectrum
par(mfrow=c(2,2))
par(mar=c(5.1,3.1,1.6,0.4),mgp=c(1.7,0.6,0),font.main=1,cex.main=0.8)
spectrum(co2,main="",ylim=c(10^(-5),10^2))
spectrum(co2,main="",spans=c(3),ylim=c(10^(-5),10^2))
spectrum(co2,main="",spans=c(5,5),ylim=c(10^(-5),10^2))
spectrum(co2,main="",spans=c(7,9),ylim=c(10^(-5),10^2))

## ----diff,  echo=FALSE, fig.width=7, fig.height=4------------------------
# data differenced
par(mar=c(5.1,3.1,1.6,0.4),mgp=c(1.7,0.6,0),font.main=1,cex.main=0.8)
co2.d <- diff(diff(co2),12)
# co2.d <- diff(co2.d)  # check if more differencing makes a difference: the fit is worse with another difference.
par(mfrow=c(1,2),pty="s")
acf(co2.d,ylim=c(-1,1))
pacf(co2.d,ylim=c(-1,1))

## ----STL, fig.keep='none', echo=FALSE, fig.width=7, fig.height=6---------
# STL of differenced data, not included in writeup
plot(stl(co2.d,s.window="periodic"))

## ----fits, echo=FALSE----------------------------------------------------
# fits of some SARIMA models
d <- 1  # vary differencing if we like
fit010010 <- arima(co2,order=c(0,d,0),seasonal=list(order=c(0,1,0),period=12))

fit110010 <- arima(co2,order=c(1,d,0),seasonal=list(order=c(0,1,0),period=12))
fit011010 <- arima(co2,order=c(0,d,1),seasonal=list(order=c(0,1,0),period=12))
fit010110 <- arima(co2,order=c(0,d,0),seasonal=list(order=c(1,1,0),period=12))
fit010011 <- arima(co2,order=c(0,d,0),seasonal=list(order=c(0,1,1),period=12))

fit110110 <- arima(co2,order=c(1,d,0),seasonal=list(order=c(1,1,0),period=12))
fit110011 <- arima(co2,order=c(1,d,0),seasonal=list(order=c(0,1,1),period=12))
fit011110 <- arima(co2,order=c(0,d,1),seasonal=list(order=c(1,1,0),period=12))
fit011011 <- arima(co2,order=c(0,d,1),seasonal=list(order=c(0,1,1),period=12))

fit111110 <- arima(co2,order=c(1,d,1),seasonal=list(order=c(1,1,0),period=12))
fit111011 <- arima(co2,order=c(1,d,1),seasonal=list(order=c(0,1,1),period=12))
fit011111 <- arima(co2,order=c(0,d,1),seasonal=list(order=c(1,1,1),period=12))
fit110111 <- arima(co2,order=c(1,d,0),seasonal=list(order=c(1,1,1),period=12))

fit111111 <- arima(co2,order=c(1,d,1),seasonal=list(order=c(1,1,1),period=12))

## ----diagnostics,  echo=FALSE--------------------------------------------
# model-checking for chosen models, not included in writeup
tsdiag(fit011011)
cpgram(residuals(fit011011))
qqnorm(residuals(fit011011))

tsdiag(fit111011)
cpgram(residuals(fit111011))
qqnorm(residuals(fit111011))

## ----predict, echo=FALSE, fig.width=6, fig.height=8----------------------
# plot parameters
par(mar=c(5.1,3.1,1.6,0.4),mgp=c(1.7,0.6,0),font.main=1,cex.main=0.8)
par(mfrow=c(3,2))
hl <- c(360,365,370,375,380,385,390,395,400,405,410,415,420)
vl <- 2000:2020
lw <- 0.2
pred <- predict(fit110011,133)
plot(window(co2,start=2001),ylim=c(365,410),xlim=c(2001,2020),main="110011",ylab="CO2 (ppmv)")
abline(h=hl,col="grey",lwd=lw)
abline(v=vl,col="grey",lwd=lw)
lines(pred$pred,col="blue")
lines(pred$pred+2*pred$se,col="red")
lines(pred$pred-2*pred$se,col="red")

pred <- predict(fit011011,133)
plot(window(co2,start=2001),ylim=c(365,410),xlim=c(2001,2020),main="011011",ylab="CO2 (ppmv)")
abline(h=hl,col="grey",lwd=lw)
abline(v=vl,col="grey",lwd=lw)
lines(pred$pred,col="blue")
lines(pred$pred+2*pred$se,col="red")
lines(pred$pred-2*pred$se,col="red")

pred <- predict(fit111011,133)
plot(window(co2,start=2001),ylim=c(365,410),xlim=c(2001,2020),main="111011",ylab="CO2 (ppmv)")
abline(h=hl,col="grey",lwd=lw)
abline(v=vl,col="grey",lwd=lw)
lines(pred$pred,col="blue")
lines(pred$pred+2*pred$se,col="red")
lines(pred$pred-2*pred$se,col="red")

pred <- predict(fit011111,133)
plot(window(co2,start=2001),ylim=c(365,410),xlim=c(2001,2020),main="011111",ylab="CO2 (ppmv)")
abline(h=hl,col="grey",lwd=lw)
abline(v=vl,col="grey",lwd=lw)
lines(pred$pred,col="blue")
lines(pred$pred+2*pred$se,col="red")
lines(pred$pred-2*pred$se,col="red")

pred <- predict(fit110111,133)
plot(window(co2,start=2001),ylim=c(365,410),xlim=c(2001,2020),main="110111",ylab="CO2 (ppmv)")
abline(h=hl,col="grey",lwd=lw)
abline(v=vl,col="grey",lwd=lw)
lines(pred$pred,col="blue")
lines(pred$pred+2*pred$se,col="red")
lines(pred$pred-2*pred$se,col="red")

pred <- predict(fit111111,133)
plot(window(co2,start=2001),ylim=c(365,410),xlim=c(2001,2020),main="111111",ylab="CO2 (ppmv)")
abline(h=hl,col="grey",lwd=lw)
abline(v=vl,col="grey",lwd=lw)
lines(pred$pred,col="blue")
lines(pred$pred+2*pred$se,col="red")
lines(pred$pred-2*pred$se,col="red")

