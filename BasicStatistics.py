#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:31:15 2025

@author: corinnekomlodi
"""

#The mean is defined by the sum of the terms divided by the total number of terms

#The variance is the sum of values minus the mean squared divided by n-1

#skew is the measure of shift of the data defined by mean minus the median divided by the standard deviation

#kurtosis is the measure of the tailedness of a distribution relative to a normal distribution


import numpy as np
import matplotlib.pyplot as plt
import math as m

n = np.arange(1,11,1)
k = 4**n
mean1 = []
variance1 = []
skew1 = []
kvalues = []

for i in range(0,len(n)):
    array = np.random.rand(1, k[i])
    kvalues.append(array[0])
    mean = np.sum(array[0])/len(array[0])
    variance = np.sum((array-mean)**2)/(len(array[0])-1)
    skew = (mean-np.median(array[0]))/variance
    mean1.append(mean)
    variance1.append(variance)
    skew1.append(skew)
    
plt.plot(1/k, mean1, label = 'mean')
plt.plot(1/k, variance1, label = 'variance', color = 'green')
plt.axhline(y = .5, label = 'top hat mean', color = 'red', linestyle = '--')
plt.axhline(y = 1/12, label = 'top hat variance', color = 'pink', linestyle = '--')
#plt.plot(1/k, skew1, label = 'skew')
plt.legend()
plt.xlabel('1/k')
plt.ylabel('value')

#Now for basic M-H MCMC
#1. draw a proposal theta prime from the proposal pdf q(thetaprime|thetak)
#2. Draw a random number 0<r<1 from the uniform distribution
#3. if f(thetaprime)/f(thetak) > r then thetaprime becomes the newest sample, else it gets discarded

#gaussian density function with mean 2 and variance 2 (note that variance is the square of standard dev)
# =============================================================================
# mea = 2
# stdev = m.sqrt(2)
# propdev = m.sqrt(1)
# x = np.random.random_sample()
# x1 = 0
# 
# for i in range(0, 10**4):
#     propdist = 1/(propdev*m.sqrt(2*m.pi))*m.exp(-0.5*((x1-x)/propdev)**2) 
#     px = 1/(stdev*m.sqrt(2*m.pi))*m.exp(-.5*((propdist-mea)/stdev)**2)
#     if px/propdist < x:
#         px = x
# =============================================================================

#for the love of god, graph a gaussian

x = np.linspace(-1,5, num = 50)
gausspx = ([])
for i in range(0, len(x)):
    px = (1/(m.sqrt(2)*m.sqrt(2*m.pi)))*m.exp(-0.5*((x[i] - 2)/m.sqrt(2))**2)
    gausspx.append(px)
    
fig, ax = plt.subplots(figsize = (20,10))
ax.plot(x, gausspx, label = 'standard gauss')
ax.axvline(x = 2, linestyle = '--')
        


