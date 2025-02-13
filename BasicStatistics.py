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
import random

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

#for the love of god, graph a gaussian
#lovely, alright.

# =============================================================================
#Lets write out the parameters and equations

# x = 0
# xprime = np.linspace(-1,5, num = 50)
# uniformdist = gausspx
# r = np.random.sample(uniformdist, 1)
# numberofsamples = range(0,10**4)
# proposaldist = (1/(m.sqrt(1)*m.sqrt(2*m.pi)))*m.exp(-0.5*((xprime - x)/m.sqrt(1))**2)
# thetaprime = np.random.sample(proposaldist,1)
# acceptableness = np.log(thetaprime)-np.log(thetak) >> np.log(r)
# =============================================================================

#rewriting everything above but furture proofed

def target(x):
    px = (1/(m.sqrt(2)*m.sqrt(2*m.pi)))*m.exp(-0.5*((x - 2)/m.sqrt(2))**2)
    return px
    
def ProposalDraw(x):
    return np.random.normal(x,1)
    
def ProposalDist(v,x):#x represents the current guess
    pd = (1/(m.sqrt(1)*m.sqrt(2*m.pi)))*m.exp(-0.5*((v - x)/m.sqrt(1))**2)
    return pd


def MHMCMC(minimumguess, maximumchain, initial_guess):
    x = initial_guess
    accepted_values = []
    for i in range(minimumguess,maximumchain):
        #draw a random proposal
        new_x = ProposalDraw(x)
        #create a ratio
        acceptablenessRatio = target(new_x)/target(x)
        if random.random() < acceptablenessRatio:
            x = new_x
            accepted_values.append(x)
    return accepted_values

accepted_values = MHMCMC(0,10**4, 0)
        
x_range = np.linspace(-2,6, num = 50)
gausspx = ([])
for i in range(0, len(x_range)):
    px = (1/(m.sqrt(2)*m.sqrt(2*m.pi)))*m.exp(-0.5*((x_range[i] - 2)/m.sqrt(2))**2)
    gausspx.append(px)


fig, ax = plt.subplots(figsize = (10,5)) 
ax.hist(accepted_values, density = True, bins = 50, color = 'silver') #The density section is important
ax.plot(x_range, gausspx, label = 'standard gauss', color = 'teal')
ax.axvline(x = 2, linestyle = '--')
ax.set_title('Metropolis-Hastings Basic MCMC')
# =============================================================================
# 
# #Redo problem 2 with 3<x<7 and zero everywhere else, what change had to be made?
# 
# =============================================================================
def flatDist(range_start, range_end, num, value):
    y_values = []
    for i in np.linspace(range_start, range_end, num):
        if 4 < i < 8:
            y1 = (1/4)*value
            y_values.append(y1)
        else:
            y2 = 0
            y_values.append(y2)
    return y_values

def flatTarget(x):
    if 3<=x or x <=7:
        x_value = 1
    else:
        x_value = 0
    return x_value

def MHMCMC_flat(maximumchain):
    xi = np.random.uniform(3,7)
    new_x_samples = []
    accepted_valuesflat = []
    for i in range(0,maximumchain):
        #draw a random proposal
        yi = np.random.uniform(3,7)
        new_x_samples.append(yi)
        #make the ratio
        k = flatTarget(yi)/flatTarget(xi) 
        #see if the ratio fits
        if random.random() > k: #alpha greater than k
            accepted_valuesflat.append(yi)
            xi = yi
        else:
            accepted_valuesflat.append(xi)
    return accepted_valuesflat, new_x_samples


normalDist = flatDist(1,10,1000, 1)
flat_range = np.linspace(0,9, num = len(normalDist))

flatvalues, x_samples = MHMCMC_flat(10**4)

fig, ax = plt.subplots(figsize = (10,5)) 
ax.hist(flatvalues, density = True, bins = 16, color = 'silver') #The density section is important
ax.plot(flat_range, normalDist, label = 'flat dist', color = 'teal')
ax.set_xticks([1,2,3,4,5,6,7,8,9])
ax.set_xlim([1,9])
ax.set_title('M-H MCMC Top Hat')


fig, ax = plt.subplots(figsize = (10,5)) 
ax.hist(x_samples, density = True, bins = 16, color = 'silver')
ax.plot(flat_range, normalDist, label = 'flat dist', color = 'teal')
ax.set_xticks([1,2,3,4,5,6,7,8,9])
ax.set_xlim([1,9])

#use the python random number as my proposal

# =============================================================================
#
# number 4
# 
# =============================================================================

#use random number generator that pulls from a gaussian for 4
#making a random number generator without a list
        


