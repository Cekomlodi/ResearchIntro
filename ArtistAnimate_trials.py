#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:16:51 2025

@author: corinnekomlodi
"""


import numpy as np
import matplotlib.pyplot as plt
import math as m
import random
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


runs = 500

def NormalCovGaussian(x):
    variance = np.array([[1,0],[0,1]])
    invVar = np.linalg.inv(variance)
    step1 = np.matmul(np.transpose(x), (invVar))
    px = (1/(((2*m.pi))*np.linalg.det(variance)**(-1/2)))*np.exp(-0.5*(np.matmul(step1,(x))))
    return px

def TwoDimAnimated(maximumchain, densityvariance, mean):
    x = np.random.multivariate_normal(mean, densityvariance)
    acceptedvalues_x = []
    acceptedvalues_y = []
    iteration = []
    iteration_trials = []
    trials_x = []
    trials_y = []
    for i in range(0, maximumchain):
        iteration.append(i)
        x_new = np.random.multivariate_normal(mean, densityvariance)
        HastingsRatio = NormalCovGaussian(x_new)/NormalCovGaussian(x)
        if np.any(random.random() < HastingsRatio):
            x = x_new
            acceptedvalues_x.append(x[0])
            acceptedvalues_y.append(x[1])
            
        else:
            iteration_trials.append(i)
            acceptedvalues_x.append(x[0])
            acceptedvalues_y.append(x[1])
            trials_x.append(x_new[0])
            trials_y.append(x_new[1])
    
    return acceptedvalues_x, acceptedvalues_y, iteration, iteration_trials, trials_x, trials_y

densityvariance = ([[2, 1.2],[1.2, 2]])
mean = [np.random.normal(), np.random.normal()]
xgood,ygood, iteration, iteration_trials, trials_x, trials_y = TwoDimAnimated(runs, densityvariance, mean)


gs_kw = dict(width_ratios=[3, 1, 1], height_ratios=[.75, .75])
#layout_pattern = [['upper left', 'right']]
layout_pattern = [['upper left', '.', 'right'], ['lower left', 'lower middle', 'lower right']]
fig, axd = plt.subplot_mosaic(layout_pattern,
                              gridspec_kw=gs_kw, figsize=(15, 6), constrained_layout=True)
fig.suptitle('plt.subplot_mosaic()')

axd['upper left'].plot(iteration_trials, trials_x, 'o', markersize=1, color = 'red')
axd['lower left'].plot(iteration_trials, trials_y, 'o', markersize=1, color = 'red')
axd['right'].hist(xgood, density = True, bins = 16, color= 'grey')
axd['lower middle'].hist(ygood, density = True, bins = 16, color= 'grey')
axd['lower right'].scatter(xgood, ygood, alpha = .3, s = 20)
axd['upper left'].plot(iteration, ygood)
axd['lower left'].plot(iteration, xgood)
axd['upper left'].set_xlim([0,runs])
axd['lower left'].set_xlim([0,runs])
axd['upper left'].set_ylim([-4,4])
axd['lower left'].set_ylim([-4,4])
fig.suptitle('Two Variable MCMC', y = .91, size = 'xx-large')
plt.show()





