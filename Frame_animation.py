#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:13:44 2025

@author: corinnekomlodi
"""

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

### All about MCMC
runs = 10000

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



### All about the plot dimension
gs_kw = dict(width_ratios=[3, 1, 1], height_ratios=[.75, .75])
#layout_pattern = [['upper left', 'right']]
layout_pattern = [['upper left', 'middle', 'right'], ['lower left', 'lower middle', 'lower right']]
fig, axd = plt.subplot_mosaic(layout_pattern,
                              gridspec_kw=gs_kw, figsize=(15, 6), constrained_layout=True)
fig.suptitle('plt.subplot_mosaic()')
axd['upper left'].set_xlim([0,runs])
axd['lower left'].set_xlim([0,runs])
axd['upper left'].set_ylim([-4,4])
axd['lower left'].set_ylim([-4,4])
fig.suptitle('Two Variable MCMC', y = 1.05, size = 'xx-large')






### All about animation
bin_edges = np.linspace(-4, 4, 17)
writer = FFMpegWriter(fps=100, metadata=dict(artist='Me'))
l, = axd['upper left'].plot([],[], color = 'steelblue')
l2, = axd['lower left'].plot([],[], color = 'steelblue')
xlist = []
ylist = []
xllist = []
yllist = []

#make it rely on data
axd['lower middle'].set_xlim([min(bin_edges), max(bin_edges)])
axd['right'].set_xlim([min(bin_edges), max(bin_edges)]) 
axd['lower right'].set_xlim([min(xgood),max(xgood)])
axd['middle'].set_xlim([min(xgood),max(xgood)])

axd['lower middle'].set_ylim([0,.4])
axd['right'].set_ylim([0,.4])
axd['lower right'].set_ylim([min(ygood),max(ygood)])
axd['middle'].set_ylim([min(ygood),max(ygood)])


with writer.saving(fig, "AttemptDownsample5.mp4", 100):

    for i in range(0, len(xgood)):
        # Lists of Data one-by-one
        xlist.append(i)
        ylist.append(xgood[i])
        yllist.append(ygood[i])
        #axd['right'].set_ylim([min(ygood), max(ygood)])
        #axd['lower right'].set_ylim([min(xgood), max(xgood)])
        plt.show()
        
        #Only takes 1 every 10 to help with run-time
        if i%10 ==0:
            # Histograms
            histy, binsy = np.histogram(ylist, bins = 16)
            densityy = histy/(sum(histy)*(bin_edges[1]-bin_edges[0]))
            axd['right'].stairs(densityy, binsy, color = "grey", fill = True)

            histx, binsx = np.histogram(yllist, bins = 16)
            densityx = histx/(sum(histx)*(bin_edges[1]-bin_edges[0]))
            axd['lower middle'].stairs(densityx, binsx, color = "grey", fill = True)
            axd['middle'].hist2d(yllist, ylist, bins = 16, cmap = 'Blues')

            # Scatter
            axd['lower right'].scatter(ylist, yllist, color = 'steelblue', alpha = .3)

            # Set line Data
            l.set_data(xlist,ylist)
            l2.set_data(xlist,yllist)
        
            #Grab the frame
            writer.grab_frame()
        
            #Clear the ones that have to be reset each time
            #axd['right'].clear()
            #axd['lower right'].clear()
            #axd['lower middle'].clear()
            #axd['middle'].clear()
        else:
            continue



