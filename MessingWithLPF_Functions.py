#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 21:01:02 2025

@author: corinnekomlodi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import lpf_graph_functions as LPF
import lpf_graph_functions_altered as LPFA
import colormap as colormap


#%%
def ReadInStuff(dirpath):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import PatchCollection
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    import lpf_graph_functions as LPF
    import lpf_graph_functions_altered as LPFA
    import colormap as colormap
    
    df = pd.read_csv(dirpath + '/impactchain.dat', header = None, delimiter = '\s+',
	    names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])

    mednmom = np.median(df['mom'])
    mednmom = "{:.4e}".format(mednmom)
    
    return mednmom, df

dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1150509781" #1144228507

#%%

def AnimateSkyLoc(lat, long, downsample, mp4Name):
    
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[.75, .75])
    layout_pattern = [['upper left','right'], ['lower left', 'right']]
    fig, axd = plt.subplot_mosaic(layout_pattern,
                              gridspec_kw=gs_kw, figsize=(15, 6), constrained_layout=True)

    writer = FFMpegWriter(fps=100, metadata=dict(artist='Me'))
    xlist = np.arange(1, len(lat)+1, step = 1)
    l, = axd['upper left'].plot([],[], color = 'steelblue')
    l2, = axd['lower left'].plot([],[], color = 'steelblue')
    
    axd['upper left'].set_xlim([0, len(long)])
    axd['upper left'].set_ylim([min(long), max(long)])
    axd['lower left'].set_xlim([0, len(lat)])
    axd['lower left'].set_ylim([min(lat), max(lat)])
    
    axd['right'].set_xlim([min(long), max(long)])
    axd['right'].set_ylim([min(lat), max(lat)])
    
    with writer.saving(fig, mp4Name, 100):
        for i in range(0, len(lat)):
            
            if i % downsample == 0:
                #Chains
                l.set_data(xlist[:i],long.head(i))
                l2.set_data(xlist[:i],lat.head(i))
                #Histograms
                axd['right'].hist2d(long.head(i), lat.head(i), bins = 50, cmap = 'Blues')
                #Grab the frame
                writer.grab_frame()
                
                axd['right'].clear()
            else:
                continue
            
            
#%%

def D3scatterMovie(df, mp4name, downsample, burnin):
    from scipy.stats import gaussian_kde
    
    points = len(df['xloc'])-burnin
    
    fig = plt.figure(figsize = (15,5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    x = df['xloc'].tail(points)
    y = df['yloc'].tail(points)
    z = df['zloc'].tail(points)
    
    #calculate Density and colorbars
    data = np.vstack([x, y, z])
    
    # Calculate the density of the points using a Gaussian KDE
    kde = gaussian_kde(data)
    density = kde(data)  # Compute the density for each point
    
    # Normalize density for colormap
    norm = plt.Normalize(vmin=density.min(), vmax=density.max())
    
    scatter = ax.scatter(x, y, z, c=density, cmap='viridis', norm=norm)
    
    # Add a colorbar
    fig.colorbar(scatter)
    
    
    writer = FFMpegWriter(fps=70, metadata=dict(artist='Me'))
    with writer.saving(fig, mp4name, 100):       
        
        for i in range(0, len(points)):
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # Set labels and title
                ax.set_xlim([-1,1])
                ax.set_ylim([-1,1])
                ax.set_zlim([0,1])
                
                x1 = [.4,.9, .9, .4, -.4, -.9, -.9, -.4, .4]
                y1 = [1,.25,-.25,-1,-1,-.25,.25, 1, 1]
                Topz = [1,1,1,1,1,1,1,1,1]
                Botz = [0,0,0,0,0,0,0,0,0]

                ax.plot(x1, y1, Topz, 'k')
                ax.plot(x1, y1, Botz, 'k')
                
                for i in range(len(x1)):
                    ax.plot([x1[i], x1[i]], [y1[i], y1[i]], [Botz[i], Topz[i]], 'k')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Scatter Plot of Impact')
                
                if i % downsample == 0:
                    ax.scatter(x.head(i), y.head(i), z.head(i), c=density, cmap='viridis')
                    #take frame
                    writer.grab_frame()
                    #clear to make it run faster maybe
                    ax.clear()
                    
                else:
                    continue


#%%

def Boring3dScatter(df):
    x = df['xloc'].tail(70000)
    y = df['yloc'].tail(70000)
    z = df['zloc'].tail(70000)
    
    from scipy.stats import gaussian_kde
    
    fig = plt.figure(figsize = (15,5))
    ax = fig.add_subplot(111, projection = '3d')
    # Combine data points into a single array
    data = np.vstack([x, y, z])
    
    # Calculate the density of the points using a Gaussian KDE
    kde = gaussian_kde(data)
    density = kde(data)  # Compute the density for each point
    
    # Normalize density for colormap
    norm = plt.Normalize(vmin=density.min(), vmax=density.max())
    
    scatter = ax.scatter(x, y, z, c=density, cmap='viridis', norm=norm)
    
    # Add a colorbar
    fig.colorbar(scatter)
    
    # Set labels and title
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of Impact')
    
    plt.show()

#%%

mednmom, df = ReadInStuff(dirpath)

#%%
mp4name = '3dScatterMovie1.mp4'
downsample = 1000
burnin = 30000
D3scatterMovie(df, mp4name, downsample, burnin)


