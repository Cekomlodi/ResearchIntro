#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 19:51:39 2025

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

#dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1144228507"
dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1150509781" 
#dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1146428350"  #very bad one
#dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1147453083" 


df = pd.read_csv(dirpath + '/impactchain.dat', header = None, delimiter = '\s+',
	    names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])

mednmom = np.median(df['mom'])
mednmom = "{:.4e}".format(mednmom)

x = df['xloc']
y = df['yloc']
z = df['zloc']

#D3scatter(x, y, z, mp4name, downsample)

#%%

x = df['xloc'].tail(70000)
y = df['yloc'].tail(70000)
z = df['zloc'].tail(70000)

from scipy.stats import gaussian_kde

#%%
# Combine data points into a single array
data = np.vstack([x, y, z])

# Calculate the density of the points using a Gaussian KDE
kde = gaussian_kde(data)
density = kde(data)  # Compute the density for each point

# Normalize density for colormap
norm = plt.Normalize(vmin=density.min(), vmax=density.max())


#%%

from scipy.stats import gaussian_kde

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection = '3d')

mp4name = '3dscatter_Widehit_turnedFrame.mp4'
downsample = 100

writer = FFMpegWriter(fps=70, metadata=dict(artist='Me'))
with writer.saving(fig, mp4name, 100):       
    
    for i in range(0, len(x)):
            
            if i % downsample == 0:
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # Set labels and title
                ax.set_xlim([-1,1])
                ax.set_ylim([-1,1])
                ax.set_zlim([0,1])
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Scatter Plot of Impact')
                #ax.view_init(azim = -30)

                # # Combine data points into a single array
                # data = np.vstack([x.head(i), y.head(i), z.head(i)])
                
                # # Calculate the density of the points using a Gaussian KDE
                # kde = gaussian_kde(data)
                # density = kde(data)  # Compute the density for each point
                
                #ax.scatter(x.head(i), y.head(i), z.head(i), c=density[:i])
                scatter = ax.scatter(x.head(i), y.head(i), z.head(i), c=density[:i])
                
                x1 = [.4,.9, .9, .4, -.4, -.9, -.9, -.4, .4]
                y1 = [1,.25,-.25,-1,-1,-.25,.25, 1, 1]
                Topz = [1,1,1,1,1,1,1,1,1]
                Botz = [0,0,0,0,0,0,0,0,0]

                ax.plot(x1, y1, Topz, 'k')
                ax.plot(x1, y1, Botz, 'k')
                
                for s in range(len(x1)):
                    ax.plot([x1[s], x1[s]], [y1[s], y1[s]], [Botz[s], Topz[s]], 'k')
                
                #cb = fig.colorbar(scatter)
                
                #take frame
                writer.grab_frame()
                #clear to make it run faster maybe
                #cb.remove()
                ax.clear()
                
            else:
                continue
            
            
            
#%% adding a wire frame
fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111, projection = '3d')

# Add a colorbar
#fig.colorbar(scatter)

#Maybe Polygon

x1 = [.4,.9, .9, .4, -.4, -.9, -.9, -.4, .4]
y1 = [1,.25,-.25,-1,-1,-.25,.25, 1, 1]
Topz = [1,1,1,1,1,1,1,1,1]
Botz = [0,0,0,0,0,0,0,0,0]

ax.plot(x1, y1, Topz, 'k')
ax.plot(x1, y1, Botz, 'k')
for i in range(len(x1)):
    ax.plot([x1[i], x1[i]], [y1[i], y1[i]], [Botz[i], Topz[i]], 'k')

scatter = ax.scatter(x, y, z, c=density)#, norm=norm1)

# Set labels and title
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([0,1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot Impact')
ax.view_init(azim = 30)
plt.show() 

#%%

from scipy.stats import gaussian_kde

def ScatterLPF(mp4name, downsample, fps, df, burnin, ImpactName):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection = '3d')
    
    x = df['xloc'].tail(len(df['xloc']-burnin))
    y = df['yloc'].tail(len(df['yloc']-burnin))
    z = df['zloc'].tail(len(df['zloc']-burnin))
    
    #calculate Density
    # Combine data points into a single array
    data = np.vstack([x, y, z])
    
    # Calculate the density of the points using a Gaussian KDE
    kde = gaussian_kde(data)
    density = kde(data)  # Compute the density for each point
    
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'))
    with writer.saving(fig, mp4name, 100):       
        
        for i in range(0, len(x)):
                
                if i % downsample == 0:
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # Set labels and title
                    ax.set_xlim([-1,1])
                    ax.set_ylim([-1,1])
                    ax.set_zlim([0,1])
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('3D Scatter Plot of Impact {0}'.format(ImpactName))
                    #ax.view_init(azim = -30)
                    
                    ax.scatter(x.head(i), y.head(i), z.head(i), c=density[:i])
                    
                    x1 = [.5,.9, .9, .5, -.5, -.9, -.9, -.5, .5]
                    y1 = [1,.25,-.25,-1,-1,-.25,.25, 1, 1]
                    Topz = [1,1,1,1,1,1,1,1,1]
                    Botz = [0,0,0,0,0,0,0,0,0]
    
                    ax.plot(x1, y1, Topz, 'k')
                    ax.plot(x1, y1, Botz, 'k')
                    
                    for s in range(len(x1)):
                        ax.plot([x1[s], x1[s]], [y1[s], y1[s]], [Botz[s], Topz[s]], 'k')
                    
                    #cb = fig.colorbar(scatter)
                    
                    #take frame
                    writer.grab_frame()
                    #clear to make it run faster maybe
                    #cb.remove()
                    ax.clear()
                    
                else:
                    continue
    

#%%

def ReadInStuff(dirpath):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    
    df = pd.read_csv(dirpath + '/impactchain.dat', header = None, delimiter = '\s+',
	    names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])

    mednmom = np.median(df['mom'])
    mednmom = "{:.4e}".format(mednmom)
    
    return mednmom, df

dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1150509781" #1144228507

#%%

mednmom, df = ReadInStuff(dirpath)


#%%
df = pd.read_csv(dirpath + '/impactchain.dat', header = None, delimiter = '\s+',
	    names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])

mednmom = np.median(df['mom'])
mednmom = "{:.4e}".format(mednmom)

#%%
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

mp4name = '3dscatter_Widehit_1.mp4'
downsample = 100
fps = 70
burnin = 16000
ImpactName = 1150509781

ScatterLPF(mp4name, downsample, fps, df, burnin, ImpactName)

#%%

from scipy.stats import gaussian_kde
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

dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1144228507"
#dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1150509781" 
#dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1146428350"  #very bad one
#dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1147453083" 



df = pd.read_csv(dirpath + '/impactchain.dat', header = None, delimiter = '\s+',
	    names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])

mednmom = np.median(df['mom'])
mednmom = "{:.4e}".format(mednmom)

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111, projection = '3d')
# Add a colorbar
#fig.colorbar(scatter)
#Maybe Polygon

x1 = [.4,.9, .9, .4, -.4, -.9, -.9, -.4, .4]
y1 = [1,.25,-.25,-1,-1,-.25,.25, 1, 1]
Topz = [1,1,1,1,1,1,1,1,1]
Botz = [0,0,0,0,0,0,0,0,0]

ax.plot(x1, y1, Topz, 'k')
ax.plot(x1, y1, Botz, 'k')
for i in range(len(x1)):
    ax.plot([x1[i], x1[i]], [y1[i], y1[i]], [Botz[i], Topz[i]], 'k')
    
x = df['xloc'].tail(len(df['xloc']-burnin))
y = df['yloc'].tail(len(df['yloc']-burnin))
z = df['zloc'].tail(len(df['zloc']-burnin))

data = np.vstack([x, y, z])

# Calculate the density of the points using a Gaussian KDE
kde = gaussian_kde(data)
density = kde(data)  # Compute the density for each point



scatter = ax.scatter(x, y, z, c=density)#, norm=norm1)

# Set labels and title
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([0,1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot Impact')
rotation = np.linspace(0, 360, 72)



writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'))

with writer.saving(fig, 'rotation.mp4', 100):
    
    for i in rotation:
        ax.view_init(azim = rotation[i])
        writer.grab_frame()
    
    for j in rotation:
        ax.view_init(elev = rotation[j])
        writer.grab_frame()
        
    for k in rotation:
        ax.view_init(roll = rotation[k])
        writer.grab_frame()
        
plt.show() 




