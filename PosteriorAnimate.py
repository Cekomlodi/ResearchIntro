# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 22:53:31 2025

@author: cekom
"""

def PostAnim(df, animate, Plot_title, video_length, downsample, mp4Name):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.animation import FFMpegWriter
    
    num_chains = len(df.columns)
    column_list = list(df.columns)
    
    #Setting up the graph environment
    width_ratios = np.concatenate(([4], np.ones(num_chains)))
    height_ratios = np.ones(num_chains)
    gridspec_height = num_chains
    gridspec_width = num_chains + 1
    total_entries = gridspec_height*gridspec_width
    arr = np.arange(1,total_entries+1, 1)
    general_layout = arr.reshape(gridspec_height, gridspec_width)
    
    #print(general_layout)
    
    #Change each entry to a string so we can make the corner
    general_string = general_layout.astype(str)
    
    #make the corner :(
    L = len(general_string[0])-2
    
    for i in range(0,len(general_string)):
        for k in range(0,L):
            general_string[i][k+i+2] = '.'
        L = L-1
        

    layout_pattern = general_string
    print(layout_pattern)
    
    
    gs_kw = dict(width_ratios= width_ratios, height_ratios= height_ratios)
    fig, axd = plt.subplot_mosaic(layout_pattern,
                                  gridspec_kw=gs_kw, figsize=(4*num_chains, 2*num_chains), constrained_layout=True)
    
    #plt.show()
    #static scary corner plots with chains
        
        #All of the chain x and y parameters to avoid growing plots in the future
    for i in range(0, num_chains):
        axd[str(layout_pattern[i][0])].set_xlim(0, len(df[column_list[i]]))
        axd[str(layout_pattern[i][0])].set_ylim(min(df[column_list[i]])-.5, max(df[column_list[i]])+.5)
        axd[str(layout_pattern[i][0])].plot(np.arange(0,len(df[column_list[i]]), 1), df[column_list[i]], linewidth = .5, color = 'darkorchid')
        print('Chain {} done'.format(i+1))
        
    axd[layout_pattern[num_chains-1][0]].set_xlabel('Chain Iteration')
    fig.suptitle(Plot_title, horizontalalignment = 'right', verticalalignment = 'top')
        
    #The corner plot stuff doesn't need parameters until animated, but it does need plotted
    #Plot all of the diagonal Histograms
    Corner_index = 0
    for i in range(0, num_chains):
        hist, bins = np.histogram(df[column_list[i]], density = True)
        axd[layout_pattern[i][Corner_index+1]].stairs(hist, bins, color = 'darkorchid', fill = True)
        Corner_index = Corner_index+1
        print('Histogram {} done'.format(i+1))
    
    for i in range(1, num_chains):
        for k in range(1, num_chains):
            if k <= i:
                axd[layout_pattern[i][k]].hist2d(df[column_list[k-1]], df[column_list[i]], cmap = 'BuPu', density = True)
            print('2dHist row {} Done'.format(k))
                  
    plt.show()
    
    ##########################################################################
    #Start animating all of the things...
   
    if animate == True:
        #set up the environment
        num_chains = len(df.columns)
        column_list = list(df.columns)
        
        #Setting up the graph environment
        width_ratios = np.concatenate(([4], np.ones(num_chains)))
        height_ratios = np.ones(num_chains)
        gridspec_height = num_chains
        gridspec_width = num_chains + 1
        total_entries = gridspec_height*gridspec_width
        arr = np.arange(1,total_entries+1, 1)
        general_layout = arr.reshape(gridspec_height, gridspec_width)
        
        #print(general_layout)
        
        #Change each entry to a string so we can make the corner
        general_string = general_layout.astype(str)
        
        #make the corner :(
        L = len(general_string[0])-2
        
        for i in range(0,len(general_string)):
            for k in range(0,L):
                general_string[i][k+i+2] = '.'
            L = L-1
            

        layout_pattern = general_string
        print(layout_pattern)
        
        
        gs_kw = dict(width_ratios= width_ratios, height_ratios= height_ratios)
        fig, axd = plt.subplot_mosaic(layout_pattern,
                                      gridspec_kw=gs_kw, figsize=(4*num_chains, 2*num_chains), constrained_layout=True)
        
        
        fps = len(df[column_list[0]])/(downsample*video_length)
        
        
        
        #start making frames
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'))
        with writer.saving(fig, mp4Name, 100):
        #Try without Setting Up blank graph lines and plots
        
            for frame in range(0,len(df[column_list[0]])):
                
                if i%downsample == 0:
                    
                    for i in range(0, num_chains):
                        axd[str(layout_pattern[i][0])].set_xlim(0, len(df[column_list[i]]))
                        axd[str(layout_pattern[i][0])].set_ylim(min(df[column_list[i]])-.5, max(df[column_list[i]])+.5)
                        
                        #first plots
                        axd[str(layout_pattern[i][0])].plot(np.arange(0,len(df[column_list[i]].head(frame)), 1), df[column_list[i]].head(frame), linewidth = .5, color = 'darkorchid')
                        #print('Chain {} done'.format(i+1))
                        
                    axd[layout_pattern[num_chains-1][0]].set_xlabel('Chain Iteration')
                    fig.suptitle(Plot_title, horizontalalignment = 'right', verticalalignment = 'top')
                        
                    #The corner plot stuff doesn't need parameters until animated, but it does need plotted
                    #Plot all of the diagonal Histograms
                    Corner_index = 0
                    for i in range(0, num_chains):
                        hist, bins = np.histogram(df[column_list[i]].head(frame), density = True)
                        axd[layout_pattern[i][Corner_index+1]].stairs(hist, bins, color = 'darkorchid', fill = True)
                        Corner_index = Corner_index+1
                        #print('Histogram {} done'.format(i+1))
                    
                    for i in range(1, num_chains):
                        for k in range(1, num_chains):
                            if k <= i:
                                axd[layout_pattern[i][k]].hist2d(df[column_list[k-1]].head(frame), df[column_list[i]].head(frame), cmap = 'BuPu', density = True)
                            #print('2dHist row {} Done'.format(k))
                            
                            writer.grab_frame()
                            print('frame {} done'.format(frame))
                            axd.clear()
        
        
        
        
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Install LISAcattools

x = np.array(np.random.normal(0, 1, 1000))
y = np.array(np.random.normal(2, 4, 1000))
z = np.array(np.random.normal(-2, 3, 1000))
h = np.array(np.random.normal(6, 1, 1000))
i = np.array(np.random.normal(-4, 4, 1000))
j = np.array(np.random.normal(0, 3, 1000))
k = np.array(np.random.normal(-2, 3, 1000))
l = np.array(np.random.normal(6, 1, 1000))
m = np.array(np.random.normal(-4, 4, 1000))
n = np.array(np.random.normal(0, 3, 1000))

d = {'x':x, 'y':y, 'z':z}
df = pd.DataFrame(data = d)

d6 = {'x':x, 'y':y, 'z':z,'h':h, 'i':i, 'j':j}
df6 = pd.DataFrame(d6)

d10 = {'x':x, 'y':y, 'z':z,'h':h, 'i':i, 'j':j, 'k':k,'l':l, 'm':m, 'n':n}
df10 = pd.DataFrame(d10)


PostAnim(df, True, 'Plot_title', 10, 10, 'app_test.mp4')
#PostAnim(df10, True, 'Test')    