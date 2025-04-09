#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:59:58 2025

@author: corinnekomlodi
"""

#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages

C1_X = -9.260000e-01 # x coordinate of spacecraft bottom deck corner 1 [m]
C1_Y = -2.168000e-01 # y coordinate of spacecraft bottom deck corner 1 [m]
C2_X = -9.260000e-01 # x coordinate of spacecraft bottom deck corner 2 [m]
C2_Y = 2.048000e-01  # y coordinate of spacecraft bottom deck corner 2 [m]
C3_X = -5.263000e-01 # x coordinate of spacecraft bottom deck corner 3 [m]
C3_Y = 8.970000e-01  # y coordinate of spacecraft bottom deck corner 3 [m]
C4_X = 5.163000e-01  # x coordinate of spacecraft bottom deck corner 4 [m]
C4_Y = 8.970000e-01  # y coordinate of spacecraft bottom deck corner 4 [m]
C5_X = 9.160000e-01  # x coordinate of spacecraft bottom deck corner 5 [m]
C5_Y = 2.048000e-01  # y coordinate of spacecraft bottom deck corner 5 [m]
C6_X = 9.160000e-01  # x coordinate of spacecraft bottom deck corner 6 [m]
C6_Y = -2.168000e-01 # y coordinate of spacecraft bottom deck corner 6 [m]
C7_X = 5.163000e-01  # x coordinate of spacecraft bottom deck corner 7 [m]
C7_Y = -9.090000e-01 # y coordinate of spacecraft bottom deck corner 7 [m]
C8_X = -5.263000e-01 # x coordinate of spacecraft bottom deck corner 8 [m]
C8_Y = -9.090000e-01 # y coordinate of spacecraft bottom deck corner 8 [m]
H = 8.315000e-01


def area(phi_off):
    phi_off=phi_off
    xproj=lambda y,x:math.sqrt(x*x+y*y)*math.sin(math.atan2(y,x)+phi_off)
    x=np.zeros(8)
    x[0]=xproj(C1_Y,C1_X);
    x[1]=xproj(C2_Y,C2_X);
    x[2]=xproj(C3_Y,C3_X);
    x[3]=xproj(C4_Y,C4_X);
    x[4]=xproj(C5_Y,C5_X);
    x[5]=xproj(C6_Y,C6_X);
    x[6]=xproj(C7_Y,C7_X);
    x[7]=xproj(C8_Y,C8_X);
    w=max(x)-min(x)
    return w*H

def area2(phi_off):
    sum=0
    for i in range(8):
        cth=math.cos(phi_norm[i]-phi_off)
        if(cth>0):
            sum+=cth*fa[i]
    return sum

def area_3d(theta): 
    return mean_area*math.sin(theta)+top_area*abs(math.cos(theta))

def triangle_area(x0,y0,x1,y1,x2,y2):
    return abs((y2-y0)*(x1-x0)-(x2-x0)*(y1-y0))/2.0

def side_area(x0,y0,x1,y1):
    return math.sqrt((y1-y0)**2+(x1-x0)**2)*H

def side_normal(x0,y0,x1,y1):
    return -np.array([y1-y0,-(x1-x0)])/math.sqrt((y1-y0)**2+(x1-x0)**2)

top_area=0
top_area+=triangle_area(C1_X,C1_Y,C2_X,C2_Y,C3_X,C3_Y)
top_area+=triangle_area(C1_X,C1_Y,C4_X,C4_Y,C3_X,C3_Y)
top_area+=triangle_area(C1_X,C1_Y,C4_X,C4_Y,C5_X,C5_Y)
top_area+=triangle_area(C1_X,C1_Y,C6_X,C6_Y,C5_X,C5_Y)
top_area+=triangle_area(C1_X,C1_Y,C6_X,C6_Y,C7_X,C7_Y)
top_area+=triangle_area(C1_X,C1_Y,C8_X,C8_Y,C7_X,C7_Y)
fa=np.zeros(10)
fa[0]=side_area(C1_X,C1_Y,C2_X,C2_Y)
fa[1]=side_area(C3_X,C3_Y,C2_X,C2_Y)
fa[2]=side_area(C3_X,C3_Y,C4_X,C4_Y)
fa[3]=side_area(C5_X,C5_Y,C4_X,C4_Y)
fa[4]=side_area(C5_X,C5_Y,C6_X,C6_Y)
fa[5]=side_area(C7_X,C7_Y,C6_X,C6_Y)
fa[6]=side_area(C7_X,C7_Y,C8_X,C8_Y)
fa[7]=side_area(C1_X,C1_Y,C8_X,C8_Y)
fa[8]=fa[9]=top_area;
for iface in range(10):
    print("Area of face ",iface," is ",fa[iface])
fn=np.zeros((10,2))
fn[0]=side_normal(C1_X,C1_Y,C2_X,C2_Y)
fn[1]=-side_normal(C3_X,C3_Y,C2_X,C2_Y)
fn[2]=side_normal(C3_X,C3_Y,C4_X,C4_Y)
fn[3]=-side_normal(C5_X,C5_Y,C4_X,C4_Y)
fn[4]=side_normal(C5_X,C5_Y,C6_X,C6_Y)
fn[5]=-side_normal(C7_X,C7_Y,C6_X,C6_Y)
fn[6]=side_normal(C7_X,C7_Y,C8_X,C8_Y)
fn[7]=-side_normal(C1_X,C1_Y,C8_X,C8_Y)

phi_norm=np.zeros(8)
for iface in range(8):
    phi_norm[iface]=math.atan2(fn[iface][1],fn[iface][0])
    print("Normal to face ",iface," is ",fn[iface],",  phi_norm=",phi_norm[iface])

phi=math.atan2(C2_Y-C1_Y,C2_X-C1_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C3_Y-C2_Y,C3_X-C2_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C4_Y-C3_Y,C4_X-C3_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C5_Y-C4_Y,C5_X-C4_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C6_Y-C5_Y,C6_X-C5_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C7_Y-C6_Y,C7_X-C6_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C8_Y-C7_Y,C8_X-C7_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
phi=math.atan2(C1_Y-C8_Y,C1_X-C8_X);print("x-sect at phi=",phi,": ",area(phi)," or ",area2(phi))
    
#data=np.loadtxt("impactchain.dat")
data=np.loadtxt("/Users/corinnekomlodi/Desktop/MessingWithLPF/1144228507/impactchain.dat")
#data=np.loadtxt("mini_impactchain.dat")


#No burn in
#nburnfac=0.3
#data=data[nburnfac*data.shape[0]:]

skydata=data[:,6:8]
print(np.shape(data))
print(data)
print(np.shape(skydata))
print(skydata)

pp = PdfPages('histograms.pdf')
print("histo err ~ ",50/math.sqrt(np.shape(data)[0]/50),"%")
n, bins, patches = plt.hist(skydata[:,1], 150, density = True, facecolor='green', alpha=0.75)
plt.xlabel('phi')
plt.ylabel('Probability')
plt.grid(True)
pp.savefig()
plt.clf()

phineareq=[t[1] for t in skydata if abs(t[0])<0.1]
print("histo err ~ ",50/math.sqrt(np.shape(phineareq)[0]/50),"%")
n, bins, patches = plt.hist(phineareq, 50, density = True, facecolor='green', alpha=0.75)
y=np.array([ area(b) for b in bins ])
mean_area=np.mean(y)
print("mean_area=",mean_area)
scale=np.mean(n)/mean_area
y=y*scale
l = plt.plot(bins, y, 'r--', linewidth=1)
y=np.array([ area2(b) for b in bins ])
mean_area=np.mean(y)
print("mean_area=",mean_area)
scale=np.mean(n)/mean_area
y=y*scale
l = plt.plot(bins, y, 'b--', linewidth=1)
plt.xlabel('phi-near-equator')
plt.ylabel('Probability')
plt.grid(True)
#plt.show()
pp.savefig()
plt.clf()

n, bins, patches = plt.hist(skydata[:,0], 50, density = True, facecolor='green', alpha=0.75)
y=np.array([ area_3d(math.acos(b)) for b in bins ])
mean_area3d=np.mean(y)
scale=np.mean(n)/mean_area3d
y=y*scale
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.xlabel('cos(theta)')
plt.ylabel('Probability')
plt.grid(True)
#plt.show()
pp.savefig()
plt.clf()

bins=np.arange(11)-0.5
n, bins, patches = plt.hist(data[:,8], bins, density = True, facecolor='green', alpha=0.75)
l = plt.plot(np.arange(10)+0.0, fa/sum(fa), '--ro', linewidth=1)
plt.xlabel('face')
plt.ylabel('Probability')
plt.grid(True)
#plt.show()
pp.savefig()
plt.clf()

facecth=[t[6] for t in data if abs(t[8])==8]
nbins=50
n, bins, patches = plt.hist(facecth, nbins, density = True, facecolor='green', alpha=0.75)
l = plt.plot(bins, -2*bins, '--r', linewidth=1)
plt.xlabel('cth on face 8')
plt.ylabel('Probability')
plt.grid(True)
#plt.show()
pp.savefig()
plt.clf()

facecth=[t[6] for t in data if abs(t[8])==9]
nbins=50
n, bins, patches = plt.hist(facecth, nbins, density = True, facecolor='green', alpha=0.75)
l = plt.plot(bins, 2*bins, '--r', linewidth=1)
plt.xlabel('cth on face 9')
plt.ylabel('Probability')
plt.grid(True)
#plt.show()
pp.savefig()
plt.clf()

#print "checking incidence angles:"
#for t in data:
#    iface=t[8]
#    if(iface<8):
#        if(iface<0 or math.cos(t[7]-phi_norm[iface])<0):
#            print "err:"+str(t.tolist())

for iface in range(8):
    facephi=[t[7] for t in data if t[8]==iface]
    nbins=50
    n, bins, patches = plt.hist(facephi, nbins, density = True, facecolor='green', alpha=0.75)
    l = plt.plot(bins, np.cos(bins-phi_norm[iface])/2, '--r', linewidth=1)
    plt.xlabel('phi on face '+str(iface))
    plt.ylabel('Probability')
    plt.grid(True)
    #plt.show()
    pp.savefig()
    plt.clf()

for iface in range(8):
    facecth=[t[6] for t in data if abs(t[8])==iface]
    nbins=50
    n, bins, patches = plt.hist(facecth, nbins, density = True, facecolor='green', alpha=0.75)
    l = plt.plot(bins, np.sqrt(1-bins*bins)/(math.pi/2), '--r', linewidth=1)
    plt.xlabel('cth on face '+str(iface))
    plt.ylabel('Probability')
    plt.grid(True)
    #plt.show()
    pp.savefig()
    plt.clf()

#We downsample the data for scatter plots    
downfac=50 #downsample factor
ncol=12
print("data shape: ",data.shape)
icross=int(data.shape[0]/downfac)
datad=data[0:downfac*icross].reshape(downfac,icross,ncol)[0]

for iface in range(10):
    x=np.array([t[4] for t in data if abs(t[8])==iface])
    y=np.array([t[5] for t in data if abs(t[8])==iface])
    xd=np.array([t[4] for t in datad if abs(t[8])==iface])
    yd=np.array([t[5] for t in datad if abs(t[8])==iface])
    plotarea=fa[iface]
    plotareatop=3.326652
    if(iface>7): #for consistent plot densities we want the area of the full rectangle rather than the octagon face area
        plotarea=plotareatop
    nbins=round(math.sqrt(plotarea/plotareatop)*50) #########edited
    #n, bins, patches = plt.hist2d(x,y, bins=nbins)
    n, xbins, ybins = np.histogram2d(x,y, bins=nbins)
    im = plt.imshow(n, interpolation='nearest', origin='lower',extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    plt.title('impact point density on face '+str(iface))
    plt.xlabel('x')
    plt.ylabel('y')
    pp.savefig()
    plt.clf()
    #print xd,yd
    print(np.shape(xd))
    plt.scatter(xd,yd)
    plt.xlim(xbins[0], xbins[-1])
    plt.ylim(ybins[0], ybins[-1])
    plt.title('impact point scatter on face '+str(iface))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    #plt.show()
    pp.savefig()
    plt.clf()

pp.close()


########## dont touch anything above this line ##################

#%%

import lpf_graph_functions as LPF
import lpf_graph_functions_altered as LPFA
import pandas as pd
import colormap as colormap


currun = 2
dirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1150509781" #1144228507
rundirpath = "/Users/corinnekomlodi/Desktop/MessingWithLPF/1150509781"  #1144228507


df = pd.read_csv(dirpath + '/impactchain.dat', header = None,delimiter = '\s+',
	    names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])

mednmom = np.median(df['mom'])
mednmom = "{:.4e}".format(mednmom)

N=50


face0,face1,face2,face3,face4,face5,face6,face7,face8,face9 = LPFA.flatten_LPF(df,N, 'lin', mednmom, currun, colormap.parula, dirpath, rundirpath)

#%%

from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.path as mpath

fig = plt.figure(figsize = (10,10))              #size of figure
ax = fig.add_subplot(1,1,1, aspect = 'equal')    #add subplot with equal axes
ax.set_xlim(-2,2)                                #xlim
ax.set_ylim(-2,4)

patches = [face0[5],face1[5],face2[5],face3[5],face4[5],face5[5],face6[5],face7[5],face8[5],face9[5]]
collection = PatchCollection(patches, match_original=True)

combined_path = mpath.Path.make_compound_path(*(p.get_path() for p in patches))

# Create a new PathPatch from the combined path
combined_patch = mpatches.PathPatch(combined_path, facecolor='red', alpha=0.5)
ax.add_patch(combined_patch)

ax.pcolormesh(face0[0], face0[1], face0[2], norm = face0[3], cmap = face0[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face1[0], face1[1], face1[2], norm = face1[3], cmap = face1[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face2[0], face2[1], face2[2], norm = face2[3], cmap = face2[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face3[0], face3[1], face3[2], norm = face3[3], cmap = face3[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face4[0], face4[1], face4[2], norm = face4[3], cmap = face4[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face5[0], face5[1], face5[2], norm = face5[3], cmap = face5[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face6[0], face6[1], face6[2], norm = face6[3], cmap = face6[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face7[0], face7[1], face7[2], norm = face7[3], cmap = face7[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face8[0], face8[1], face8[2], norm = face8[3], cmap = face8[4], 
              clip_path = combined_patch, clip_on = True)
ax.pcolormesh(face9[0], face9[1], face9[2], norm = face9[3], cmap = face9[4], 
              clip_path = combined_patch, clip_on = True)


ax.annotate('0', xy=(-1.85, 0))#, xytext=(xsc[0] - H -.5,0))
ax.annotate('1', xy=(-1.5, 1))#, xytext=(xsc[0] - H -.5,0))
ax.annotate('3', xy=(1.5, 1))#, xytext=(xsc[0] - H -.5,0))
ax.annotate('4', xy=(1.85, 0))#, xytext=(xsc[0] - H -.5,0)) 
ax.annotate('5', xy=(1.5, -1))#, xytext=(xsc[0] - H -.5,0))
ax.annotate('6', xy=(0, -1.9))#, xytext=(xsc[0] - H -.5,0))
ax.annotate('7', xy=(-1.6,-1))#, xytext=(xsc[0] - H -.5,0))
ax.annotate('8', xy=(0, 3.6))#, xytext=(xsc[0] - H -.5,0))
#face8 = [xedges,yedges,Hist,norm,my_cmap, patchbot]#######################
#ax.pcolormesh(xedges,yedges,Hist, norm = norm, cmap = my_cmap, #interpolation='nearest', origin='lower',
#                    clip_path = patchbot, clip_on=True)


#%%

print('Likelihood Chain')
fig = plt.figure(figsize = (7,6))
ax = fig.add_subplot(1,1,1)

light_purple = colormap.parula(0) #'#5F2CFF'

index = list(df['logp'].index.values)                       #Lets you skip NaN stuff
ax.plot(index, df['logp'], color = light_purple)
ax.set_title('Log Likelihood of Impact Chain, \n Median Momentum = %s kg m/s'%(mednmom), y=1.03)#, size='small')
#ax.title(fig_title, y=1.08)
ax.set_xlabel('Step in Chain')
ax.set_ylabel('Log Likelihood')
ax.set_xlim(min(index))

plt.tight_layout()
#plt.tight_layout()
#plt.savefig('%s/logprob_chain_%s_%s.png'%(dirpath,currun,mednmom))
#plt.savefig('%s/logprob_chain_%s.png'%(rundirpath,currun))
plt.show()

#%%


#df = pd.read_csv(dirpath + '/impactchain.dat', header = None,delimiter = '\s+',
#names = ['logp','impactnum','time','mom','whatever','who??','coslat', 'longi', 'face', 'xloc','yloc','zloc'])


print('CosLat Chain')
fig = plt.figure(figsize = (7,6))
ax = fig.add_subplot(1,1,1)

light_purple = colormap.parula(0) #'#5F2CFF'

index = list(df['coslat'].index.values)                       #Lets you skip NaN stuff
ax.plot(index, df['coslat'], color = light_purple)
ax.set_title('Log Likelihood of Cosine Latitude', y=1.03)#, size='small')
#ax.title(fig_title, y=1.08)
ax.set_xlabel('Step in Chain')
ax.set_ylabel('Cosine Latitude')
ax.set_xlim(min(index))

plt.tight_layout()
#plt.tight_layout()
#plt.savefig('%s/logprob_chain_%s_%s.png'%(dirpath,currun,mednmom))
#plt.savefig('%s/logprob_chain_%s.png'%(rundirpath,currun))
plt.show()

#%%

gs_kw = dict(width_ratios=[1, 1], height_ratios=[.75, .75])
#layout_pattern = [['upper left', 'right']]
layout_pattern = [['upper left','right'], ['lower left', 'right']]
fig, axd = plt.subplot_mosaic(layout_pattern,
                              gridspec_kw=gs_kw, figsize=(15, 6), constrained_layout=True)
axd['upper left'].set_xlim([0,1000000])
axd['lower left'].set_xlim([0,1000000])
axd['upper left'].set_ylim([min(df['coslat']), max(df['coslat'])])
axd['lower left'].set_ylim([min(df['longi']-1), max(df['longi'])+1])

axd['upper left'].plot(index, df['coslat'], color = light_purple)
axd['lower left'].plot(index, df['longi'], color = light_purple)
axd['right'].hist2d(df['longi'], df['coslat'], cmap = 'inferno', bins = 50)

#%% Animating the sky map bitches

from matplotlib.animation import FFMpegWriter

def AnimateSkyLoc(xloc, yloc, maploc, lat, long, downsample, mp4Name):
    
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[.75, .75])
    layout_pattern = [['upper left','right'], ['lower left', 'right']]
    fig, axd = plt.subplot_mosaic(layout_pattern,
                              gridspec_kw=gs_kw, figsize=(15, 6), constrained_layout=True)

    writer = FFMpegWriter(fps=100, metadata=dict(artist='Me'))
    xlist = np.arange(1, len(lat)+1, step = 1)
    l, = axd[xloc].plot([],[], color = 'steelblue')
    l2, = axd[yloc].plot([],[], color = 'steelblue')
    
    axd[xloc].set_xlim([0, len(long)])
    axd[xloc].set_ylim([min(long), max(long)])
    axd[yloc].set_xlim([0, len(lat)])
    axd[yloc].set_ylim([min(lat), max(lat)])
    
    axd[maploc].set_xlim([min(long), max(long)])
    axd[maploc].set_ylim([min(lat), max(lat)])
    
    with writer.saving(fig, mp4Name, 100):
        for i in range(0, len(lat)):
            
            if i % downsample == 0:
                #Chains
                l.set_data(xlist[:i],long.head(i))
                l2.set_data(xlist[:i],lat.head(i))
                #Histograms
                axd[maploc].hist2d(long.head(i), lat.head(i), bins = 50, cmap = 'Blues')
                #Grab the frame
                writer.grab_frame()
                
                axd[maploc].clear()
            else:
                continue


#%%

#Set up variables
mp4Name = "SkyMapAttempt3.mp4"
xloc = 'upper left'
yloc = 'lower left'
maploc = 'right'
lat = df['coslat']
long = df['longi']
downsample = 1000

AnimateSkyLoc(xloc, yloc, maploc, lat, long, downsample, mp4Name)

#%%

import matplotlib.pyplot as plt


def annotate_axes(ax, text, fontsize=18):
    ax.text(x=0.5, y=0.5, z=0.5, s=text,
            va="center", ha="center", fontsize=fontsize, color="black")

# (plane, (elev, azim, roll))
views = [('XY',   (90, -90, 0)),
         ('XZ',    (0, -90, 0)),
         ('YZ',    (0,   0, 0)),
         ('-XY', (-90,  90, 0)),
         ('-XZ',   (0,  90, 0)),
         ('-YZ',   (0, 180, 0))]

layout = [['XY',  '.',   'L',   '.'],
          ['XZ', 'YZ', '-XZ', '-YZ'],
          ['.',   '.', '-XY',   '.']]
fig, axd = plt.subplot_mosaic(layout, subplot_kw={'projection': '3d'},
                              figsize=(12, 8.5))
for plane, angles in views:
    axd[plane].set_xlabel('x')
    axd[plane].set_ylabel('y')
    axd[plane].set_zlabel('z')
    axd[plane].set_proj_type('ortho')
    axd[plane].view_init(elev=angles[0], azim=angles[1], roll=angles[2])
    axd[plane].set_box_aspect(None, zoom=1.25)

    label = f'{plane}\n{angles}'
    annotate_axes(axd[plane], label, fontsize=14)

for plane in ('XY', '-XY'):
    axd[plane].set_zticklabels([])
    axd[plane].set_zlabel('')
for plane in ('XZ', '-XZ'):
    axd[plane].set_yticklabels([])
    axd[plane].set_ylabel('')
for plane in ('YZ', '-YZ'):
    axd[plane].set_xticklabels([])
    axd[plane].set_xlabel('')

label = 'mplot3d primary view planes\n' + 'ax.view_init(elev, azim, roll)'
annotate_axes(axd['L'], label, fontsize=18)
axd['L'].set_axis_off()

plt.show()


#%%
from scipy.stats import gaussian_kde

x = df['xloc']
y = df['yloc']
z = df['zloc']

kde = gaussian_kde(np.vstack([x, y, z]))
density = kde(np.vstack([x, y, z]))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x, y, z, c = density, cmap='Blues')


