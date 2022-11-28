# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:35:02 2022

@author: s163761
"""

#%% Functions
def len_3D_list(lst3D):
    dim0 = len(lst3D)
    dim1 = len(lst3D[0])
    dim2 = len(lst3D[0][0])
    return [dim0, dim1, dim2]


#%% Generel 
import os
import sys
import numpy as np
import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

folder_structure = r'output_linear_non'
#%% Load K-matrixes
df_KMatrix = pd.read_pickle( os.path.join(folder_structure, '00_KMatrix.pkl') )
df_Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )


# Create Sub folder
folder_save = os.path.join(folder_structure, r'K_gif')
folder_save_M1 = os.path.join(folder_save, r'M1')
folder_save_M2 = os.path.join(folder_save, r'M2')
if not os.path.exists(folder_save):
    os.mkdir(folder_save)
    
    os.mkdir(folder_save_M1)
    
    os.mkdir(folder_save_M2)
    

#%% Plotting to GIF

for GMID in range(df_KMatrix.shape[0]):
    K_list = df_KMatrix['K'][GMID]
    K_time = df_KMatrix['Time'][GMID]
    GM_name = df_Index_Results['Ground motion'][GMID]
    
    print(f'EQ ID: {GMID}')
    # K_plot
    
    K_list_len0 = len(K_list)
    K_list_len1 = len(K_list[0])
    frame_step = 100
    
    
    remove_ids = list(range(0,K_list_len0))
    for i in list(range(0,K_list_len0, frame_step)):
        remove_ids.remove(i)
    K_list_plot = np.delete(K_list, remove_ids , axis=0)
    K_list_plot[0,:,:] = np.zeros([K_list_len1,K_list_len1])
    
    K_list_plot_len = len(K_list_plot)
    
    
    # Clim
    #c_lim = [10**(0), 10**(10)]
    
    Kmin = int(np.amin(K_list_plot[-1,:,:]))
    Kmax = int(np.amax(K_list_plot[-1,:,:]))
    
    Kmax_string = "{:e}".format(Kmax)
    Kmin_string = "{:e}".format(Kmin)
    
    
    # Norm
    from matplotlib.colors import LogNorm
    from matplotlib.colors import SymLogNorm
    from matplotlib.colors import TwoSlopeNorm
    norm = SymLogNorm(linthresh=1, linscale=1,
                                                  vmin=Kmin, vmax=Kmax, base=10)# LogNorm()
    c_lim = [Kmin, Kmax]
    #c_lim = [-1e5, 1e10]
    
    # norm = TwoSlopeNorm(vmin=Kmin, vcenter=0., vmax=Kmax)
    
    # c_lim = [Kmin, Kmax]
    
    
    
    
    
    #Animation 
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=10)
    
    interval=50 # Default:200 (NOT IN USE)
    repeat_delay=1000 # Default:0
    
    
    #%% Initial investigations
    
    init_im = np.empty([18,18])*1.22*1e9
    init_im = np.ones([18,18])*1.25*1e9
    init_im = K_list[1,:,:]
    # init_im = np.random.rand(18,18)
    
    fig = plt.figure()
    im = plt.imshow(init_im, cmap= 'viridis') #, norm=LogNorm())
    plt.colorbar()
    plt.clim(c_lim[0], c_lim[1])
    plt.show()
    
    
    plt.figure()
    plt.imshow(K_list[0,:,:], cmap= 'viridis', norm=norm, interpolation='none')
    plt.clim(c_lim[0], c_lim[1])
    plt.colorbar()
    
    plt.title('K[0]')
    
    
    plt.figure()
    plt.imshow(K_list[1000,:,:], cmap= 'viridis', norm=norm, interpolation='none')
    plt.colorbar()
    plt.clim(c_lim[0], c_lim[1])
    plt.title('K[-1]')
    
    #%% Method 1: Append no Steps
    
    
    # Figure initilization
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    
    # Data-extraction
    ims = []
    for i in list(range(0,K_list_plot_len)):
       
        im = plt.imshow(K_list_plot[i,:,:], cmap= 'viridis', norm=norm, interpolation='none')
        ims.append([im])
    
    # Figure general layout  
    ax.set_xlabel("X")  
    plt.clim(c_lim[0], c_lim[1])
    plt.colorbar() 
    plt.title(f'Global Stiffness Matrix \n {GM_name}') 
    
    
    # Animation setup
    ani1 = animation.ArtistAnimation(fig, ims, 
                                     blit=True, repeat_delay=repeat_delay)
    ani1.save(os.path.join(folder_save_M1, f'{GMID}_Method_1.gif'), writer=writer)
    
    # plt.show()
    plt.close()
    
    
    
    #%% Menthod 2: Structure with Steps
    
    # Define function for plot
    def updateline2(num, data3, line3):
        line3.set_data(data3[num,:,:])
        time_text.set_text("Step: %.0f \nTime: %.2f s" % (int(num*frame_step), K_time[num*frame_step] ))
        return line3
    
    #------------------------------------------------------------------------------
    
    # Figure initilization
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    # Data-extraction
    data3 = K_list_plot
    
    # Initial frame
    init_im = data3[0,:,:]
    m = ax.imshow(init_im, cmap= 'viridis', norm=norm, interpolation='none')
    
    # Figure general layout   
    ax.set_xlabel("X")
     
    plt.colorbar(m)
    plt.title(f'Global Stiffness Matrix \n {GM_name}')
    
    # Text setup
    time_text = ax.text(0, 1, "", transform=ax.transAxes, va = 'baseline', ha='left',
                        fontsize=8, color='red')
    
    # Animation setup
    ani2 = animation.FuncAnimation(fig, updateline2, 
                                   frames=len(data3), fargs=(data3, m),
                                   repeat_delay=repeat_delay)
    ani2.save(os.path.join(folder_save_M2, f'{GMID}_Method_2.gif'), writer=writer)
