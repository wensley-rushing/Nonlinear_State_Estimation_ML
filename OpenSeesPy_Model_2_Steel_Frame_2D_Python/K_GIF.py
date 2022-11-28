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

folder_structure = r'output_files_gif'

#%% Inputs

# Fix range for all Anims or determine indevidually
fixed_range_all = True
# Only relevant if fixed_range_all = True
K_min = -663810530
K_max =  5e7#133931139

# K_min = 0
# K_max =  10


# Data shound be absolute or 'true' values
# If true: K_min == 0
abs_data = True 


# Animation type
save_ani_as = 'mp4' # 'gif' or 'mp4'
interval=50 # Default:200 (NOT IN USE)
repeat_delay=200 # Default:0


# Start/End Animation at specific frames, and framestep
start_frame = 0
end_frame = 'End' # Full length: 'End'
frame_step = 100

fps = 7


# Figure size
figure_size = (10,10)

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
    GM_Energy = round(df_Index_Results['E - glob'][GMID], 2)
    
    print(f'EQ ID: {GMID}')
    # K_plot
    
    # Length of Kmatrix
    K_list_len0 = len(K_list)
    K_list_len1 = len(K_list[0])
    
    # Determine end of range 
    if type(end_frame) == str:
        end_frame = K_list_len0
    
    # Remove elements from K
    remove_ids = list(range(0,K_list_len0))
    for i in list(range(start_frame,end_frame, frame_step)):
        if i in remove_ids:
            remove_ids.remove(i)
    K_list_plot = np.delete(K_list, remove_ids , axis=0)
    
    # First matrix of K=0 (as reference)
    #K_list_plot[0,:,:] = np.zeros([K_list_len1,K_list_len1])
       
    # Lenth of plotting data
    K_list_plot_len = len(K_list_plot)
    
    
    ## Min/Max Range
    if fixed_range_all:
        Kmin = K_min
        Kmax = K_max
    else:
        Kmin = int(np.amin(K_list_plot[-1,:,:]))
        Kmax = int(np.amax(K_list_plot[-1,:,:]))
        
    #Kmax_string = "{:e}".format(Kmax)
    #Kmin_string = "{:e}".format(Kmin)
    
    # Data shound be absolute or 'true'
    if abs_data:
        K_list_plot = np.absolute(K_list_plot)
        Kmin = 0
    
    
    ## Norm 
    c_lim = [Kmin, Kmax]
    ani_normalizer = 'Normalize'
    
    if ani_normalizer == 'LogNorm':
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=Kmin, vmax=Kmax)
        
    elif ani_normalizer == 'SymLogNorm':
        from matplotlib.colors import SymLogNorm
        norm = SymLogNorm(linthresh=1, linscale=1,
                                                      vmin=Kmin, vmax=Kmax, base=10) 
    elif ani_normalizer == 'TwoSlopeNorm':
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=Kmin, vcenter=0., vmax=Kmax)
        
    elif ani_normalizer == 'Normalize':
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=Kmin, vmax=Kmax)
        
    

    
    ## Animation 
    if save_ani_as == 'gif':
    
        from matplotlib.animation import PillowWriter
        animation_type = '.gif'
        writer = PillowWriter(fps=fps)
    
    elif save_ani_as == 'mp4':
    
        from matplotlib.animation import FFMpegWriter
        animation_type = '.mp4'
        writer = FFMpegWriter(fps=fps)
        #https://holypython.com/how-to-save-matplotlib-animations-the-ultimate-guide/
        matplotlib.rcParams['animation.ffmpeg_path'] = r'ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
    
    
    #%% Initial investigations
    if False:
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
        plt.imshow(K_list[1,:,:], cmap= 'viridis', norm=norm, interpolation='none')
        plt.clim(c_lim[0], c_lim[1])
        plt.colorbar()
        
        plt.title('K[1]')
        
        
        plt.figure()
        plt.imshow(K_list[1000,:,:], cmap= 'viridis', norm=norm, interpolation='none')
        plt.colorbar()
        plt.clim(c_lim[0], c_lim[1])
        plt.title('K[-1]')
    
    #%% Method 1: Append no Steps
    if False:
        # Figure initilization
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
        
        
        # Data-extraction
        ims = []
        for i in list(range(0,K_list_plot_len)):
           
            im = plt.imshow(K_list_plot[i,:,:], cmap= 'viridis', norm=norm, interpolation='none')
            ims.append([im])
        
        # Figure general layout  
        #ax.set_xlabel("Global Degree of Freedom")  
        plt.clim(c_lim[0], c_lim[1])
        plt.colorbar() 
        plt.title(f'Global Stiffness Matrix \n {GM_name}') 
        
        ax.text(1, 1, f'Global Energy: {GM_Energy} kNm', transform=ax.transAxes, va = 'bottom', ha='right',
                            fontsize=8, color='k')
        
        
        # Animation setup
        ani1 = animation.ArtistAnimation(fig, ims, 
                                         blit=True, repeat_delay=repeat_delay)
        ani1.save(os.path.join(folder_save_M1, f'{GMID}_Method_1' + animation_type), writer=writer)
        
        # plt.show()
        plt.close()
    
    
    
    #%% Menthod 2: Structure with Steps
    if True:
        # Define function for plot
        def updateline2(num, data3, line3):
            line3.set_data(data3[num,:,:]-data3[0,:,:])
            time_text.set_text("Step: %.0f \nTime: %.2f s" % (int(num*frame_step), K_time[num*frame_step] ))
            return line3
        
        #------------------------------------------------------------------------------
        
        # Figure initilization
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
        
        # Data-extraction
        data3 = K_list_plot
        
        # Initial frame
        init_im = data3[1,:,:]-data3[0,:,:]
        m = ax.imshow(init_im, cmap= 'viridis', norm=norm, interpolation='none')
        
        # Figure general layout   
        #ax.set_xlabel("DOFS")
         
        plt.colorbar(m)
        plt.title(f'Global Stiffness Matrix \n {GM_name}')
        
        # Text setup
        time_text = ax.text(0, 1, "", transform=ax.transAxes, va = 'bottom', ha='left',
                            fontsize=8, color='k')
        ax.text(1, 1, f'Global Energy: {GM_Energy} kNm', transform=ax.transAxes, va = 'bottom', ha='right',
                            fontsize=8, color='k')
        
        # Animation setup
        ani2 = animation.FuncAnimation(fig, updateline2, 
                                       frames=len(data3), fargs=(data3, m),
                                       repeat_delay=repeat_delay)
        ani2.save(os.path.join(folder_save_M2, f'{GMID}_Method_2' + animation_type), writer=writer)
    
        plt.close()