# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:10:17 2022

@author: s163761
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import sys

data_directory = r'output_NN\Linear\K1_Fold_900'

# folder_gm = r'output_files_All'

#%% Find Noise level
# noise_idx = data_directory.find('Noise_')
# if noise_idx != -1:
#     noise_level = data_directory[noise_idx+6:]
# else:
#     noise_level = 'N/A'

#%% Functions
def str3_to_int(list_str):
        
    '''
    Takes list of index in (string) 000 format
    Reurns list of index in integers
    E.g.:
        0   --> '000'
        20  --> '020'
        100 --> '100'
    '''
    
    list_int = []
    for i in list_str:
    
        if i[0] != str(0):
            list_int.append( int(i[0:3]))
        elif i[1] != str(0):
            list_int.append( int(i[1:3]))
        else:
            list_int.append( int(i[2:3]))
            
    return list_int

#%% Parameters of the study, does not need to be changed

# errors = [ 'RMSE' , 'SMSE', 'MAE' , 'MAPE', 'TRAC']

#%% Plot settings

# plot_error = 'RMSE'   # which error should be plot
# index_error = errors.index(plot_error)

# plot = True

#%% Create the datasets

# df_GMs = pd.read_pickle(os.path.join(folder_gm , '00_Index_Results.pkl'))
df_all_errors = pd.DataFrame(index = [0])
# df_errors.loc[:, 'E - glob'] = df_GMs.loc[:, 'E - glob']

#%% Extract data
#fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16))

idx = 0
for rdirs, dirs, files in os.walk(data_directory):
        
        for file in files:
            
            if file.endswith("00_All_Errors.pkl"):
                print(rdirs)
                
                
                print(file)  
                
                noise_idx = rdirs.find('Noise_')
                if noise_idx != -1:
                    noise_level = int(rdirs[noise_idx+6:])
                else:
                    noise_level = 'N/A'
                    
                print(noise_level)
                
                # frames = [df_Results_1, df_Results_0, df_Results_2]
                # df_Results_Merged = pd.concat(frames)
                df_all_errors = pd.concat([df_all_errors, pd.DataFrame(columns = [noise_level], index = [0])],axis=1)
                df_all_errors[noise_level][0] = pd.read_pickle(os.path.join(rdirs, file))
                
                
                idx += 1
                # df_temp = 
                
                # df_all_errors = pd.concat([df_all_errors, df_temp], axis=1)
                

#%%

df_diff = pd.DataFrame(columns = df_all_errors.columns[1:], index = [0])

column_0 = df_all_errors.columns[0]
for column_i in df_all_errors.columns[1:]:
    
    df_0 = df_all_errors[column_0][0].iloc[:,[1,2,5]].values
    df_i = df_all_errors[column_i][0].iloc[:,[1,2,5]].values
    
    array_i = (df_i - df_0) / df_0
    array_i = abs(array_i)

    df_diff[column_i][0] = pd.DataFrame(array_i)
                            
                                        
#%% Plot
Errors = ['RMSE', 'SMSE', 'TRAC']
    
plot_df = df_diff.copy()
# plot_df.to_pickle(os.path.join(data_directory, '00_All_Errors.pkl')) 

# print(f'\n\nPlotting {len(plot_df)} GMs points')
  

# Plotting
for i in plot_df.columns:
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16), constrained_layout=True, sharex=True) 
    fig.suptitle(f'Dissipated global energy \n Noise level: {i} dB', fontsize=16)
    
    
    axes[0].set_ylim(0,0.5)
    axes[1].set_ylim(0,1.1)
    axes[2].set_ylim(0,0.045)
    
    axes[2].set_xlabel('Global Dissipated Energy [XX]',fontsize = '10')
    
    ax_idx = 0
    for ax in axes:
        error = Errors[ax_idx]
        
        # Plot
        # ax.hexbin(df_all_errors[i][0]['E - glob'], df_diff[i][0][ax_idx], gridsize=30)
        ax.scatter(df_all_errors[i][0]['E - glob'], df_diff[i][0][ax_idx])
        
        mean = df_diff[i][0][ax_idx].mean()
        ax.axhline(y=mean, ls='--', linewidth=2, color='black')
        ax.text(0, mean + 0.2*mean, round(mean,4), va='bottom', ha='left', fontsize=12, fontweight="bold")
        
        # Labels & Titels
        # ax.set_xlabel('Energy',fontsize = '10')
        ax.set_ylabel(f'{error}', fontweight="bold", fontsize = '10')
        # ax.set_title(f'{error}',  y=1.05, fontweight="bold", fontsize = '12')
        
        # Limits & Grids
        ax.set_xlim(0,30)
        ax.grid(True)
        
        ax_idx += 1
    
    # Distances
    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1,
    #                 right=0.9,
    #                 top=0.9,
    #                 wspace=0.4,
    #                 hspace=0.4)
    
    
    # Save plot
    plt.savefig(os.path.join(data_directory, f'Error_Scatter_Diff_{i}.png'))
    


                            
