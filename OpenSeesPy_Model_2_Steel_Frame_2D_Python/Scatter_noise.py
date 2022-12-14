# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:21:34 2022

@author: s202277
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import sys

data_directory = r'output_NN\Linear\K1_Fold_900\Noise_1000'

folder_gm = r'output_files_All'

#%% Find Noise level
noise_idx = data_directory.find('Noise_')
if noise_idx != -1:
    noise_level = data_directory[noise_idx+6:]
else:
    noise_level = 'N/A'

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

errors = [ 'RMSE' , 'SMSE', 'MAE' , 'MAPE', 'TRAC']

#%% Plot settings

# plot_error = 'RMSE'   # which error should be plot
# index_error = errors.index(plot_error)

plot = True

#%% Create the datasets

df_GMs = pd.read_pickle(os.path.join(folder_gm , '00_Index_Results.pkl'))

df_errors = pd.DataFrame(index = df_GMs.index.tolist() , columns = ['E - glob', 'RMSE' , 'SMSE', 'MAE' , 'MAPE', 'TRAC'])
df_errors.loc[:, 'E - glob'] = df_GMs.loc[:, 'E - glob']

#%% Extract data
K_folds = 10

fold_idx = 0 

#fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16))

for rdirs, dirs, files in os.walk(data_directory):
        
        
        
        
        for file in files:
            
            if file.endswith("Error.pkl") and fold_idx < K_folds:
                
                # print(fold_idx)
                
                
                K_errors = pd.read_pickle(os.path.join(rdirs, file))
                
                for gm in K_errors.columns:  # update DataFrame with the errors of all the earthquakes
                    
                    gm_idx =  str3_to_int([gm])[0]
                    
                    df_errors.iloc[gm_idx, 1] = K_errors.loc['RMSE', gm]
                    df_errors.iloc[gm_idx, 2] = K_errors.loc['SMSE', gm]
                    df_errors.iloc[gm_idx, 3] = K_errors.loc['MAE', gm]
                    df_errors.iloc[gm_idx, 4] = K_errors.loc['MAPE', gm]
                    df_errors.iloc[gm_idx, 5] = K_errors.loc['TRAC', gm]

                fold_idx += 1
                         
                                        
#%% Plot

if plot:
    
    plot_df = df_errors.dropna()
    plot_df.to_pickle(os.path.join(data_directory, '00_All_Errors.pkl')) 
    
    print(f'\n\nPlotting {len(plot_df)} GMs points')
    
      
    Errors = ['RMSE', 'SMSE', 'TRAC']
    # Plotting
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16), constrained_layout=True, sharex=True) 
    fig.suptitle(f'Dissipated global energy \n Noise level: {noise_level} dB', fontsize=16)
    
    axes[0].set_ylim(0,1.80)
    axes[1].set_ylim(0,0.65)
    axes[2].set_ylim(0.5,1.0)
    
    axes[2].set_xlabel('Global Dissipated Energy [XX]',fontsize = '10')
    
    ax_idx = 0
    for ax in axes:
        error = Errors[ax_idx]
        
        # Plot
        ax.scatter(plot_df.loc[:, 'E - glob'], plot_df.loc[:, error])
        
        # Labels & Titels
        # ax.set_xlabel('Energy',fontsize = '10')
        ax.set_ylabel(f'{error}', fontweight="bold", fontsize = '10')
        # ax.set_title(f'{error}',  y=1.05, fontweight="bold", fontsize = '12')
        
        # Limits & Grids
        #ax.set_xlim(0,3)
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
    plt.savefig(os.path.join(data_directory, f'Error_Scatter_{noise_level}.png'))
    


                            
