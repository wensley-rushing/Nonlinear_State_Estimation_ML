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

data_directory = r'output_NN\Linear\K1_Fold_300_Noise_1000'

folder_gm = r'output_files_All'


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

plot_error = 'RMSE'   # which error should be plot
index_error = errors.index(plot_error)

plot = True

#%% Create the datasets

df_GMs = pd.read_pickle(os.path.join(folder_gm , '00_Index_Results.pkl'))

df_errors = pd.DataFrame(index = df_GMs.index.tolist() , columns = ['E - glob', 'RMSE' , 'SMSE', 'MAE' , 'MAPE', 'TRAC'])



df_errors.loc[:, 'E - glob'] = df_GMs.loc[:, 'E - glob']

#%%


K_folds = 10

fold_idx = 0 

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16))

for rdirs, dirs, files in os.walk(data_directory):
        
        
        
        
        for file in files:
            
            if file.endswith("Error.pkl") and fold_idx < K_folds:
                
                print(fold_idx)
                
                
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
    print(f'\n\nPlotting {len(plot_df)} GMs points')
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16))
    axes[0].scatter(plot_df.loc[:, 'E - glob'], plot_df.loc[:, 'RMSE'])
    axes[0].set_xlabel('Energy',fontsize = '10')
    axes[0].set_xlim(0,3)
    axes[0].set_ylabel('RMSE', fontsize = '10')
    axes[0].set_title('Diss. energy vs RMSE',  y=1.05, fontweight="bold", fontsize = '12')
    axes[0].grid()
    
    axes[1].scatter(plot_df.loc[:, 'E - glob'], plot_df.loc[:, 'SMSE'])
    axes[1].set_xlabel('Energy',fontsize = '10')
    axes[1].set_xlim(0,3)
    axes[1].set_ylabel('SMSE', fontsize = '10')
    axes[1].set_title('Diss. energy vs SMSE',  y=1.05, fontweight="bold", fontsize = '12')
    axes[1].grid()
    
    axes[2].scatter(plot_df.loc[:, 'E - glob'], plot_df.loc[:, 'TRAC'])
    axes[2].set_xlabel('Energy',fontsize = '10')
    axes[2].set_xlim(0,3)
    axes[2].set_ylabel('TRAC', fontsize = '10')
    axes[2].set_title('Diss. energy vs TRAC',  y=1.05, fontweight="bold", fontsize = '12')
    axes[2].grid()
    
    
    
    # plt.suptitle('Error vs dissipated energy', y=0.95, fontsize = '20', fontweight="bold" )
    
    for ax in axes:
        ax.yaxis.grid(True)
    
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    
    

                            
