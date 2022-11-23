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

data_directory = r'output_files\error_plot'

#%% Parameters of the study, does not need to be changed

L_parameter_values = [1, 10, 20, 25, 40]

S_parameter_values = [1,2,3,4,5]

Diff_Nodes = [22, 23, 32, 42]

errors = [ 'RMSE' , 'SMSE', 'MAE' , 'MAPE', 'TRAC']

dataframe_headers = ['L1', 'L10', 'L20', 'L25', 'L40']


dataframe_index = ['s1','s2','s3', 's4','s5']

#%% Plot settings

plot_error = 'RMSE'   # which error should be plot
index_error = errors.index(plot_error)

plot_nodes = [22] # for which nodes

plot = True

df_node22 = pd.DataFrame( index = dataframe_index , columns = dataframe_headers)

df_node23 = pd.DataFrame( index = dataframe_index , columns = dataframe_headers)

df_node32 = pd.DataFrame( index = dataframe_index , columns = dataframe_headers)

df_node42 = pd.DataFrame( index = dataframe_index , columns = dataframe_headers) 

#%% Create the datasets

for node in plot_nodes:
    
    i = Diff_Nodes.index(node)
    
    folder_path = os.path.join(data_directory, f'node_{node}')
    
    for l in L_parameter_values:
    
        for s in S_parameter_values:
            
            folder_path = os.path.join(folder_path, f'L{l}_s{s}')
            
            for rdirs, dirs, files in os.walk(folder_path):
                    
                    for file in files:
                        
                        if file.endswith("Error.pkl"):
                            
                            
                            df = pd.read_pickle(os.path.join(folder_path, file))
                            
                            mean = df.mean(axis=1)
                            
                            if i == 0:
                                df_node22[f'L{l}'][f's{s}'] = mean[plot_error]
                                
                            elif i == 1:
                                df_node23[f'L{l}'][f's{s}'] = mean[plot_error]
                                
                            elif i == 2:
                                df_node32[f'L{l}'][f's{s}'] = mean[plot_error]
                                
                            elif i == 3:
                                df_node42[f'L{l}'][f's{s}'] = mean[plot_error]
                            
                            break
                    
            
                                        
 #%% Plot

if plot:
    
    for node in plot_nodes:
        i = Diff_Nodes.index(node)
        
        if  i == 0:
            df_plot = df_node22
        elif i == 1:
            df_plot = df_node23
        elif i == 2:
            df_plot = df_node32
        elif i == 3:
            df_plot = df_node42
    
    
        plt.figure(figsize =(6, 4))
        plt.pcolor(df_plot.values.tolist())
        plt.yticks(np.arange(0.5, len(df_plot.index), 1), df_plot.index)
        plt.xticks(np.arange(0.5, len(df_plot.columns), 1), df_plot.columns)
        # plt.title('Train set: 5 random GMs \nTest set: 296 GMs', loc='Left', fontsize = 9)
        plt.colorbar(label=f'{plot_error} Mean')
        
        # Lines
        # plt.axvline(x=4, ls='--', linewidth=1, color='black')
        # plt.axvline(x=8, ls='--', linewidth=1, color='black')
        
        # plt.axhline(y=4, ls='--', linewidth=1, color='black')
        # plt.axhline(y=8, ls='--', linewidth=1, color='black')
        
        # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
        
        
        # # Loop over data dimensions and create text annotations.
        for i in range(len(df_plot.index)):
            for j in range(len(df_plot.columns)):
                if round(df_plot.iloc[i,j],2) >  np.amax(df_plot.to_numpy())*0.8: #0.8:
                    text = plt.text(j+0.5, i+0.5, round(df_plot.iloc[i,j],2),
                                    ha="center", va="center", color="k", fontsize='small')#, transform = ax.transAxes)
                else:
                    text = plt.text(j+0.5, i+0.5, round(df_plot.iloc[i,j],2),
                                    ha="center", va="center", color="w", fontsize='small')#, transform = ax.transAxes)
                
        
        plt.suptitle( 'Mean ' + plot_error + f' - Node {node}', y=1.02)
        plt.xlabel('Subvectors length')
        plt.ylabel('Subvectors step size')
        plt.show()
        
        # plt.savefig(os.path.join(folder_data, f'ErrorMap_{error_text}_train{train}_IN{num_in}_OUT{num_out}.png'))
        plt.close()                     
                        
                            
