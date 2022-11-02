# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:02:21 2022

@author: s163761
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import sys
import os

# Import time-keeping
import time

# Create distance matrix faster
from scipy.spatial import distance_matrix

# For GPy
import GPy
GPy.plotting.change_plotting_library('matplotlib')

import pylab as pb

#%%
def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

#%% Folder structure
folder_data = r'output_files\Figures'


#%% INPUTS
prediction_node = 32

num_in = 5
num_out = 20

df_Basis = pd.DataFrame(columns = ['SubVec_Len', 'SubVec_Step', 'IN_EQs', 'OUT_EQs', 'IN_Nodes', 'OUT_Nodes'])
df_Error = pd.DataFrame(index = ['RMSE', 'SMSE', 'MAE', 'MAPE', 'TRAC'])

#%%
# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_data):
    print('--------------------------------------------------------------------')
    # print(rdirs)
    # print(dirs)
    # print(files)
    
    rdis_node_id = rdirs.find('node')+4
    rdis_IN_id = rdirs.find('IN')+2
    rdis_OUT_id = rdirs.find('OUT')+3
    
    if len(rdirs) > len(folder_data) and rdirs[rdis_node_id:rdis_node_id + len(str(prediction_node))] == str(prediction_node) and rdirs[rdis_IN_id:rdis_IN_id + len(str(num_in))] == str(num_in) and rdirs[rdis_OUT_id:rdis_OUT_id + len(str(num_out))] == str(num_out): # if subfolder
        print(rdirs)
        print(dirs)
        print(files)
    
    
        for file in files:
            
            if file.endswith('Basis.pkl'): 
               
                unpickled_df = pd.read_pickle(os.path.join(rdirs, file))
                df_Basis = pd.concat([df_Basis, unpickled_df], axis=0, ignore_index=True)
                
            elif file.endswith('Error.pkl'): 
                
                unpickled_df = pd.read_pickle(os.path.join(rdirs, file))
                unpickled_df['comb'] = unpickled_df.values.tolist()
                df_Error = pd.concat([df_Error, unpickled_df['comb']], axis=1, ignore_index=True)
                
if unique_cols(df_Basis)[2] == True:
    status_train ='All same training inputs'
    train = 'Same'
else:
    status_train = 'Different training inputs'
    train = 'Diff'

#%% Dataframe


# columns = ['Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5','Sensor_6','Sensor_7']
# index = ['RMSE', 'SMSE', 'TRAC']
# df = pd.DataFrame(columns=columns, index=index)

# for sensor in columns:
#     for error in index:
#         df[sensor][error] = np.random.normal(loc=100, scale=1, size=10)


#%% 
df = df_Error
# Plot bloxplot
fig, ax = plt.subplots(nrows=df.shape[0], ncols=1, figsize =(10, 10), sharex=True)


plot_right = 1

len_error = len(df.index.tolist())-1
len_sensor = len(df.columns.tolist()) + plot_right

error_id = 0
for error in df.index.tolist():
    cur_error = list(df.index)[error_id]
    
    Data = np.array([]).reshape(-1,1)
    
    sensor_id = 1
    for sensor in df.columns.tolist():
        cur_sensor = list(df.columns)[sensor_id-1]
        
        data = np.array(df[sensor][error]).reshape(-1,1)
        
        if error_id == len_error:
            ax[error_id].boxplot(data,widths=0.5, positions=[sensor_id], labels=[f'{cur_sensor}'])        
            ax[error_id].set_ylabel(f'{cur_error}')
        else:
            ax[error_id].boxplot(data,widths=0.5, positions=[sensor_id], labels=[' '])        
            ax[error_id].set_ylabel(f'{cur_error}')
        
        ax[error_id].text(x=(sensor_id-.5)/len_sensor, y=1, s=f'({round(data.mean(),2)})', va='bottom', ha='center', transform = ax[error_id].transAxes)
        #ax[error_id].set_title('subplot 1')
        
        Data = np.append(Data,data, axis=0)
        
        sensor_id+=1
        
    if plot_right == 1:    
        # Mean plot        
        ax[error_id].axvline(x = sensor_id-0.5, color = 'black', linestyle='dashed', linewidth=1.3, label = 'axvline - full height')
        
        if error_id == len_error:
            ax[error_id].boxplot(Data,widths=0.5, positions=[sensor_id], labels=['All data'])
        else:
            ax[error_id].boxplot(Data,widths=0.5, positions=[sensor_id], labels=[' '])
        
        
        ax[error_id].text(x=(sensor_id-.5)/len_sensor, y=1, s=f'({round(Data.mean(),2)})', va='bottom', ha='center', transform = ax[error_id].transAxes)
    
    
    ax[error_id].grid(axis='y')
    
    error_id+=1
    

#ax[error_id-1].set_xlabel('XXX')
fig.suptitle(f'Error for estimation of node {prediction_node} - {status_train} \n Input: {num_in}   Estimations: {num_out},  (mean)') #, fontsize=16)

plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees
#plt.tight_layout()

plt.savefig(os.path.join(folder_data, f'GeneralError_train{train}_node{prediction_node}_IN{num_in}_OUT{num_out}.png'))
