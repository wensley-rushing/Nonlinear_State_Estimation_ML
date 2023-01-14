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
#import time

# Create distance matrix faster
#from scipy.spatial import distance_matrix

# For GPy
import GPy
GPy.plotting.change_plotting_library('matplotlib')

#import pylab as pb

#%%
def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

#%% Folder structure
folder_data = r'output_files\GP_K1_Fold_300_Noise_1000'
folder_data = r'output_files\Figures_Singular'
folder_data = r'output_NN\Linear\K1_Fold_900_Pred_23'

# GP / NN Trained on Same 20 GMs and predict same 281 GMs
folder_data = r'output_NN\Linear\K1_Fold_300_Noise_1000'
# folder_data = r'output_NN\Linear\NN_9Matrix_20_EQs_V2'


#%% INPUTS
# prediction_node = 43

# # EQs
# num_in = 5
# num_out = 296

# plot_ErrorMap = True

def GenError(prediction_node=32, EQ_IN_OUT=[5,296], ID_error=0, plot_ErrorMap=False):
    # Understanding input
    # EQs
    num_in = EQ_IN_OUT[0]
    num_out = EQ_IN_OUT[1]
    
    
    #%% DataFrame Initilization
    df_Basis = pd.DataFrame(columns = ['SubVec_Len', 'SubVec_Step', 'IN_EQs', 'OUT_EQs', 'IN_Nodes', 'OUT_Nodes'])
    df_Error = pd.DataFrame(index = ['RMSE', 'SMSE', 'MAE', 'MAPE', 'TRAC'])
    df_ErrorMap_idx = 0
    #%%
    # r=root, d=directories, f = files
    for rdirs, dirs, files in os.walk(folder_data):
        #print('--------------------------------------------------------------------')
        # print(rdirs)
        # print(dirs)
        # print(files)
        
        for file in files:
            if file.endswith('.pkl') and file.startswith('00_HeatMap'): 
                if df_ErrorMap_idx == 0:
                    df_ErrorMap = pd.read_pickle(os.path.join(rdirs, file))
                    df_ErrorMap_idx +=1 
                else:
                    unpickled_df = pd.read_pickle(os.path.join(rdirs, file))
                    #unpickled_df['comb'] = unpickled_df.values.tolist()
                    df_ErrorMap = pd.concat([df_ErrorMap, unpickled_df], axis=1, ignore_index=False)
                
                
        
        rdis_node_id = rdirs.find('node')+4
        pred_node = str(prediction_node); len_pred_node = rdirs[rdis_node_id:].find('_')
        
        rdis_IN_id = rdirs.find('IN')+2
        pred_IN = str(num_in); len_pred_IN = rdirs[rdis_IN_id:].find('_')
        
        rdis_OUT_id = rdirs.find('OUT')+3
        pred_OUT = str(num_out); len_pred_OUT = rdirs[rdis_OUT_id:].find('_')
        

        
        if (len(rdirs) > len(folder_data)  and rdirs.count('\\') == folder_data.count('\\')+1              # if sub-folder
            and rdirs[rdis_node_id:rdis_node_id + len_pred_node] == pred_node       # if 
            #and rdirs[rdis_IN_id:rdis_IN_id + len_pred_IN] == pred_IN 
            #and rdirs[rdis_OUT_id:rdis_OUT_id + len_pred_OUT] == pred_OUT
            ):
            print(rdirs)
            #print(dirs)
            #print(files)
            #print(rdirs[rdis_OUT_id:rdis_OUT_id + len_pred_OUT], pred_OUT)
        
        
            for file in files:
                
                if file.endswith('Basis.pkl'): 
                   
                    unpickled_df = pd.read_pickle(os.path.join(rdirs, file))
                    df_Basis = pd.concat([df_Basis, unpickled_df], axis=0, ignore_index=True)
                    #print('Basis loaded')
                    
                elif file.endswith('Error.pkl'): 
                    
                    unpickled_df = pd.read_pickle(os.path.join(rdirs, file))
                    unpickled_df['comb'] = unpickled_df.values.tolist()
                    df_Error = pd.concat([df_Error, unpickled_df['comb']], axis=1, ignore_index=True)
                    print('Errors loaded')
                    
                    
            
                
                    
    # if unique_cols(df_Basis)[2] == True:
    #     status_train ='All same training inputs'
    #     train = 'Same'
    # else:
    #     status_train = 'Different training inputs'
    #     train = 'Diff'
        
    status_train ='All same training inputs'
    train = 'Same'
    
    
    #%% 
    df = df_Error.copy()
    df.drop('MAPE', inplace=True)
    df.drop('MAE', inplace=True)
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
            cur_sensor = [23, 32, 42][sensor_id-1]
            
            data = np.array(df[sensor][error]).reshape(-1,1)
            
            if error_id == len_error:
                ax[error_id].boxplot(data,widths=0.5, positions=[sensor_id], labels=[f'{cur_sensor}'])        
                ax[error_id].set_ylabel(f'{cur_error}', fontsize=14)
            else:
                ax[error_id].boxplot(data,widths=0.5, positions=[sensor_id], labels=[' '])        
                ax[error_id].set_ylabel(f'{cur_error}', fontsize=14)
            
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
    fig.suptitle(f'Error for estimation of node {prediction_node} \n Input: {num_in}   Estimations: {num_out},  (mean)') #, fontsize=16)
    
    plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees
    #plt.tight_layout()
    plt.xlabel('Training Nodes', fontsize=14)
    
    plt.savefig(os.path.join(folder_data, f'GeneralError_train{train}_node{prediction_node}_IN{num_in}_OUT{num_out}.png'))
    # plt.close()
    
    #%% Plot ErrorMap
    error_text = ['RMSE', 'SMSE', 'MAE', 'MAPE' , 'TRAC'][ID_error]
    if plot_ErrorMap:
        plt.figure(figsize =(8, 7))
        plt.pcolor(df_ErrorMap.values.tolist())
        plt.yticks(np.arange(0.5, len(df_ErrorMap.index), 1), df_ErrorMap.index)
        plt.xticks(np.arange(0.5, len(df_ErrorMap.columns), 1), df_ErrorMap.columns)
        plt.colorbar(label=f'{error_text} Mean')
        
        # Lines
        if False:
            plt.axvline(x=4, ls='--', linewidth=1, color='black')
            plt.axvline(x=8, ls='--', linewidth=1, color='black')
            
            plt.axhline(y=4, ls='--', linewidth=1, color='black')
            plt.axhline(y=8, ls='--', linewidth=1, color='black')
        
        plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees
        
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(df_ErrorMap.index)):
            for j in range(len(df_ErrorMap.columns)):
                if round(df_ErrorMap.iloc[i,j],2) >  np.amax(df_ErrorMap.to_numpy())*0.8: #0.8:
                    text = plt.text(j+0.5, i+0.5, round(df_ErrorMap.iloc[i,j],2),
                                   ha="center", va="center", color="k", fontsize='small')#, transform = ax.transAxes)
                else:
                    text = plt.text(j+0.5, i+0.5, round(df_ErrorMap.iloc[i,j],2),
                                   ha="center", va="center", color="w", fontsize='small')#, transform = ax.transAxes)
        
        
        plt.suptitle( f'Error Heat Map - IN: {num_in}, OUT: {num_out} \n {error_text} Error' )
        plt.xlabel('Estimation Nodes', fontsize=14)
        plt.ylabel('Training Nodes', fontsize=14)
        #plt.show()
        
        plt.savefig(os.path.join(folder_data, f'ErrorMap_train{train}_IN{num_in}_OUT{num_out}_{error_text}.png'))
        #plt.close()
        
        df_ErrorMap.to_pickle(folder_data + '/00_ErrorMap.pkl') 
    
    #%% Heat Map of erros & Location
    
    def mean(list0):
        if len(list0) != 0:
            av = sum(list0) / len(list0)
        else:
            av=0
        return av
    
    nodes = [20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
    nodes = [23, 32, 42]
    
        
    df_map = pd.DataFrame(columns = [f'{prediction_node}'], index = nodes)
    
    
    for In_node in nodes:
        
        in_node = [In_node]
        true_idx = []
        for i in df_Basis.index:
            if df_Basis['IN_Nodes'][i] == in_node:
                #print(i)
                true_idx.append(i)
        
        #for i in [0,1,2,3,4]:        
            #print(mean(sum(df_Error[true_idx].values.tolist()[4], [])))
        df_map[f'{prediction_node}'][in_node] = mean(sum(df_Error[true_idx].values.tolist()[ID_error], []))
            
    df_map.to_pickle(folder_data + f'/00_HeatMap_results_{prediction_node}.pkl') 
    
    return df_Error

#%% RUN

Struc_Nodes = [20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
Struc_Nodes = [23, 32, 42]

EQ_IN_OUT = [20,281]

for i in [1]:
    for Node in Struc_Nodes:
        df_Error = GenError(Node, EQ_IN_OUT=EQ_IN_OUT, ID_error=i, plot_ErrorMap=False)
        
    GenError(Struc_Nodes[0], EQ_IN_OUT=EQ_IN_OUT, ID_error=i, plot_ErrorMap=True)

sys.exit()
#%% Load
RMSE_Mean = []
for i in df_Error.columns:
    value = np.array(df_Error[i]['RMSE']).mean()
    RMSE_Mean.append(round(value,4))
    

RMSE_Mean = np.array(RMSE_Mean)
print(round(RMSE_Mean.mean(),4))
print(round(RMSE_Mean.std(),4))
