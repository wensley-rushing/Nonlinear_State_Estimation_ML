# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:00:41 2022

@author: s163761
"""

#%% Imports

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
#import GPy
#GPy.plotting.change_plotting_library('matplotlib')

import pylab as pb

import DamageTools


#%% Folder structure

folder_accs = r'output_files\ACCS'

folder_structure = r'output_files'

folder_figure_save = r'output_files\Entropy'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = Structure.Nodes[0]

struc_periods = list(Structure.Periods[0])

df_SampEn = pd.DataFrame(0,columns = struc_nodes, index = Index_Results.index)

#%%

# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_accs):
    for file in files:
        
        # Load Ground Motions for X/Y
        if rdirs == folder_accs and file.endswith("Accs.out"):
            #print(os.path.join(rdirs, file))
            #print(idx)
            #print('Loading file: ',file)
            
            time_Accs = np.loadtxt( os.path.join(folder_accs, file) )
            
            if file[3:6][0] != str(0):
                idx = int(file[3:6])
            elif file[3:6][1] != str(0):
                idx = int(file[4:6])
            else:
                idx = int(file[5:6])
                    
            # GM = Index_Results['Ground motion'][idx]
            # LF = Index_Results['Load factor'][idx]
            # print('GM: ',GM ,'Loadfactor: ', LF)
            print('EarthQuake ID: ', idx)
            
            
            
            # Load Accelerations in nodes X
            for j in range(1,len(time_Accs[0])):
                #time = time_Accs[:,0]
                accs = time_Accs[:,j].tolist()
                
                SampEn = DamageTools.SampEn(accs, 2, 0.2*np.std(accs))
                
                df_SampEn[struc_nodes[j-1]][idx] = SampEn
                
            
#%%
df_SampEn.to_pickle(folder_structure + "/00_SampEn.pkl")
#%%
df_SampEn = pd.read_pickle( os.path.join(folder_structure, '00_SampEn.pkl') ) 

#%% 
x = struc_periods
cm = 1/2.54  # centimeters in inches
for i in df_SampEn.index:
    
    GM = Index_Results['Ground motion'][i]
    
        
    fig = plt.figure(figsize=(20*cm, 15*cm)); ax = fig.add_subplot(111)
    x = df_SampEn.columns.tolist()
    plt.plot(x,df_SampEn.loc[i].tolist())
    plt.grid()
    
    plt.xticks(x[0:len(x):4] + x[3:len(x):4])
    
    rel_height = 0.35
    plt.text(x=0.09, y=rel_height, s='Ground',    rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    plt.text(x=0.37, y=rel_height, s='1st Floor', rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    plt.text(x=0.65, y=rel_height, s='2nd Floor', rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    plt.text(x=0.92, y=rel_height, s='3rd Floor', rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    
    for j in [0, 1, 2, 3]:
        plt.axvspan(x[4*j], x[(4*j)+3], alpha=0.4, color='tab:blue')
    
    plt.xlabel('Nodes')
    plt.ylabel('SampEn')
    
    plt.title(f' SampEn estimations of Accelerations \n {GM}')
    
    
    plt.savefig(os.path.join(folder_figure_save, f'ACC_Entropy_{GM}.png'))
    plt.close()
    

#%%

df = df_SampEn
# Plot bloxplot
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25*cm, 15*cm))
#plt.figure(figsize =(10, 10)); ax = fig.add_subplot(111)


len_sensor = len(df.columns.tolist())

 #for error in df.index.tolist():
     #cur_error = list(df.index)[error_id]

sensor_id = 1
for sensor in df.columns.tolist():
    cur_sensor = list(df.columns)[sensor_id-1]
    
    data = np.array(df[sensor].tolist()).reshape(-1,1)
    
    ax.boxplot(data,widths=0.5, positions=[sensor_id], labels=[f'{cur_sensor}'])  
    ax.text(x=(sensor_id-.5)/(len_sensor), y=1, s=f'({round(data.mean(),2)})', va='bottom', ha='center', transform = ax.transAxes)
        
    if sensor_id in [5, 9, 13]:
        ax.axvline(x = sensor_id-0.5, color = 'black', linestyle='dashed', linewidth=1.3, label = 'axvline - full height')
    
    sensor_id+=1


rel_height = 0.99
plt.text(x=0.12, y=rel_height, s='Ground',    rotation=0, va='top', ha='center', transform = ax.transAxes)
plt.text(x=0.37, y=rel_height, s='1st Floor', rotation=0, va='top', ha='center', transform = ax.transAxes)
plt.text(x=0.63, y=rel_height, s='2nd Floor', rotation=0, va='top', ha='center', transform = ax.transAxes)
plt.text(x=0.87, y=rel_height, s='3rd Floor', rotation=0, va='top', ha='center', transform = ax.transAxes)

plt.grid(axis='y')
plt.title('SampEn of Acceleration \n All Earthquake loadings (mean) \n')
plt.xlabel('Nodes')
plt.ylabel('SampEn')


plt.savefig(os.path.join(folder_figure_save, 'All_ACC_Entropy.png'))
#plt.close()
    
    

