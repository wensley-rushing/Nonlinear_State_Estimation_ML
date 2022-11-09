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
import GPy
GPy.plotting.change_plotting_library('matplotlib')

import pylab as pb

import DamageTools


#%% Folder structure

folder_accs = r'output_files\ACCS'

folder_structure = r'output_files'

folder_figure_save = r'output_files\Testing'

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

for i in df_SampEn.index:
    
    x = df_SampEn.columns.tolist()
    plt.plot(x,df_SampEn.loc[1].tolist())
    plt.grid()
    
    plt.xlabel('Nodes')
    plt.ylabel('SampEn')
    