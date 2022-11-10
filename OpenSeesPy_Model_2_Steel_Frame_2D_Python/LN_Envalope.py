# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:21:43 2022

@author: s163761
"""


#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

#%%

folder_structure = r'output_linear_non'


#%%

Index_Results = pd.read_pickle(os.path.join(folder_structure, '00_Index_Results.pkl'))
Structure = pd.read_pickle(os.path.join(folder_structure, '00_Structure.pkl'))

#%%

LN_nodes = Index_Results['LN_Node']
LN_energy = Index_Results['LN_Energy']
LN_res_def = Index_Results['LN_Res_Def']

struc_nodes = Structure['Nodes'][0]

df_LN = pd.DataFrame('L', columns=struc_nodes, index = Index_Results.index)

energy_thr = 0.01 # Larger values are non-linear

plt.figure()

for i in range(LN_nodes.shape[0]):
    for j in range(len(LN_nodes[0])): # 2 in each element (21 elements)
        node_0 = LN_nodes[i][j][0]
        node_1 = LN_nodes[i][j][1]
        
        energy_0 = LN_energy[i][j][0]
        energy_1 = LN_energy[i][j][1]
        
        res_dif_0 = np.abs(LN_res_def[i][j][0])
        res_dif_1 = np.abs(LN_res_def[i][j][1])
        
        if energy_0 > energy_thr:
            df_LN[node_0][i] = 'N'
            
        if energy_1 > energy_thr:
            df_LN[node_1][i] = 'N'
            

        plt.scatter([energy_0, energy_1],[res_dif_0, res_dif_1])#, c='tab:blue')
        
plt.grid()
plt.xlabel('Energy')
plt.ylabel('Elasic deformation (residual rotation)')
plt.axvline(energy_thr, linewidth=1, linestyle='--', c='k')
        


sys.exit()
# Results from Damage Index
#df.to_csv(output_directory + r'/00_Index_Results.csv')  # export dataframe to cvs
#df.to_pickle(output_directory + "/00_Index_Results.pkl") 
#unpickled_df = pd.read_pickle("./dummy.pkl")  