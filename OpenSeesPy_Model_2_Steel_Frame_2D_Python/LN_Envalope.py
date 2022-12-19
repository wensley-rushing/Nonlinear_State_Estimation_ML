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

folder_structure = r'output_files_All'
# folder_structure = r'output_files'


#%%

Index_Results = pd.read_pickle(os.path.join(folder_structure, '00_Index_Results.pkl'))
Structure = pd.read_pickle(os.path.join(folder_structure, '00_Structure.pkl'))

#%% Based on Local Energy

# LN_nodes = Index_Results['LN_Node']
# LN_energy = Index_Results['LN_Energy']
# LN_res_def = Index_Results['LN_Res_Def']

# struc_nodes = Structure['Nodes'][0]

# df_LN = pd.DataFrame('L', columns=struc_nodes, index = Index_Results.index)

# energy_thr = 0.01 # Larger values are non-linear

# plt.figure()

# for i in range(LN_nodes.shape[0]):
#     for j in range(len(LN_nodes[0])): # 2 in each element (21 elements)
#         node_0 = LN_nodes[i][j][0]
#         node_1 = LN_nodes[i][j][1]
        
#         energy_0 = LN_energy[i][j][0]
#         energy_1 = LN_energy[i][j][1]
        
#         res_dif_0 = np.abs(LN_res_def[i][j][0])
#         res_dif_1 = np.abs(LN_res_def[i][j][1])
        
#         if energy_0 > energy_thr:
#             df_LN[node_0][i] = 'N'
            
#         if energy_1 > energy_thr:
#             df_LN[node_1][i] = 'N'
            

#         plt.scatter([energy_0, energy_1],[res_dif_0, res_dif_1])#, c='tab:blue')
        
# plt.grid()
# plt.xlabel('Energy')
# plt.ylabel('Elasic deformation (residual rotation)')
# plt.axvline(energy_thr, linewidth=1, linestyle='--', c='k')
        


# sys.exit()
# Results from Damage Index
#df.to_csv(output_directory + r'/00_Index_Results.csv')  # export dataframe to cvs
#df.to_pickle(output_directory + "/00_Index_Results.pkl") 
#unpickled_df = pd.read_pickle("./dummy.pkl")  

#%% Based on Global Energy

E_Glob = Index_Results['E - glob']
num_tot = E_Glob.shape[0]

# Number of L/N
for i in [301,602,903]:
    
    NL = E_Glob[i-301:i][E_Glob[i-301:i]<=1.5].shape[0]
    NN = E_Glob[i-301:i][E_Glob[i-301:i]>1.5].shape[0]
    print(f'L/N {NL}/{NN}, ({NL+NN})')

NL = E_Glob[E_Glob<=1.5].shape[0]
NN = E_Glob[E_Glob>1.5].shape[0]
print(f'L/N {NL}/{NN}, ({NL+NN})')


# Threshold
thr_L = [1.25, 1.5, 2.0]
thr_L = np.linspace(0, E_Glob.max(), num=1000)
num_L = []

for thr in thr_L:
    num = E_Glob[E_Glob <= thr].shape[0]
    num_L.append(num / num_tot)
    
    
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4), constrained_layout=True, sharex=True)
plt.plot(thr_L, num_L)
plt.axvline(x=1.5, ls='--', linewidth=1.2, color='black')
plt.text(1.5,1,'1.5 kNm', va='top', ha='right', rotation=90, fontsize=14)

plt.xlim(0,10)

plt.grid()
axes.xaxis.set_tick_params(labelsize=14)
axes.yaxis.set_tick_params(labelsize=14)

plt.title(f'Change of Linear Responses given different thresholds \n Number of Ground Motions: {num_tot}', fontsize=14)
plt.xlabel('Global Energy [kNm] \n Threshold between Linear and Non-linear regime', fontsize=14)
plt.ylabel('Ratio of Linear Resonses [-]', fontsize=14)
