# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 08:49:02 2022

@author: s163761
"""

import numpy as np
import pandas as pd
import os

import sys

#%% Folder structure

folder_lf_1 = r'output_files'  
folder_lf_0 = r'output_files_LF_0'
folder_lf_2 = r'output_files_LF_2'
# folder_lf_2C = r'output_files_Conv'

folder_merged = r'output_files_All'

#%% Load pkl files

# LF = 1.0
df_Results_1 = pd.read_pickle(os.path.join(folder_lf_1, '00_Index_Results.pkl'))
#df_Betas_1 = pd.read_pickle(os.path.join(folder_lf_1, '00_Beta_Verification.pkl'))
df_Structure_1 = pd.read_pickle(os.path.join(folder_lf_1, '00_Structure.pkl'))

# LF = 0.5
df_Results_0 = pd.read_pickle(os.path.join(folder_lf_0, '00_Index_Results.pkl'))
#df_Betas_0 = pd.read_pickle(os.path.join(folder_lf_0, '00_Beta_Verification.pkl'))
#df_Structure_0 = pd.read_pickle(os.path.join(folder_lf_0, '00_Structure.pkl'))

# LF = 1.5
df_Results_2 = pd.read_pickle(os.path.join(folder_lf_2, '00_Index_Results.pkl'))
#df_Betas_2 = pd.read_pickle(os.path.join(folder_lf_2, '00_Beta_Verification.pkl'))
#df_Structure_2 = pd.read_pickle(os.path.join(folder_lf_2, '00_Structure.pkl'))

# LF = 1.5 (Converged)
# df_Results_2C = pd.read_pickle(os.path.join(folder_lf_2C, '00_Index_Results.pkl'))


# Results are needed to be merged for all the files
# Beta is no longer of interest since these are related to Park-Ang index (Invalid)
# All Structure are the same. Only one is needed

#%% Merge so all GMs are converged

# if False:
#     # Remove non-converged GM in original df
#     df_Results_2.drop(644, inplace=True)
    
#     # Add converged GM and sort according to index
#     df_Results_2 = pd.concat([df_Results_2, df_Results_2C])
#     df_Results_2.sort_index(ascending=True, inplace=True)
    
#     # Save corrected df
#     df_Results_2.to_pickle(os.path.join(folder_lf_2, '00_Index_Results.pkl'))

# sys.exit()
# #%% Merge Results

# frames = [df_Results_1, df_Results_0, df_Results_2]
# df_Results_Merged = pd.concat(frames)

# df_Structure_Merged = df_Structure_1.copy()

# #%% Save pkl files

# df_Results_Merged.to_pickle(os.path.join(folder_merged, '00_Index_Results.pkl'))
# df_Structure_Merged.to_pickle(os.path.join(folder_merged, '00_Structure.pkl'))

#%%

IDs_L = pd.DataFrame(columns = ['GM', 'Energy'])
IDs_NL = pd.DataFrame(columns = ['GM', 'Energy'])

for idx_0 in df_Results_0.index: # takes the ID of the GM in first dataset
    
    if df_Results_0.loc[idx_0, 'E - glob'] <= 2:
        
        idx_2 = idx_0 + 301
        
        if df_Results_0.loc[idx_0, 'Ground motion'] != df_Results_2.loc[idx_2, 'Ground motion']:
            print('Error: the two GMs do not correspond! Check index')
        
        if df_Results_2.loc[idx_2, 'E - glob'] > 2:
            
            IDs_L.loc[idx_0] = df_Results_0.loc[idx_0, 'Ground motion'], df_Results_0.loc[idx_0, 'E - glob']
            IDs_NL.loc[idx_2] = df_Results_2.loc[idx_2, 'Ground motion'], df_Results_2.loc[idx_2, 'E - glob']
            


IDs_L.to_pickle(os.path.join('output_NN', 'Linear' , 'Physical_inter','IDs_linear.pkl'))
IDs_NL.to_pickle(os.path.join('output_NN', 'Linear' , 'Physical_inter','IDs_non_linear.pkl'))

            
