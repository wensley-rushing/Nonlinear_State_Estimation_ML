# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:55:51 2022

@author: larsk
"""

import openseespy.opensees as ops
import opsvis as opsv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import signal
from scipy.fft import fftshift


#from Model_definition_2D_frame import createModel
from Model_definition_3x3_frame import createModel
from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord

import sys
import os


#%% Folder structure

folder_structure = r'C:\Users\larsk\Danmarks Tekniske Universitet\Thesis_Nonlinear-Damage-Detection\OpenSeesPy_Model_2_Steel_Frame_2D_Python\output_files'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )

df_beta = pd.read_pickle( os.path.join(folder_structure, '00_Beta_Verification.pkl') )

struc_elements = df_beta['Element ID'][0]

#%%
load_IDs = ['000', '001', '002', '003', '004', '005']
load_Elements = [1121]
values_Beta = [0.05, 0.1, 0.6]

load_Elements_id = []
for i in range(len(load_Elements)):
    load_Elements_id.append( struc_elements.index(load_Elements[i]) )
    
    
load_IDs_id = []
for i in range(len(load_IDs)):
    
    if load_IDs[i][0] != str(0):
        idx = int(load_IDs[i])
    elif load_IDs[i][1] != str(0):
        idx = int(load_IDs[i][1:])
    else:
        idx = int(load_IDs[i][2])
        
    load_IDs_id.append(idx)
    
    GM = Index_Results['Ground motion'][idx]
    LF = Index_Results['Load factor'][idx]
            
    

for i in load_IDs_id:
    for j in load_Elements_id:
    
        # df_beta = pd.DataFrame(columns = ['Element ID', 'beta', 'Section ID (PA el.)', 'PA el.', 'PA el. - class', 'PA T1 Time', 'PA T2 Time', 'PA el. Time', 'Curvature Time'])
        curvature = df_beta['Curvature Time'][i][j]
        PA_idx = df_beta['PA el. Time'][i][j]
        PA_Beta = df_beta['beta'][i][j]
        
        PA_Beta_T1 = df_beta['PA T1 Time'][i][j]
        PA_Beta_T2 = df_beta['PA T2 Time'][i][j]
        
        plt.figure()
        plt.scatter(curvature,PA_idx, c = 'k', label=f'beta = {PA_Beta}')
        
        
        for betas in values_Beta:
            PA_Beta_T12  = np.array(PA_Beta_T1) + betas*np.array(PA_Beta_T2)
            
            plt.scatter(curvature,PA_Beta_T12, label=f'beta = {betas}', alpha = 0.8, s = 5)
    
        plt.title(f'PA Index vs. Curvature  Element {struc_elements[j]}, beta = {PA_Beta} \n {GM} , Lf = {LF}')
        plt.xlabel('Curvature [-]')
        plt.ylabel('PA Index [-]')
        plt.legend()
        plt.grid()
        plt.xlim((0,0.10))
        plt.ylim((0,10))
        plt.show()
    
sys.exit()
    
