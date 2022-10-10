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

# from scipy import signal
# from scipy.fft import fftshift


#from Model_definition_2D_frame import createModel
from Model_definition_3x3_frame import createModel
from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord

import sys
import os


#%% Folders

folder_structure = 'output_files'
output_capacity_directory = ('elements_capacity')

#%% Load Capacity Files

# AnySection

AnySection = pd.read_csv((output_capacity_directory+'/anysection_curves.csv'), usecols = [1,2,3,4])
AnySection.iloc[:,1] = - AnySection.iloc[:,1]
AnySection.iloc[:,3] = - AnySection.iloc[:,3]

col_yielding_idx = AnySection[AnySection.iloc[:,3]==56.06].index.values.astype(int)[0]
beam_yielding_idx = AnySection[AnySection.iloc[:,1]==123.26].index.values.astype(int)[0]

col_ult_idx = AnySection[AnySection.iloc[:,3]==67.04].index.values.astype(int)[0]
beam_ult_idx = AnySection[AnySection.iloc[:,1]==127.04].index.values.astype(int)[0]

# OpenSees Pushover


beam_capacity = pd.read_csv((output_capacity_directory + "/beam_pushover.csv"), usecols = [1,2,3,4,5,6])

curv_y_beam = beam_capacity['Yield - curv'][0]
M_y_beam = beam_capacity['Yield - M'][0]

curv_u_beam = beam_capacity['Ult - curv'][0]
M_u_beam = beam_capacity['Ult - M'][0]

column_capacity = pd.read_csv((output_capacity_directory + "/column_pushover.csv"), usecols = [1,2,3,4,5,6])

curv_y_col = column_capacity['Yield - curv'][0]
M_y_col = column_capacity['Yield - M'][0]

curv_u_col = column_capacity['Ult - curv'][0]
M_u_col = column_capacity['Ult - M'][0]


#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )

df_beta = pd.read_pickle( os.path.join(folder_structure, '00_Beta_Verification.pkl') )

struc_elements = df_beta['Element ID'][0]

#%%
load_IDs = ['000']
load_Elements = [1121]


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
    
    
            
    

for i in load_IDs_id:
    for j in load_Elements_id:
        
        GM = Index_Results['Ground motion'][i]
        LF = Index_Results['Load factor'][i]
    
        # df_beta = pd.DataFrame(columns = ['Element ID', 'beta', 'Section ID (PA el.)', 'PA el.', 'PA el. - class', 'PA T1 Time', 'PA T2 Time', 'PA el. Time', 'Curvature Time'])
        curvature = df_beta['Curvature Time'][i][j]
        PA_idx = df_beta['PA el. Time'][i][j]
        PA_Beta = df_beta['beta'][i][j]
        
        PA_Beta_T1 = df_beta['PA T1 Time'][i][j]
        PA_Beta_T2 = df_beta['PA T2 Time'][i][j]
        
        
        
        
        fig, ax = plt.subplots()
        
        twin1 = ax.twinx()
        
        if struc_elements[j] in Structure['Beam El'][0]:
            values_Beta = [0.05, 0.15, 0.69, 0.3]
            ax.plot(AnySection.iloc[:beam_ult_idx,0], AnySection.iloc[:beam_ult_idx,1], "b-", label = 'AnySection')
            ax.plot(beam_capacity.iloc[0, ::2], beam_capacity.loc[:,'M'], label = 'OpenSees')
            ax.plot([0, curv_y_beam, curv_u_beam], [0, M_y_beam, M_u_beam])
        else:
            values_Beta = [0.05]
            ax.plot(AnySection.iloc[:col_ult_idx,2], AnySection.iloc[:col_ult_idx,3], "b-", label = 'AnySection')
            ax.plot(column_capacity.loc[:,'Curv'], column_capacity.loc[:,'M'], label = 'OpenSees')
            ax.plot([0, curv_y_col, curv_u_col], [0, M_y_col, M_u_col])
        for betas in values_Beta:
            PA_Beta_T12  = np.array(PA_Beta_T1) + betas*np.array(PA_Beta_T2)
            twin1.scatter(curvature,PA_Beta_T12, label=f'beta = {betas}', alpha = 0.8, s = 5)
    
        
        twin1.axhline(y=1, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 1.02, 'C')

        twin1.axhline(y=0.5, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 0.52, 'S')

        twin1.axhline(y=0.2, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 0.22, 'MO')

        twin1.axhline(y=0.1, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 0.12, 'MI')
    
        ax.set_title(f'PA Index and Element {struc_elements[j]} capacity\n {GM} , Lf = {LF}')
        ax.set_xlabel('Curvature [-]')
        ax.set_ylabel('Moment [kNm]')
        twin1.set_ylabel('PA Index [-]')
        
        ax.yaxis.label.set_color('blue')
        twin1.yaxis.label.set_color('black')
        
        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='y', colors='blue', **tkw)
        ax.tick_params(axis='x', **tkw)
        twin1.tick_params(axis='y', colors='black', **tkw)
        
        
        
        # ax.set_xlim(0, max(curvature)*1.6)
        # ax.set_ylim(0, 70)
        # twin1.set_ylim(0, 1.2)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.15, 1))
        twin1.legend(loc='center left', bbox_to_anchor=(1.15, 0.75))
        plt.show()
        
        
        fig, ax = plt.subplots()
        
        twin1 = ax.twinx()
        
        if struc_elements[j] in Structure['Beam El'][0]:
            values_Beta = [0.05, 0.15, 0.69, 0.3]
            ax.plot(AnySection.iloc[:beam_ult_idx,0], AnySection.iloc[:beam_ult_idx,1], "b-", label = 'AnySection')
            ax.plot(beam_capacity.loc[:,'Curv'], beam_capacity.loc[:,'M'], label = 'OpenSees')
            ax.plot([0, curv_y_beam, curv_u_beam], [0, M_y_beam, M_u_beam])
        else:
            values_Beta = [0.05]
            ax.plot(AnySection.iloc[:col_ult_idx,2], AnySection.iloc[:col_ult_idx,3], "b-", label = 'AnySection')
            ax.plot(column_capacity.loc[:,'Curv'], column_capacity.loc[:,'M'], label = 'OpenSees')
            ax.plot([0, curv_y_col, curv_u_col], [0, M_y_col, M_u_col])
        for betas in values_Beta:
            PA_Beta_T12  = np.array(PA_Beta_T1) + betas*np.array(PA_Beta_T2)
            twin1.scatter(curvature,PA_Beta_T12, label=f'beta = {betas}', alpha = 0.8, s = 5)
    
        
        twin1.axhline(y=1, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 1.02, 'C')

        twin1.axhline(y=0.5, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 0.52, 'S')

        twin1.axhline(y=0.2, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 0.22, 'MO')

        twin1.axhline(y=0.1, color = 'black', linestyle = '--', linewidth=1)
        twin1.text(0, 0.12, 'MI')
    
        ax.set_title(f'PA Index and Element {struc_elements[j]} capacity\n {GM} , Lf = {LF}')
        ax.set_xlabel('Curvature [-]')
        ax.set_ylabel('Moment [kNm]')
        twin1.set_ylabel('PA Index [-]')
        
        ax.yaxis.label.set_color('blue')
        twin1.yaxis.label.set_color('black')
        
        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='y', colors='blue', **tkw)
        ax.tick_params(axis='x', **tkw)
        twin1.tick_params(axis='y', colors='black', **tkw)
        
        
        
        ax.set_xlim(0, max(curvature)*1.5)
        # ax.set_ylim(0, 70)
        # twin1.set_ylim(0, 1.2)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.15, 1))
        twin1.legend(loc='center left', bbox_to_anchor=(1.15, 0.75))
        plt.show()
        
    
sys.exit()
    
