# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:34:29 2022

@author: gabri


‘back propagation feed forward network’ 
20 neurons 
the network hidden layer have tangential sigmoid type as transfer functions, 
the transfer function in the output layer being of the linear type.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import sys
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#%% Folder structure

folder_accs = r'output_files\ACCS'

folder_structure = r'output_files'

folder_figure_save = r'output_files\18_tests\Test_11_V2'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = Structure.Nodes[0]

    #%% Function - Create w vectors
    
    
def vertical_append(matrix, subvec_list):
    
    if len(matrix) == 0:
        matrix = np.array(subvec_list)
    else:
        matrix = np.row_stack((matrix, np.array(subvec_list)))
        
    return matrix

    

def load_to_matrix(load_IDs, load_Nodes_X, len_sub_vector=100, step_size=50):
    load_Nodes_X_id = []
    for i in range(len(load_Nodes_X)):
        load_Nodes_X_id.append( struc_nodes.index(load_Nodes_X[i]) )
    
    # length of sub-vector 1 <= L <= len (x) 
    l = len_sub_vector 
     
    # 1 <= step <=  L-1  --!! IF STEP >= L NO OVERLAP !!--
    move_step = step_size

    train_matrix_acc = []
    train_matrix_time = []
     
    # r=root, d=directories, f = files
    for rdirs, dirs, files in os.walk(folder_accs):
        for file in files:
            
            # Load Ground Motions for X/Y
            if rdirs == folder_accs and file.endswith("Accs.out") and file[3:6] in load_IDs:
                #print(os.path.join(rdirs, file))
                #print(idx)
                #print('Loading file: ',file)
                
                time_Accs = np.loadtxt( os.path.join(folder_accs, file) )  # load the txt file: col1 = time, col2=acc[n10] ...
                
                if file[3:6][0] != str(0):
                    idx = int(file[3:6])
                elif file[3:6][1] != str(0):
                    idx = int(file[4:6])
                else:
                    idx = int(file[5:6])
                        
                # GM = Index_Results['Ground motion'][idx]
                # LF = Index_Results['Load factor'][idx]
                # print('GM: ',GM ,'Loadfactor: ', LF)
                
                # Load Accelerations in nodes X
                for j in range(len(load_Nodes_X)):
                    #time = time_Accs[:,0]
                    accs = time_Accs[:12,load_Nodes_X_id[j]+1].tolist()
                    time = np.arange(0,len(accs))*0.02
                    #accs = list(range(1,11, 1))
                    
                    # df_ZX[load_Nodes_X[j]]['ACCS'].append( accs )   # save whole signal as 1 list
                    
                    
                    # Create sum vector from accelerations in node
                    
                    
                    #(if step = 0 then the vector never changes...)
                    #(if step = 1 then the vector takes 1 step [1 new value + spits out 1 value] )
                    #(if step = L then the vector takes L steps NO OVERLAP [L new values + spits out L values] )
                    
                    # Range of sub-vectors z_ik (i = last element in sub-vector)
                    
                    for i in range(l,len(accs)+1, move_step):
                        t = []
                        z = []
                        # Range of sub-vector elements in z_ik
                        for idx in range(i-l,i):
                            t.append(time[idx])
                            z.append(accs[idx])
                            
                            
                            
                        # z is the subvector with length = L and s...
                        
                        train_matrix_acc = vertical_append(train_matrix_acc, z)   
                        train_matrix_time = vertical_append(train_matrix_time, z)
                                
    return train_matrix_acc, train_matrix_time, accs, time

    


#%%

len_sub_vector=3
step_size=3

train_matrix_acc, train_matrix_time, acc, time = load_to_matrix(['000'], [10], len_sub_vector, step_size)

plt.figure()
plt.plot(acc, time)
for i in range(0, np.shape(train_matrix_acc)[0]):
    # plt.figure()
    plt.plot(train_matrix_acc[i], train_matrix_acc[i])
    
plt.legend()


