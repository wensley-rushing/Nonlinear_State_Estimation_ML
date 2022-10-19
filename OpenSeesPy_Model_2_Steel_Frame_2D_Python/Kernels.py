# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:34:20 2022

@author: s163761
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import sys
import os


#%%
def norm_p(x1, x2, p=2):
    
    
    norm_list = []
    for i in range(len(x1)):
        norm_list.append(abs(x1[i] - x2[i])**p)
    
    norm = sum(norm_list)**(1/p)
    return norm


# x1 = [1,2,3,4]
# x2 = [2,3,4,5]
   
# norm = norm_p(x1,x2,p=2)
# print(f'Norm p={2}: {norm} \n')


#%% Folder structure

folder_accs = r'output_files\ACCS'

folder_structure = r'output_files'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = Structure.Nodes[0]

struc_periods = list(Structure.Periods[0])

#%%


def load_to_w(load_IDs, load_Nodes_X, load_Nodes_Y, len_sub_vector=100, step_size=50):
    load_Nodes_X_id = []
    for i in range(len(load_Nodes_X)):
        load_Nodes_X_id.append( struc_nodes.index(load_Nodes_X[i]) )
        
    load_Nodes_Y_id = []
    for i in range(len(load_Nodes_Y)):
        load_Nodes_Y_id.append( struc_nodes.index(load_Nodes_Y[i]) )
        
    
    # Create Overall Dataframe X
    columns = np.array(load_Nodes_X) # k ---> (p)
    #index = np.array(load_IDs) # i
    df_ZX = pd.DataFrame(columns = columns , index = ['ACCS', 'Z'])
    for head in df_ZX.columns:
        df_ZX[head]['ACCS'] = []
        df_ZX[head]['Z'] = []
        
        
    # Create Overall Dataframe Y
    columns = np.array(load_Nodes_Y) # k ---> (p)
    #index = np.array(load_IDs) # i
    df_ZY = pd.DataFrame(columns = columns , index = ['ACCS', 'Yi'])
    for head in df_ZY.columns:
        df_ZY[head]['ACCS'] = []
        df_ZY[head]['Yi'] = []
    
    
    # r=root, d=directories, f = files
    for rdirs, dirs, files in os.walk(folder_accs):
        for file in files:
            
            # Load Ground Motions for X/Y
            if rdirs == folder_accs and file.endswith("Accs.out") and file[3:6] in load_IDs:
                #print(os.path.join(rdirs, file))
                #print(idx)
                print('Loading file: ',file)
                
                time_Accs = np.loadtxt( os.path.join(folder_accs, file) )
                
                if file[3:6][0] != str(0):
                    idx = int(file[3:6])
                elif file[3:6][1] != str(0):
                    idx = int(file[4:6])
                else:
                    idx = int(file[5:6])
                        
                GM = Index_Results['Ground motion'][idx]
                LF = Index_Results['Load factor'][idx]
                
                print('GM: ',GM ,'Loadfactor: ', LF)
                
                # Load Accelerations in nodes X
                for j in range(len(load_Nodes_X)):
                    #time = time_Accs[:,0]
                    accs = time_Accs[:,load_Nodes_X_id[j]+1].tolist()
                    #accs = [1,2,3,4,5]
                    
                    df_ZX[load_Nodes_X[j]]['ACCS'].append( accs )
                    
                    
                    # Create sum vector from accelerations in node
                    
                    # length of sub-vector 1 <= L <= len (x) 
                    l = len_sub_vector 
    
                    # 1 <= step <=  L-1  --!! IF STEP >= L NO OVERLAP !!--
                    move_step = step_size 
                    #(if step = 0 then the vector never changes...)
                    #(if step = 1 then the vector takes 1 step [1 new value + spits out 1 value] )
                    #(if step = L then the vector takes L steps NO OVERLAP [L new values + spits out L values] )
                    # Range of sub-vectors z_ik (i = last element in sub-vector)
                    for i in range(l,len(accs)+1, move_step):
                        #print(f'i={i}--')
                        
                        z = []
                        # Range of sub-vector elements in z_ik
                        for idx in range(i-l,i):
                            
                            z.append(accs[idx])
                        df_ZX[load_Nodes_X[j]]['Z'].append( z )
                        #print(z)
                        
                # Load Accelerations in nodes Y
                for j in range(len(load_Nodes_Y)):
                    #time = time_Accs[:,0]
                    accs = time_Accs[:,load_Nodes_Y_id[j]+1].tolist()
                    #accs = [11,33,55,77,99]
                    
                    df_ZY[load_Nodes_Y[j]]['ACCS'].append( accs )
                    
                    df_ZY[load_Nodes_Y[j]]['Yi'].extend( accs[l-1::move_step] )
                
    return df_ZX, df_ZY     
   

#%% Getting ws
length_subvec = 25; length_step = 5


# Training - X
load_IDs = ['000','001', '002', '003', '004', '005', '006', '007', '008', '009'] # Indicator if total time n
load_Nodes_X = [23, 33, 43] # Indicator of dimension d

# Training - Y
#load_IDs : Same as for Training X
load_Nodes_Y = [32]

df_ZX, df_ZY = load_to_w(load_IDs, load_Nodes_X, load_Nodes_Y, len_sub_vector=length_subvec, step_size=length_step)


# Testing - X*
load_IDss = ['010'] # Indicator if total time m
load_Nodes_Xs = load_Nodes_X

# Testing - Y*
#load_IDs : Same as for Testing X*
load_Nodes_Ys = load_Nodes_Y                 
    
df_ZXs, df_ZYs = load_to_w(load_IDss, load_Nodes_Xs, load_Nodes_Ys, len_sub_vector=length_subvec , step_size=length_step)        

print('End Loading')
#sys.exit()

#%% Kernel

def kernel_sum(df_ZX_s, df_ZX_t):
    sigma = np.ones(len(df_ZX_s.columns)).tolist()
    
    index_s = range(len(df_ZX_s.iloc[1,0]))
    index_t = range(len(df_ZX_t.iloc[1,0]))
    
    #print(f'S: {index_s}, T: {index_t}')
    
        
    df_w = pd.DataFrame(columns = index_t , index = index_s)
    df_w.fillna(0, inplace=True)
    
    #df_w_pro = df_w.copy()
    df_w_sum = df_w.copy()
    
    for s in index_s:
        for t in index_t:
            
            for j in df_ZX_s.columns:
                # Product kernel
                #df_w_pro[s][t] += -1/(2*sigma[j-1])* norm_p(df_ZX[j][s], df_ZX[j][t], p=2)**2 
                
                # Sum kernel
                sigma_idx = df_ZX_s.columns.tolist().index(j) 
                df_w_sum[t][s] += np.exp(-1/(2*sigma[sigma_idx])* norm_p(df_ZX_s[j]['Z'][s], df_ZX_t[j]['Z'][t], p=2)**2 )
    
    #df_w_pro = np.exp(df_w_pro)
    #print(df_w)
    
    #print('Product kernel: \n', df_w_pro)
    #print('Sum kernel: \n', df_w_sum)
    
    # HeatMap
    if True:
        data = df_w_sum.to_numpy()
        plt.imshow( data , cmap = 'autumn' , interpolation = 'nearest' )
        plt.colorbar();
        
        plt.title( 'Heat Map of Kernel' )
        plt.show()
    
    return data

#%%

K = pd.DataFrame(columns=[0,1,2,3], index=['K'])

df_id = 0
for dfs in [df_ZX, df_ZXs]:
    for dft in [df_ZX, df_ZXs]:
        #print(df_id)
        a = kernel_sum(dfs, dft)
        K[df_id]['K'] = a
        df_id += 1

'''0: WW, 1: WS, 2: SW, 3: SS '''
y = np.array(df_ZY[load_Nodes_Y[0]]['Yi'])
mus = K[2]['K'].dot( np.linalg.inv(K[0]['K']) ).dot(y)

Sigma = K[3]['K'] - K[2]['K'].dot( np.linalg.inv(K[0]['K']) ).dot(K[1]['K'])
#%%

df_ZX_s = df_ZX.copy()

for head in df_ZX_s.columns.tolist():
    #print()
    #print(df_ZX_s[head]['Z'])
    df_ZX_s[head]['Z'].extend(df_ZXs[head]['Z'])
    #print()
    #print(df_ZX_s[head]['Z'])
    
kernel_sum(df_ZX_s, df_ZX_s)

#%% Plots
plt.figure()
node_head = load_Nodes_Y[0]
plt.plot(df_ZYs[node_head]['ACCS'][0], alpha=0.2)
plt.plot(range(length_subvec,len(df_ZYs[node_head]['ACCS'][0]),length_step), mus)
plt.grid()

