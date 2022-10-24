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

# Import time-keeping
import time


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

#%% Time - tik
global_tic_0 = time.time()

start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(global_tic_0))
print(f'Start time: {start_time} \n')

#%% Function - Create w vectors

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
    df_ZY = pd.DataFrame(columns = columns , index = ['ACCS', 'Yi','Yi_list'])
    for head in df_ZY.columns:
        df_ZY[head]['ACCS'] = []
        df_ZY[head]['Yi'] = []
        df_ZY[head]['Yi_list'] = []
    
    # r=root, d=directories, f = files
    for rdirs, dirs, files in os.walk(folder_accs):
        for file in files:
            
            # Load Ground Motions for X/Y
            if rdirs == folder_accs and file.endswith("Accs.out") and file[3:6] in load_IDs:
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
                    df_ZY[load_Nodes_Y[j]]['Yi_list'].append( accs[l-1::move_step] )
                
    return df_ZX, df_ZY     
   

#%% Obtain w-vectors -- RUN Function
# w_tic = time.time()
# w_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(w_tic))
# print(f'Convering to w-vectors @ {w_time}')
#------------------------------------------------------------------------------

length_subvec = 25; length_step = 5


# Training - X
load_IDs = ['000','001', '002', '003', '004', '005', '006', '007', '008', '009'] # Indicator if total time n
load_Nodes_X = [23, 33, 43] # Indicator of dimension d

# Training - Y
#load_IDs : Same as for Training X
load_Nodes_Y = [32]

df_ZX, df_ZY = load_to_w(load_IDs, load_Nodes_X, load_Nodes_Y, len_sub_vector=length_subvec, step_size=length_step)


# Testing - X*
load_IDss = ['010', '011'] # Indicator if total time m
load_Nodes_Xs = load_Nodes_X

# Testing - Y*
#load_IDs : Same as for Testing X*
load_Nodes_Ys = load_Nodes_Y                 
    
df_ZXs, df_ZYs = load_to_w(load_IDss, load_Nodes_Xs, load_Nodes_Ys, len_sub_vector=length_subvec , step_size=length_step)        

# print('END - Convering to w-vectors')

#%% Function Sum Kernel + Plot
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
    data = df_w_sum.to_numpy()
    
    # HeatMap
    if True:
        plt.imshow( data , cmap = 'autumn' , interpolation = 'nearest' )
        plt.colorbar();
        
        plt.title( 'Heat Map of Kernel' )
        plt.show()
    
    return data

#%% Obtain Full Kernel K -- RUN Function
ker_tic = time.time()
ker_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ker_tic))
print(f'Determine Kernel @ {ker_time}')
#------------------------------------------------------------------------------

K = pd.DataFrame(columns=[0,1,2,3], index=['K'])

df_id = 0
for dfs in [df_ZX, df_ZXs]:
    for dft in [df_ZX, df_ZXs]:
        ker_tic_k = time.time()
        ker_time_k = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ker_tic_k))
        
        print(f'Kernel: {df_id} @ {ker_time_k}')
        a = kernel_sum(dfs, dft)
        K[df_id]['K'] = a
        df_id += 1


#------------------------------------------------------------------------------
ker_toc = time.time()
ker_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ker_toc))
#print(f'END: Determine Kernel @ {ker_time}')
print(f'Duration [sec]: {round((ker_toc-ker_tic),4)} - [min]: {round((ker_toc-ker_tic)/60,4)} - [hrs]: {round((ker_toc-ker_tic)/60/60,4)} \n')
#%% Determine mean: mu and variance Sigma
'''0: WW, 1: WS, 2: SW, 3: SS '''
y = np.array(df_ZY[load_Nodes_Y[0]]['Yi'])
mus = K[2]['K'].dot( np.linalg.inv(K[0]['K']) ).dot(y)

Sigma = K[3]['K'] - K[2]['K'].dot( np.linalg.inv(K[0]['K']) ).dot(K[1]['K'])

sigma_i = np.diagonal(Sigma)**.5
#%% TEST - Obtain FULL Kernel 
if False:
    df_ZX_s = df_ZX.copy()
    
    for head in df_ZX_s.columns.tolist():
        #print()
        #print(df_ZX_s[head]['Z'])
        df_ZX_s[head]['Z'].extend(df_ZXs[head]['Z'])
        #print()
        #print(df_ZX_s[head]['Z'])
        
    kernel_sum(df_ZX_s, df_ZX_s)

#%%
global_tic_1 = time.time()
print('End time: %.4f [s]' %(global_tic_1 - global_tic_0 ))
print('-- [min]:  %.4f [min]' %( (global_tic_1 - global_tic_0) /60))
print('-- [hrs]:  %.4f [hrs]' %( (global_tic_1 - global_tic_0) /60/60))
print()

#%% Function - String --> Integer

def str3_to_int(list_str):
    
    '''
    Takes list of index in (string) 000 format
    Reurns list of index in integers
    E.g.:
        0   --> '000'
        20  --> '020'
        100 --> '100'
    '''
    
    list_int = []
    for i in list_str:
    
        if i[0] != str(0):
            list_int.append( int(i[0:3]))
        elif i[1] != str(0):
            list_int.append( int(i[1:3]))
        else:
            list_int.append( int(i[2:3]))
            
    return list_int



#%% Function - Error

def errors(y_true, y_pred):
    RMSE = ((y_pred - y_true)**2).mean() **.5
        
    DIST = ((y_pred - y_true)**2).sum() **.5
    DISTN = ((y_pred/y_true - 1)**2).sum() **.5
    return RMSE, DIST, DISTN

#%% Plots 
print('Plotting Routine')

temp = 0
for i in range(len(load_IDss)):
    
    # True acceleration vs. prediction ----------------------------------------
    plt.figure()
    node_head = load_Nodes_Y[0] # Only one node : 32
    acc = df_ZYs[node_head]['ACCS'][i]
    
    plt.plot(acc, 
             alpha=0.3, linewidth=3, label='True')
    
    x_temp = range(length_subvec,len(acc),length_step)
    mus_temp = mus[temp:temp+len(x_temp)]
    
    plt.plot(x_temp, mus_temp, 
             alpha=0.8, label='Predicted')
    
    
    
    
    plt.grid()
    plt.legend()
    
    
    
    idx = str3_to_int(load_IDss)[i]
       
    GM = Index_Results['Ground motion'][idx]
    LF = Index_Results['Load factor'][idx]
    
    plt.title(f'Acceleration node {node_head} from nodes {load_Nodes_X} \n GM: {GM}, LF: {LF}')
    #plt.xlim(2000,3000)


    # Error estimation --------------- ----------------------------------------
    if True:
        df_dist = pd.DataFrame({'True': acc[length_subvec:len(acc):length_step], 'Pred': mus_temp})
        
        
        
        RMSE = []
        DIST = []
        DISTN = []
        for i in range(len(mus_temp)):
            y_true = np.array(acc[length_subvec:len(acc):length_step][:i])
            y_pred = mus_temp[:i]
            
            R, D, DN = errors(y_true, y_pred)
            RMSE.append(R)
            DIST.append(D)
            DISTN.append(DN)
        
        
        plt.figure()
        plt.plot(range(length_subvec,len(acc),length_step), DIST, alpha=1, label='Distance')
        #plt.plot(range(length_subvec,len(acc),length_step), DISTN, alpha=1, label='Distance - Norm')
        plt.plot(range(length_subvec,len(acc),length_step), RMSE, alpha=1, label='RMSE')
        plt.grid()
        plt.legend()
        
        plt.title('Error measurement over time')
    
    
    temp += len(x_temp)


#%% Function - Random number generator
import random
# Remove all non valid alanysis
Index_Results.drop(Index_Results['OK=0'][Index_Results['OK=0']!=0],axis=1, inplace=True)

index_list = Index_Results.index.tolist()

proc_train = 0.7
num_train = int( np.ceil(len(index_list)*proc_train) )
num_test = int( len(index_list)-num_train )

train_list = random.sample(index_list,num_train)
test_list = random.sample(index_list,num_test)

def int_to_str3(list_int):
    
    '''
    Takes list of index (integers)
    Reurns list index in (string) 000 format
    E.g.:
        0   --> '000'
        20  --> '020'
        100 --> '100'
    '''
    
    list_str = []
    for i in list_int:
    
        i_str = str(i)
        
        if len(i_str) == 1:
            list_str.append( f'00{i_str}')
        elif len(i_str) == 2:
            list_str.append( f'0{i_str}')
        else:
            list_str.append( f'{i_str}')
            
    return list_str
        
Train_idx = int_to_str3(train_list)
Test_idx = int_to_str3(test_list)