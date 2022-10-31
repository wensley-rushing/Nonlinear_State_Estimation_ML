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

# Create distance matrix faster
from scipy.spatial import distance_matrix


#%% Function - P-Norm
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

folder_hyperOpt = r'output_files\Kernel Learning\HyperParameter Optimization'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = Structure.Nodes[0]

struc_periods = list(Structure.Periods[0])


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

#%% Function - Integer --> String
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

#%% Function - Random number generator
import random
def random_str_list(Index_Results, Train_procent = 0.07):
    # Remove all non valid alanysis
    Index_Results.drop(Index_Results['OK=0'][Index_Results['OK=0']!=0],axis=1, inplace=True)
    
    index_list = Index_Results.index.tolist()
    
    # INPUT Percentage of traoning data (remaining will be test data)
    proc_train = Train_procent
    num_train = int( np.ceil(len(index_list)*proc_train) )
    num_test = int( len(index_list)-num_train )
    
    # Draw random samples
    train_list = random.sample(index_list,num_train)
    test_list = random.sample(index_list,num_test)
    
    # Conver to string list  
    Train_idx = int_to_str3(train_list)
    Test_idx = int_to_str3(test_list)
    
    return Train_idx, Test_idx




#%% Gaussian Process Model for Regression
def GPR(W_par=[25, 5], #[length_subvec, length_step], 
        Ker_par=[1, 1, 0], #[sigma2_ks, tau2_ks, sigma2_error],
        
        Train_par=[['182',  '086',  '247',  '149',  '052',  '094',  '250',  '138',  
                    '156',  '251',  '248',  '073',  '163',  '025',  '258',  '249',  
                    '130',  '098',  '040',  '078',  '297',  '012'], 
                   [23, 33, 43], 
                   [32]], #[load_IDs, load_Nodes_X, load_Nodes_Y], 
        
        Test_par=[['292', '023'],
                  [23, 33, 43],
                  [32]]): #[load_IDss, load_Nodes_Xs, load_Nodes_Ys]):
    
    #%% Time - tic
    global_tic_0 = time.time()

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(global_tic_0))
    print(f'Start time: {start_time}')
    
    #%% Understanding inputs
    
    # Creation of Ws
    length_subvec = W_par[0]
    length_step = W_par[1]
    print(f'Sub-vector parameters: Length = {length_subvec}, Step = {length_step}')
    
    # Creation of kernel (Hyper-parameters)
    sigma2_ks = Ker_par[0]
    tau2_ks = Ker_par[1]
    sigma2_error = Ker_par[2]
    print(f'Hyper-parameters: Scale_Factor = {sigma2_ks}, Length_Factor = {tau2_ks}, Error_Factor = {sigma2_error} \n')

    # Training data
    load_IDs = Train_par[0]
    load_Nodes_X = Train_par[1] 
    load_Nodes_Y = Train_par[2] 
    
    # Testing data
    load_IDss = Test_par[0]
    load_Nodes_Xs = Test_par[1]
    load_Nodes_Ys = Test_par[2]
    
    
    
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
       
    #%% Obtain w vectors -- RUN Function
    
    # Training - X
    df_ZX, df_ZY = load_to_w(load_IDs, load_Nodes_X, load_Nodes_Y, len_sub_vector=length_subvec, step_size=length_step)
                           
    # Testing - X* 
    df_ZXs, df_ZYs = load_to_w(load_IDss, load_Nodes_Xs, load_Nodes_Ys, len_sub_vector=length_subvec , step_size=length_step)        
    
    # print('END - Convering to w-vectors')
    
    #%% Function Sum Kernel + Plot
    def kernel_sum(df_ZX_s, df_ZX_t, sigma2_scale_factor=1, tau2_lenth_scale=1):
        sigma2_sf = (np.ones(len(df_ZX_s.columns))*sigma2_scale_factor).tolist()
        tau2_ls =   (np.ones(len(df_ZX_s.columns))*tau2_lenth_scale).tolist()
        
        ''' OLD SLOW METHOD
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
        '''
        
        # Lenth of sides
        length_s = len(df_ZX_s.iloc[1,0])
        length_t = len(df_ZX_t.iloc[1,0])
        
        
        ker  = np.zeros((length_s,length_t), dtype=np.float64)
        for i in df_ZX_s.columns:
            
            # Sum kernel
            # Sum kernel
            sigma2_idx = df_ZX_s.columns.tolist().index(i)
            tau2_idx = df_ZX_s.columns.tolist().index(i)
            ker += sigma2_sf[sigma2_idx]*np.exp( -1/(2*tau2_ls[tau2_idx])*(distance_matrix(df_ZX_s[i]['Z'],df_ZX_t[i]['Z'],p=2)**2))
        
        
        data = ker.copy()
        
        # HeatMap
        if False:
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
    
    K = pd.DataFrame(columns=[0,1,2,3, 4], index=['K'])
    
    df_id = 0
    for dfs in [df_ZX, df_ZXs]:
        for dft in [df_ZX, df_ZXs]:
            ker_tic_k = time.time()
            ker_time_k = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ker_tic_k))
            
            print(f'Kernel: {df_id} @ {ker_time_k}')
            ker = kernel_sum(dfs, dft, sigma2_scale_factor=sigma2_ks, tau2_lenth_scale=tau2_ks)
            K[df_id]['K'] = ker
            df_id += 1
    
    # Assign error to KWW and KSS
    K[0]['K'] += sigma2_error*np.eye(len(K[0]['K']))
    K[3]['K'] += sigma2_error*np.eye(len(K[3]['K']))
    
    # Create total Kernel
    KL = np.append(K[0]['K'],K[2]['K'] ,axis=0)
    KR = np.append(K[1]['K'],K[3]['K'] ,axis=0)
    K[4]['K'] = np.append(KL,KR ,axis=1)
    #------------------------------------------------------------------------------
    ker_toc = time.time()
    ker_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ker_toc))
    #print(f'END: Determine Kernel @ {ker_time}')
    print(f'Duration [sec]: {round((ker_toc-ker_tic),4)} - [min]: {round((ker_toc-ker_tic)/60,4)} - [hrs]: {round((ker_toc-ker_tic)/60/60,4)} \n')
    
    #%% Plot of Kernel
    if True:
        length_WW = len(df_ZX.iloc[1,0])
        length_ss = len(df_ZXs.iloc[1,0])
        
        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots(1, figsize=(20*cm, 15*cm))
        #plt.figure()
        plt.imshow( K[4]['K'] , cmap = 'autumn' , interpolation = 'nearest' )
        plt.colorbar();
        
        plt.axvline(x=length_WW, ls='--', linewidth=1, color='black')
        plt.axhline(y=length_WW, ls='--', linewidth=1, color='black')
        
        # Test in figure
        font_size = 10; a_trans = 0.8
        plt.text(length_WW/2, length_WW/2, r'$K(W,W)$', 
                 fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
        plt.text(length_WW + length_ss/2, length_WW/2, r'$K(W,W^{*})$', 
                 fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
        plt.text(length_WW/2, length_WW + length_ss/2, r'$K(W^{*},W)$', 
                 fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
        plt.text(length_WW + length_ss/2, length_WW + length_ss/2, r'$K(W^{*},W^{*})$', 
                 fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
        
        # General in figure
        fig.suptitle( 'Kernel Heat Map' )
        ax.set_title(f' General: $l$ = {length_subvec}, step = {length_step} \n' +
                     f' $\sigma^2_k$ = {sigma2_ks}, $\u03C4^2_k$ = {tau2_ks}, $\sigma^2_\epsilon$ = {sigma2_error} \n' +
                     f' Input: {len(load_IDs)}, Nodes {load_Nodes_X} \n Output: {len(load_IDss)}, Nodes {load_Nodes_Y}', 
                     x=0, y=1, ha='left', va='bottom', fontsize=7)
        #plt.show()
        
        plt.savefig(os.path.join(folder_hyperOpt,f'Kernel_l{length_subvec}_step{length_step}_sigma{sigma2_ks}_tau{tau2_ks}_error{sigma2_error}.png'))
        plt.close()
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
            
        kernel_sum(df_ZX_s, df_ZX_s, sigma2_scale_factor=sigma2_ks, tau2_lenth_scale=tau2_ks)
    
    #%% Time - toc
    global_tic_1 = time.time()
    print('End time: %.4f [s]' %(global_tic_1 - global_tic_0 ))
    print('-- [min]:  %.4f [min]' %( (global_tic_1 - global_tic_0) /60))
    print('-- [hrs]:  %.4f [hrs]' %( (global_tic_1 - global_tic_0) /60/60))
    print()  
    
    
    #%% Function - Error
    
    def errors(y_true, y_pred):
        RMSE = ((y_pred - y_true)**2).mean() **.5
        
        SMSE = ((y_pred - y_true)**2).mean() / y_true.var()
            
        DIST = ((y_pred - y_true)**2).sum() **.5
        DISTN = ((y_pred/y_true - 1)**2).sum() **.5
        return RMSE, SMSE, DIST, DISTN
    
    #%% Plot Predictions and errors
    if True:
        
        columns = []
        for IDss in load_IDss:
            columns.append(IDss + f'_{load_Nodes_Y[0]}')
        
        df_error = pd.DataFrame(columns = columns, index = ['RMSE', 'SMSE', 'DIST'])
        
        print('Plotting Routine')
        print('------------------------------------------------- \n')
        
        temp = 0
        for i in range(len(load_IDss)):
            
            cm = 1/2.54  # centimeters in inches
            fig, ax = plt.subplots(2, figsize=(20*cm, 15*cm), sharex=True)
            # True acceleration vs. prediction ----------------------------------------
            #plt.figure()
            node_head = load_Nodes_Y[0] # Only one node : 32
            
            # True
            acc = df_ZYs[node_head]['ACCS'][i]    
            x_acc = np.arange(0,len(acc))*0.02
            
            ax[0].plot(x_acc, acc, 
                     alpha=0.3, linewidth=3, label='True')
            
            
            # Predict
            x_temp = np.arange(length_subvec,len(acc),length_step)*0.02 # range(length_subvec,len(acc),length_step)
            mus_temp = mus[temp:temp+len(x_temp)]
            
            ax[0].plot(x_temp, mus_temp, 
                     alpha=0.8, label='Predicted')
            
            
            #ax[0].fill_between(x_temp, mus_temp + 2*sigma_i, mus_temp-1, alpha = 0.3, color = 'tab:gray')
            
            #plt.xlabel('time [s]')
            ax[0].set_ylabel('Acceleration [m/s\u00b2]')    
            ax[0].grid()
            ax[0].legend()
            
            
            
            idx = str3_to_int(load_IDss)[i]
               
            GM = Index_Results['Ground motion'][idx]
            LF = Index_Results['Load factor'][idx]
            
            fig.suptitle(f'Acceleration in node {node_head} predicted from nodes {load_Nodes_X} \n GM: {GM}, LF: {LF}')
            ax[0].set_title(f' General: $l$ = {length_subvec}, step = {length_step} \n' +
                            f' $\sigma^2_k$ = {sigma2_ks}, $\u03C4^2_k$ = {tau2_ks}, $\sigma^2_\epsilon$ = {sigma2_error} \n' +
                            f' Input: {len(load_IDs)}, Nodes {load_Nodes_X} \n Output: {len(load_IDss)}, Nodes {load_Nodes_Y}', 
                            x=0, y=0.97, ha='left', va='bottom', fontsize=7)
            plt.xlabel('time [s]')
            fig.tight_layout()
            #plt.xlim(2000,3000)
        
        
            #% Error estimation ----------------------------------------------
            
            
            RMSE = []
            SMSE = []
            DIST = []
            DISTN = []
            for i in range(len(mus_temp)):
                y_true = np.array(acc[length_subvec:len(acc):length_step][:i])
                y_pred = mus_temp[:i]
                
                RM, SM, D, DN = errors(y_true, y_pred)
                RMSE.append(RM)
                SMSE.append(SM)
                DIST.append(D)
                DISTN.append(DN)
                  
            df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['RMSE'] = RMSE[-1]
            df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['SMSE'] = SMSE[-1]
            df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['DIST'] = DIST[-1]
                
            #plt.figure()
            ax[1].plot(x_temp, DIST, alpha=1, label='Distance')
            #plt.plot(range(length_subvec,len(acc),length_step), DISTN, alpha=1, label='Distance - Norm')
            ax[1].plot(x_temp, RMSE, alpha=1, label='RMSE')
            #ax[1].plot(x_temp, SMSE, alpha=1, label='SMSE')
            
            #plt.xlabel('time [s]')
            ax[1].set_ylabel('Measure of error')
            ax[1].grid()
            ax[1].legend()
            
            ax[1].set_title(f' Error: Dist = {round(DIST[-1],2)}, RMSE = {round(RMSE[-1],2)}, SMSE = {round(SMSE[-1],2)}', 
                         x=0, y=0.97, ha='left', va='bottom', fontsize=7)
            
            
            temp += len(x_temp)
            
            plt.savefig(os.path.join(folder_hyperOpt,f'Predict_ACC_{int_to_str3([idx])[0]}_l{length_subvec}_step{length_step}_sigma{sigma2_ks}_tau{tau2_ks}_error{sigma2_error}.png'))
            plt.close()
    
    return df_error
    

#%% INPUT

# Training data ---------------------------------------------------------------

# Indicator if total time n
load_IDs = ['182',  '086',  '247',  '149',  '052',  '094',  '250',  '138',  
            '156',  '251',  '248',  '073',  '163',  '025',  '258',  '249',  
            '130',  '098',  '040',  '078',  '297',  '012']

# Training - X                                                                                 
load_Nodes_X = [23, 33, 43] # Indicator of dimension d

# Training - Y
load_Nodes_Y = [32]

# Combine it all
#Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y]



# Testing Data ----------------------------------------------------------------

# Indicator if total time m
load_IDss = ['292', '023']
   
# Testing - X*  (Same as X)                                                                             
load_Nodes_Xs = load_Nodes_X

# Testing - Y* (Same as Y)
load_Nodes_Ys = load_Nodes_Y  

# Combine it all
#Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys]



#Creation of sub-vecors W -----------------------------------------------------
# Length of sub-vectors
length_subvec = 25

# Overlaping parameter (number of new values in sub-vector)
length_step = 5
#W_par=[length_subvec, length_step]


# Creation of kernel ----------------------------------------------------------
# Scale factor for each sensor
sigma2_ks = 1

# Length sacle
tau2_ks = 1

# Error scale for both WW and SS
sigma2_error = 0
#Ker_par=[sigma2_ks, tau2_ks, sigma2_error]


# W
length_subvec = 25; length_step = 5

# Ker
sigma2_ks = 1; tau2_ks = 1; sigma2_error = 0





#%% RUN Analysis

#------------------------------------------------------------------------------ 
# # W
# length_subvec = 25; length_step = 5

# # Ker
# sigma2_ks = 1; tau2_ks = 1; sigma2_error = 0

# for length_subvec in [10, 25, 50, 100]:
#     GPR(W_par=[length_subvec, length_step], 
#             Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
#             Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
#             Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys])
   
    
# #------------------------------------------------------------------------------  
# # W
# length_subvec = 25; length_step = 5

# # Ker
# sigma2_ks = 1; tau2_ks = 1; sigma2_error = 0
    
# for length_step in [3, 5, 10, 15]:
#     GPR(W_par=[length_subvec, length_step], 
#             Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
#             Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
#             Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys])



#------------------------------------------------------------------------------ 
# W
length_subvec = 25; length_step = 5

# Ker
sigma2_ks = 1; tau2_ks = 1; sigma2_error = 0
    
for sigma2_ks in [0.5, 1, 2, 4]:
    GPR(W_par=[length_subvec, length_step], 
            Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
            Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
            Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys])


#------------------------------------------------------------------------------ 
# # W
# length_subvec = 25; length_step = 5

# # Ker
# sigma2_ks = 1; tau2_ks = 1; sigma2_error = 0
    
# for tau2_ks in [0.5, 1, 2, 4]:
#     GPR(W_par=[length_subvec, length_step], 
#             Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
#             Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
#             Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys])


#------------------------------------------------------------------------------ 
sys.exit()
#%%

# Loops concerning sub-vectors W
for length_subvec in [10, 25, 50, 100]:
    for length_step in [5]: #[3, 5, 10, 15]:
        
        # Loops concerning Kernel hyper-parameters
        for sigma2_ks in [1]:
            for tau2_ks in [1] :#[0.5, 1, 2, 4]:
                
                GPR(W_par=[length_subvec, length_step], 
                        Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
                        Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
                        Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys])
                
