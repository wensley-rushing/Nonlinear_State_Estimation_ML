# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:06:35 2022

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

# For GPy
import GPy
GPy.plotting.change_plotting_library('matplotlib')

import pylab as pb

import DamageTools

#%% Initialize figures
plt.figure()
plt.plot(range(0,10))
plt.close()

#%% LOG FILE
"""
Transcript - direct print output to a file, in addition to terminal.

Usage:
    import transcript
    transcript.start('logfile.log')
    print("inside file")
    transcript.stop()
    print("outside file")
"""

#import sys

class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        #self.log2 = open('logfile.txt', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def start(filename):
        """Start transcript, appending print output to given filename"""
        sys.stdout = Transcript(filename)
    
    def stop():
        """Stop transcript and return print functionality to normal"""
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
    
    
# Transcript.start('logfile0.txt')
# print("inside file")
# Transcript.stop()
# print("outside file")


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

folder_figure_save = r'output_NN\Linear\Study_LR_Epoch_GA'

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


#%% Input
# # Training data ---------------------------------------------------------------
# Train_data, Test_data = random_str_list(Index_Results, Train_procent = .9)
# # sys.exit()
# # Indicator if total time n
# #load_IDs = Train_data # 0.015 --> 5
# load_IDs = ['108', '001', '231', '079', '251']
# load_IDs = Train_data
# load_IDss = Test_data
# if '015' in load_IDs:
#     print('Add/Remove 015')
#     load_IDs.remove('015')
    
#     load_IDss.append('015')

# # load_IDs = ['052']

# # Training - X                                                                                 
# load_Nodes_X = [23] # Indicator of dimension d

# # Training - Y
# load_Nodes_Y = [42]

# # Combine it all
# Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y]


# # Testing Data ----------------------------------------------------------------


# # Indicator if total time m
# #load_IDss = Test_data # 20
# # load_IDss = ['012', '277', '015']
# # load_IDss = load_IDss
# #load_IDss = ['023']  
# # load_IDss = int_to_str3(Index_Results.index.tolist())
# # for i in load_IDs:
# #     load_IDss.remove(i)

# # Testing - X*  (Same as X)                                                                             
# load_Nodes_Xs = load_Nodes_X

# # Testing - Y* (Same as Y)
# load_Nodes_Ys = load_Nodes_Y  

# # Combine it all
# Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys]



# #Creation of sub-vecors W -----------------------------------------------------
# # Length of sub-vectors
# length_subvec = 25

# # Overlaping parameter (number of new values in sub-vector)
# length_step = 5
# length_step_test = 1

# W_par=[length_subvec, length_step, length_step_test]

# # Model Optimization Y/N
# optimize_model = 1

# # Creation of kernel ----------------------------------------------------------
# # Scale factor for each sensor

# # Create Epoch / Batch / Learning rate ----------------------------------------
# Epochs = 30

# # Train Batch Size
# train_batch = 25


# # Learning rate
# learning_rate = 1e-5

# Hyper_par = [Epochs, train_batch, learning_rate]


# # if False:
# #     GPR(W_par, 
# #                 Ker_par, 
# #                 Train_par, 
# #                 Test_par)



#%% Neural Network Model for Regression
def NNR(W_par=[25, 5, 1, 20], #[length_subvec, length_step], 
        Hyper_par = [10, 25, 1e-5], #[Epochs, train_batch, learning_rate],
        
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
    start_time_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(global_tic_0))
    #print(f'Start time: {start_time}') MOVED
    
    
    
    #%% Understanding inputs
    
    # Creation of Ws
    length_subvec = W_par[0]
    length_step = W_par[1]
    length_step_test = W_par[2]
    hidden_dim = W_par[3]
    #print(f'Sub-vector parameters: Length = {length_subvec}, Step = {length_step}') MOVED
    
    # Creation Epoch / Batch / Learning rate (Hyper-parameters) 
    epochs = Hyper_par[0]
    BatchSize = Hyper_par[1]  #(Training)
    learning_rate = Hyper_par[2]
    learning_rate_exp = exp_number = "{:.2e}".format(learning_rate)
    #print(f'Hyper-parameters: Scale_Factor = {sigma2_ks}, Length_Factor = {tau2_ks}, Error_Factor = {sigma2_error} \n') MOVED
    
    # Training data
    load_IDs = Train_par[0]; load_IDs.sort()
    load_Nodes_X = Train_par[1] 
    load_Nodes_Y = Train_par[2] 
    
    # Testing data
    load_IDss = Test_par[0]; load_IDss.sort()
    load_Nodes_Xs = Test_par[1]
    load_Nodes_Ys = Test_par[2]
    
    
    #%% Create Sub-folder for plots
    sub_folder_plots = f'Pred_node{load_Nodes_Ys[0]}_IN{len(load_IDs)}_OUT{len(load_IDss)}_Time{start_time_name}'
      
    # Create applicabe sub-folder
    os.mkdir(os.path.join(folder_figure_save, sub_folder_plots))
    
    
    Transcript.start(os.path.join(folder_figure_save, sub_folder_plots, '00_logfile.txt'))
    # print("inside file")
    print(f'Start time: {start_time}')
    print(f'Sub-vector parameters: Length = {length_subvec}, Step = {length_step}')
    print(f'Hyper-parameters: Epochs = {epochs}, BatchSize = {BatchSize}, Learn.Rate = {learning_rate_exp} \n')
    # Transcript.stop()
    # print("outside file")
    
    #%%
    # Create bassi for error estimations
    df_error_basis = pd.DataFrame(columns = ['SubVec_Len', 'SubVec_Step', 'IN_EQs', 'OUT_EQs', 'IN_Nodes', 'OUT_Nodes'])
    df_error_basis['SubVec_Len'] = [length_subvec]
    df_error_basis['SubVec_Step'] = [length_step] #--------------------------------
    df_error_basis['IN_EQs'] = [load_IDs]
    df_error_basis['OUT_EQs'] = [load_IDss]
    df_error_basis['IN_Nodes'] = [load_Nodes_X]
    df_error_basis['OUT_Nodes'] = [load_Nodes_Ys]
    
    # General Structure
    df_error_basis.to_pickle(os.path.join(folder_figure_save, sub_folder_plots, '00_Basis.pkl'))
    
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
        df_ZX = pd.DataFrame(columns = columns , index = ['ACCS', 'Z', 'Z_list'])
        for head in df_ZX.columns:
            df_ZX[head]['ACCS'] = []
            df_ZX[head]['Z'] = []
            df_ZX[head]['Z_list'] = []
            
            
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
                        #accs = list(range(1,11, 1))
                        
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
                        Zi = []
                        for i in range(l,len(accs)+1, move_step):
                            #print(f'i={i}--')
                            
                            z = []
                            # Range of sub-vector elements in z_ik
                            for idx in range(i-l,i):
                                
                                z.append(accs[idx])
                            df_ZX[load_Nodes_X[j]]['Z'].append( z )
                            Zi.append(z)
                        df_ZX[load_Nodes_X[j]]['Z_list'].append( Zi )
                            #print(z)
                            
                    # Load Accelerations in nodes Y
                    for j in range(len(load_Nodes_Y)):
                        #time = time_Accs[:,0]
                        accs = time_Accs[:,load_Nodes_Y_id[j]+1].tolist()
                        #accs = list(range(100,1100,100))
                        
                        df_ZY[load_Nodes_Y[j]]['ACCS'].append( accs )
                        
                        df_ZY[load_Nodes_Y[j]]['Yi'].extend( accs[l-1::move_step] )
                        df_ZY[load_Nodes_Y[j]]['Yi_list'].append( accs[l-1::move_step] )
                    
        return df_ZX, df_ZY     
       
    #%% Obtain w vectors -- RUN Function
    
    # Training - X
    df_ZX, df_ZY = load_to_w(load_IDs, load_Nodes_X, load_Nodes_Y, len_sub_vector=length_subvec, step_size=length_step)
    
    X = df_ZX[df_ZX.columns[0]]['Z']; X = np.array(X)
    if len(list(df_ZX.columns)) > 1:
        for Node in list(df_ZX.columns)[1:]:
            X = np.append(X,df_ZX[Node]['Z'], axis=1)
        
    Y = df_ZY[df_ZY.columns[0]]['Yi']
    Y = np.array(Y).reshape(-1,1)
    
    
                           
    # Testing - X* (Validation)
    df_ZXs, df_ZYs = load_to_w(load_IDss, load_Nodes_Xs, load_Nodes_Ys, len_sub_vector=length_subvec , step_size=length_step_test)        
    
    Xs = df_ZXs[df_ZXs.columns[0]]['Z']; Xs = np.array(Xs)
    if len(list(df_ZXs.columns)) > 1:
        for Node in list(df_ZXs.columns)[1:]:
            Xs = np.append(Xs,df_ZXs[Node]['Z'], axis=1)
              
    Ys = df_ZYs[df_ZYs.columns[0]]['Yi']
    Ys = np.array(Ys).reshape(-1,1)
    
    
    
    df_Xs_list = pd.DataFrame(columns = [0], index = list(range(0,len(load_IDss))))
    for i in df_Xs_list.index.tolist():
        df_Xs_list[0][i] = np.array(df_ZXs[df_ZXs.columns[0]]['Z_list'][i])
        
        if len(list(df_ZXs.columns)) > 1:
            for Node in list(df_ZXs.columns)[1:]:
                df_Xs_list[0][i] = np.append(df_Xs_list[0][i], df_ZXs[Node]['Z_list'][i], axis=1)
    
    # print('END - Convering to w-vectors')
    
    
    #%% Implement new structure to PyTorch ----------------------------------------
    from torch.utils.data import Dataset, DataLoader
    
    class ElecDataset(Dataset):
        def __init__(self,feature,target):
            self.feature = feature
            self.target = target
        
        def __len__(self):
            return len(self.feature)
        
        def __getitem__(self,idx):
            item = self.feature[idx]
            label = self.target[idx]
            
            return item, label
        
    #%% Determine train / Testing Data in Certain Input Structure
    
    Case_Nr = 1
    
    
    if Case_Nr == 0:
        # Case 0  
        #X0 = X.reshape(X.shape[0],X.shape[1],1)  
        train = ElecDataset(X.reshape(X.shape[0],X.shape[1],1),Y)
        valid = ElecDataset(Xs.reshape(Xs.shape[0],Xs.shape[1],1),Ys)
        
        input_pred = Xs.reshape(Xs.shape[0],Xs.shape[1],1)
    elif Case_Nr == 1:
        # Case 1
        # X1 = X.reshape(X.shape[0],int(X.shape[1]/25) ,25)   
        train = ElecDataset(X.reshape(X.shape[0],int(X.shape[1]/length_subvec),length_subvec),Y)
        valid = ElecDataset(Xs.reshape(Xs.shape[0],int(Xs.shape[1]/length_subvec),length_subvec),Ys)
        
        input_pred = Xs.reshape(Xs.shape[0],int(Xs.shape[1]/length_subvec),length_subvec)
    elif Case_Nr == 2:
        # Case 2 
        # X2 = X.reshape(X.shape[0],1,int(X.shape[1]/25) ,25)
        train = ElecDataset(X.reshape(X.shape[0],1,int(X.shape[1]/length_subvec),length_subvec),Y)
        valid = ElecDataset(Xs.reshape(Xs.shape[0],1,int(Xs.shape[1]/length_subvec),length_subvec),Ys)
        
        input_pred = Xs.reshape(Xs.shape[0],1,int(Xs.shape[1]/length_subvec),length_subvec)
    
    
    
    train_loader = DataLoader(train,batch_size=BatchSize,shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid,batch_size=1,shuffle=False, drop_last=False)
    
    x_train, y_train = next(iter(train_loader))
    x_valid, y_valid = next(iter(valid_loader))
    
    
    
    
    #%% Define model
    import torch
    import gc
    import torch.nn as nn
    from tqdm import tqdm_notebook as tqdm
    from torch.utils.data import Dataset,DataLoader
    
    #%% Model Definition
    
    # Model Case 0
    class ANN_Sigmoid_0(nn.Module):
        def __init__(self, length_subvec, num_X_channels, hidden_dim):
            super(ANN_Sigmoid_0,self).__init__()
            self.lin1 = nn.Linear(length_subvec*num_X_channels, hidden_dim)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.lin2 = nn.Linear(hidden_dim, 1)
           
            
        def forward(self,x):
            # print(f'Input: {x.size()}')
            x = x[:,:,-1]
            # print(f'Rehaped Input: {x.size()}')
            x = self.lin1(x)
            # print(f'Lin_1: {x.size()}')
            # x = self.sigmoid(x)
            x = self.relu(x)
            # print(f'Sigmoid: {x.size()}')
            x = self.lin2(x)
            # print(f'Output: {x.size()}')
            
            return x
        
    # -----------------------------------------------------------------------------
    
    # Model Case 1
    class ANN_Sigmoid_1(nn.Module):
        def __init__(self, length_subvec, num_X_channels, hidden_dim):
            super(ANN_Sigmoid_1,self).__init__()
            self.flat = nn.Flatten()
            self.lin1 = nn.Linear(length_subvec*num_X_channels, hidden_dim)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.lin2 = nn.Linear(hidden_dim, 1)
           
            
        def forward(self,x):
            # print(f'Input: {x.size()}')
            x = self.flat(x)
            # print(f'Input: {x.size()}')
            x = self.lin1(x)
            # print(f'Output conv: {x.size()}')
            # x = self.sigmoid(x)
            x = self.relu(x)
            # print(f'Output relu 1: {x.size()}')
            x = self.lin2(x)
            # print(f'Output linear 1: {x.size()}')
            
            return x
        
    if False:
        '''
        # Model Case 0
        class CNN_ForecastNet0(nn.Module):
            def __init__(self, length_subvec, num_X_channels):
                super(CNN_ForecastNet0,self).__init__()
                self.conv1d = nn.Conv1d(length_subvec*num_X_channels,128,kernel_size=1)
                self.relu = nn.ReLU()
                self.fc1 = nn.Linear(128,50) # Training    
                self.fc2 = nn.Linear(50,1)        
                
            def forward(self,x):
                # print(f'Input: {x.size()}')
                x = self.conv1d(x)
                # print(f'Output conv: {x.size()}')
                x = self.relu(x)
                # print(f'Output relu 1: {x.size()}')
                #x = x.view(-1)
                x = x[:,:,-1]
                # print(f'Output relu 1 - reshape : {x.size()}')
                x = self.fc1(x)
                # print(f'Output linear 1: {x.size()}')
                x = self.relu(x)
                # print(f'Output relu 2: {x.size()}')
                x = self.fc2(x)
                # print(f'Output linear 2 = prediction: {x.size()}')
                
                return x
            
        # -----------------------------------------------------------------------------
        
        # Model Case 1
        class CNN_ForecastNet1(nn.Module):
            def __init__(self, length_subvec, num_X_channels):
                super(CNN_ForecastNet1,self).__init__()
                self.conv1d = nn.Conv1d(num_X_channels,128,kernel_size=length_subvec)
                self.relu = nn.ReLU()
                self.fc1 = nn.Linear(128,50) # Training    
                self.fc2 = nn.Linear(50,1)        
                
            def forward(self,x):
                # print(f'Input: {x.size()}')
                x = self.conv1d(x)
                # print(f'Output conv: {x.size()}')
                x = self.relu(x)
                # print(f'Output relu 1: {x.size()}')
                #x = x.view(-1)
                x = x[:,:,-1]
                # print(f'Output relu 1 - reshape : {x.size()}')
                x = self.fc1(x)
                # print(f'Output linear 1: {x.size()}')
                x = self.relu(x)
                # print(f'Output relu 2: {x.size()}')
                x = self.fc2(x)
                # print(f'Output linear 2 = prediction: {x.size()}')
                
                return x
            
        # -----------------------------------------------------------------------------
        
        # Model Case 2
        class CNN_ForecastNet2(nn.Module):
            def __init__(self, length_subvec, num_X_channels):
                super(CNN_ForecastNet2,self).__init__()
                self.conv2d1 = nn.Conv2d(1, 128, kernel_size = num_X_channels, stride=(1,1), padding=(1,1))
                self.ap2d1 = nn.AvgPool2d(kernel_size=1, stride=1)
                self.conv2d2 = nn.Conv2d(128, 20, kernel_size = num_X_channels, stride=(1,1), padding=(1,1))
                self.ap2d2 = nn.AvgPool2d(kernel_size=2, stride=2)  
                self.flat = nn.Flatten()
                self.fc1 = nn.Linear(240,50)    
                self.fc2 = nn.Linear(50,1) 
                
            def forward(self,x):
                # print(f'Input: {x.size()}')
                x = self.conv2d1(x)
                # print(f'Output conv: {x.size()}')
                x = self.ap2d1(x)
                # print(f'Output AvgPool: {x.size()}')
                x = self.conv2d2(x)
                # print(f'Output conv - reshape : {x.size()}')
                x = self.ap2d2(x)
                # print(f'Output AvgPool: {x.size()}')
                x = self.flat(x)
                # print(f'Output flatten: {x.size()}')
                x = self.fc1(x)
                # print(f'Output linear 1 = prediction: {x.size()}')
                x = self.fc2(x)
                # print(f'Output linear 2 = prediction: {x.size()}')
                
                return x
        '''    
    #%% Choose Device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device( "cpu")
    
    if Case_Nr == 0:
        # Case 0  
        model = ANN_Sigmoid_0(length_subvec, len(load_Nodes_X), hidden_dim).to(device)
    elif Case_Nr == 1:
        # Case 1
        model = ANN_Sigmoid_1(length_subvec, len(load_Nodes_X), hidden_dim).to(device)
    elif Case_Nr == 2:
        # Case 2 
        model = ANN_Sigmoid_1(length_subvec, len(load_Nodes_X), hidden_dim).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()#nn.MSELoss()
    loss_text = 'MSE'
    
    
    #%% Define Train/Test Functions
    train_losses = []
    valid_losses = []
    def Train():
        
        running_loss = .0
        
        model.train()
        
        for idx, (inputs,labels) in enumerate(train_loader):
            inputs = inputs.to(device).float() # X: float32
            labels = labels.to(device).float() # Y: float32
            
            optimizer.zero_grad()
            preds = model(inputs) # X
            loss = criterion(preds,labels)  # Pred, Y
            loss.backward()
            optimizer.step()
            running_loss += loss
            
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss.cpu().detach().numpy())
        
        print_text = f'train_loss {train_loss}'
        # print(print_text)
        return train_loss.cpu().detach().numpy().tolist()
    
    #------------------------------------------------------------------------------    
    def Valid():
        running_loss = .0
        
        model.eval()
        
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                
                optimizer.zero_grad()
                preds = model(inputs)
                loss = criterion(preds,labels)
                running_loss += loss
                
            valid_loss = running_loss/len(valid_loader)
            valid_losses.append(valid_loss.cpu().detach())#.numpy())
            
            print_text = f'valid_loss {valid_loss}'
            # print(print_text)
            
        return valid_loss.cpu().detach().numpy().tolist()
           
    #%% Run EPOCHS loop
    
    epoch_tic = time.time()
    epoch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_tic))
    print(f'Begin Epochs @ {epoch_time}')
    #------------------------------------------------------------------------------
    
    epoch = 0
    valid_res = 1
        
    
    while epoch < epochs and valid_res >= 0.01:
    # for epoch in range(epochs):
        # print('epochs {}/{}'.format(epoch+1,epochs))
        
        # Training
        train_loss = Train()
        # Validation
        valid_loss = Valid()       
        gc.collect()
        
        # ---------------------------------------------------------------------
        if epoch < 2:
            valid_res = 1
        else:
            valid_res = (valid_losses[epoch-1] - valid_losses[epoch])/valid_losses[epoch-1]
            valid_res = valid_res.detach().numpy().tolist()
            
        print(f'epoch {epoch+1} / {epochs} - ' +
              f'train_loss = {round(train_loss,4)}, test_loss = {round(valid_loss,4)}, test_change = {round(valid_res,4)} ')
        epoch += 1
        
    #------------------------------------------------------------------------------
    epoch_toc = time.time()
    epoch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_toc))
    #print(f'END: Determine Kernel @ {ker_time}')
    print(f'Duration [sec]: {round((epoch_toc-epoch_tic),4)} - [min]: {round((epoch_toc-epoch_tic)/60,4)} - [hrs]: {round((epoch_toc-epoch_tic)/60/60,4)} \n')
    print()
    #%% PLOT - See results after training
    
    # Estimate min vald error
    valid_losses_num = []
    for tensor_element in valid_losses:
        valid_losses_num.append(tensor_element.detach().numpy().tolist())
    
    valid_loss_min = round(min(valid_losses_num),4)
    index_min = min(range(len(valid_losses_num)), key=valid_losses_num.__getitem__)
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.array(range(1,epoch+1)), train_losses,label='train_loss')
    plt.plot(np.array(range(1,epoch+1)), valid_losses,label='valid_loss')
    plt.suptitle(f'{loss_text} Loss', y=1.1 )
    plt.text(0,1,f' Min valid error: {valid_loss_min} \n Epoch*: {index_min+1}/{epochs}' , transform=ax.transAxes, va = 'bottom', ha='left', fontsize=8)
    plt.text(1,1,f' BatchSize: {BatchSize} \n Learn.Rate: {learning_rate_exp}\n Hidden dim: {hidden_dim}' , transform=ax.transAxes, va = 'bottom', ha='right', fontsize=8)
    plt.ylim(0, 2.2)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join(folder_figure_save,sub_folder_plots,
                      'Loss_Epoch.png'))
    plt.close()
        
    
    #%% Split data for prediction
    
    # inputs = input_pred
    # target_y = Ys
    
    # # Make predictions -----------------------------------------------------------
    # model.eval()
    # prediction = []
    # batch_size = 1
    # iterations =  int(inputs.shape[0]/batch_size)
    
    # model.cpu()
    
    
    # #idx = 0
    # for i in range(iterations):
    #     preds = model(torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float())
    #     #print(batch_size*i,batch_size*(i+1))
    #     #idx += 1
    #     prediction.append(preds.detach().numpy()[0][0])
        
    #print(idx)
        
    # See predictions ------------------------------------------------------------
    
    
    # fig, ax = plt.subplots(1, 2,figsize=(11,4))
    # ax[0].set_title('predicted one')
    # ax[0].plot(prediction)
    # ax[1].set_title('real one')
    # ax[1].plot(target_y)
    
    # for a in range(len(ax)):
    #     ax[a].grid()
        
    #     Min = min(min(prediction), min(target_y))
    #     Max = max(max(prediction), max(target_y))
        
    # #     ax[a].set_ylim(1.2*Min, 1.2*Max)
        
    # # plt.show()
     
    
    #%% Function - Error
    
    def errors(y_true, y_pred):
        RMSE = ((y_pred - y_true)**2).mean() **.5
        
        SMSE = ((y_pred - y_true)**2).mean() / y_true.var()
        
        MAE = (abs(y_pred - y_true)).mean()
        MAPE = (abs((y_pred - y_true)/y_true)).mean()*100
        
        TRAC = ( np.dot(y_pred.T,y_true)**2 / (np.dot(y_true.T,y_true)*np.dot(y_pred.T,y_pred)) )[0][0]
        # Dustance    
        #DIST = ((y_pred - y_true)**2).sum() **.5
        #DISTN = ((y_pred/y_true - 1)**2).sum() **.5
        return RMSE, SMSE, MAE, MAPE, TRAC
    
    #%% Determine mean: mu and variance Sigma
    '''0: WW, 1: WS, 2: SW, 3: SS '''
    
    '''OLD
    y = np.array(df_ZY[load_Nodes_Y[0]]['Yi'])
    mus = K[2]['K'].dot( np.linalg.inv(K[0]['K']) ).dot(y)
    
    Sigma = K[3]['K'] - K[2]['K'].dot( np.linalg.inv(K[0]['K']) ).dot(K[1]['K'])
    
    sigma_i = np.diagonal(Sigma)**.5
    '''
    
    # Errors
    columns = []
    for IDss in load_IDss:
        columns.append(IDss + f'_{load_Nodes_Y[0]}')
    
    df_error = pd.DataFrame(columns = columns, index = ['RMSE', 'SMSE', 'MAE','MAPE', 'TRAC'])
    df_values = pd.DataFrame(columns = ['T', 'TR', 'P'], index = load_IDss)
    
    print('Plotting Routine')
    print('------------------------------------------------- \n')
    
    for i in df_Xs_list.index.tolist():
        #df_values = pd.DataFrame(columns = ['True', 'True_Reduced', 'Predicted'], index = [0])
        
        # Determine specific load to be predicted
        Xs_list = df_Xs_list[0][i]
        
        # Reshape specific inputs
        if Case_Nr == 0:
            # Case 0
            inputs = Xs_list.reshape(Xs_list.shape[0],Xs_list.shape[1],1)
    
        elif Case_Nr == 1:
            # Case 1
            inputs = Xs_list.reshape(Xs_list.shape[0],int(Xs_list.shape[1]/length_subvec),length_subvec)
        
        elif Case_Nr == 2:
            # Case 2 
            inputs = Xs_list.reshape(Xs_list.shape[0],1,int(Xs_list.shape[1]/length_subvec),length_subvec)
        
        
        # Make Prediction
        model.eval()
        mus_EQ = []
        batch_size = 1
        iterations =  int(inputs.shape[0]/batch_size)
    
        model.cpu()
    
    
        #idx = 0
        for j in range(iterations):
            preds = model(torch.tensor(inputs[batch_size*j:batch_size*(j+1)]).float())
            #print(batch_size*i,batch_size*(i+1))
            #idx += 1
            mus_EQ.append(preds.detach().numpy()[0][0])
        
        mus_EQ = np.array(mus_EQ)
        
        #mus_EQ = model.predict(Xs_list)[0]
        #sigma_iEQ = model.predict(Xs_list)[1]**.5
        
        
     
            
        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots(2, figsize=(20*cm, 15*cm), sharex=True)
        # True acceleration vs. prediction ----------------------------------------
        #plt.figure()
        node_head = load_Nodes_Y[0] # Only one node : 32
        
        # True
        acc = df_ZYs[node_head]['ACCS'][i]
        x_acc = np.arange(0,len(acc))*0.02
        
        ax[0].plot(x_acc, acc,
                  alpha=0.3, linewidth=3, label='True', color = 'tab:blue')
        
        acc_reduced = acc[length_subvec-1:len(acc):length_step_test]
        x_acc_reduced = (np.arange(0,len(acc_reduced)) *length_step_test*0.02) + (length_subvec*0.02)
        
        ax[0].plot(x_acc_reduced, acc_reduced,
                  alpha=0.3, linewidth=2, label='True Red.', color = 'k')
        
        
        
        #SampEn_acc = DamageTools.SampEn(acc, 2, 0.2*np.std(acc))
        # Predict
        mus_temp = mus_EQ
        x_temp = (np.arange(0,len(mus_temp)) *length_step_test*0.02) + (length_subvec*0.02)
        #np.arange(length_subvec*0.02,mus_temp[-1],length_step)*0.02
        
        #sigma_i_temp = sigma_iEQ
        
        ax[0].plot(x_temp, mus_temp,
                 alpha=0.8,linewidth=1, label='Predicted', color = 'tab:orange')
        
        # ax[0].plot(x_acc, acc,
        #           alpha=0.8, linewidth=1, label='True')
        
        #SampEn_mus = DamageTools.SampEn(mus_temp, 2, 0.2*np.std(mus_temp))
        
        #print(SampEn_acc, SampEn_mus)
        # ax[0].plot(x_temp, sigma_i_temp,
        #          alpha=0.8, label='SD')
        
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
                        f' Epochs: {epochs}, BatchSize: {BatchSize}, Learn.rate: {learning_rate_exp} \n' + 
                     f' Input: {len(load_IDs)}, Nodes {load_Nodes_X} \n Output: {len(load_IDss)}, Nodes {load_Nodes_Y}',
                        x=0, y=0.97, ha='left', va='bottom', fontsize=10)
        #model_optimizer
        plt.xlabel('time [s]')
        fig.tight_layout()
        #plt.xlim(2000,3000)
    
    
        #% Error estimation ----------------------------------------------
        
        
        RMSE = []
        SMSE = []
        MAE = []
        MAPE = []
        TRAC = []
        # for i in range(len(mus_temp)):
        y_true = np.array(acc_reduced).reshape(-1,1)        
        y_pred = mus_temp.reshape(-1,1)
        
        RM, SM, MA, MP, TR = errors(y_true, y_pred)
        RMSE.append(RM)
        SMSE.append(SM)
        MAE.append(MA)
        MAPE.append(MP)
        TRAC.append(TR)
              
        df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['RMSE'] = RMSE[-1]
        df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['SMSE'] = SMSE[-1]
        df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['MAE'] = MAE[-1]
        df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['MAPE'] = MAPE[-1]
        df_error[f'{int_to_str3([idx])[0]}_{load_Nodes_Y[0]}']['TRAC'] = TRAC[-1]
            
        #plt.figure()
        
        # Plot errors
        '''
        #ax[1].plot(x_temp, DIST, alpha=1, label='Distance')
        #plt.plot(range(length_subvec,len(acc),length_step), DISTN, alpha=1, label='Distance - Norm')
        ax[1].plot(x_temp, RMSE, alpha=1, label='RMSE')
        #ax[1].plot(x_temp, SMSE, alpha=1, label='SMSE')
        ax[1].plot(x_temp, MAE, alpha=1, label='MAE')
        #ax[1].plot(x_temp, MAPE, alpha=1, label='MAPE')
        ax[1].plot(x_temp, TRAC, alpha=1, label='TRAC')
        
        
        #plt.xlabel('time [s]')
        ax[1].set_ylabel('Measure of error')
        ax[1].grid()
        ax[1].legend()
        '''
        
        # ax[1].fill_between(x_temp, (mus_temp+2*sigma_i_temp).flatten(), (mus_temp-2*sigma_i_temp).flatten(),
        #          alpha=0.5, label='95-CI')
    
        
        ax[1].plot(x_temp, mus_temp, 
                 alpha=0.8, label='Predicted', color='tab:orange')
        
               
        #plt.xlabel('time [s]')
        ax[1].set_ylabel('Standard deviation')
        ax[1].grid()
        ax[1].legend()
        
    
        # if optimize_model == 1:
        #     if model_optimizer.status.find ('onverged') == -1:
        #         model_status = 'Failed'
        #     else:
        #         model_status = 'Converged'
        # else:
        #     model_status = 'Not optimized'
            
        model_status = 'N/A'
        
        ax[1].set_title(f'Opt. Status: {model_status}, Error: RMSE = {round(RMSE[-1],2)}, SMSE = {round(SMSE[-1],2)}, MAE = {round(MAE[-1],2)}, MAPE = {round(MAPE[-1],2)}, TRAC = {round(TRAC[-1],2)}', 
                     x=0, y=0.97, ha='left', va='bottom', fontsize=10)   
        
        plt.savefig(os.path.join(folder_figure_save,sub_folder_plots,
                          f'ACC{int_to_str3([idx])[0]}_l{length_subvec}_step{length_step}_node{load_Nodes_Ys[0]}_time{start_time_name}.png'))
        plt.close()
        
        
        # Save all Values in DF
        df_values['T'][int_to_str3([idx])[0]] = acc
        df_values['TR'][int_to_str3([idx])[0]] = y_true.tolist()
        df_values['P'][int_to_str3([idx])[0]] = y_pred.tolist()
    
    
    
    #%% Save df_error
    #df.to_pickle(output_directory + "/00_Index_Results.pkl") 
    #unpickled_df = pd.read_pickle("./dummy.pkl")
    
    df_error.to_pickle( os.path.join(folder_figure_save,sub_folder_plots, '00_Error.pkl')  )
    df_values.to_pickle( os.path.join(folder_figure_save,sub_folder_plots, '00_Values.pkl')  )
    
    #%% Time - toc
    global_tic_1 = time.time()
    print('End time: %.4f [s]' %(global_tic_1 - global_tic_0 ))
    print('-- [min]:  %.4f [min]' %( (global_tic_1 - global_tic_0) /60))
    print('-- [hrs]:  %.4f [hrs]' %( (global_tic_1 - global_tic_0) /60/60))
    print() 
    
    # Ending write in file
    Transcript.stop()


    return 

#%% Generate loads
if False:
    # Initialize
    Train_data, Test_data = random_str_list(Index_Results, Train_procent = .9)
    
    load_IDs = Train_data
    load_IDss = Test_data
    
    # Save '015' as training 
    if '015' in load_IDs:
        print('Add/Remove 015')
        load_IDs.remove('015')
        
        load_IDss.append('015')
    
    # Create DataFrame and save data
    df_NN_loads = pd.DataFrame([], columns = ['Train', 'Test'], index = [0])
    df_NN_loads['Train'][0] = load_IDs
    df_NN_loads['Test'][0] = load_IDss
    
    df_NN_loads.to_pickle(os.path.join(folder_figure_save, '00_NN_Loads.pkl'))

#%% Input

# Load df with loads
df_NN_loads = pd.read_pickle(os.path.join(folder_figure_save, '00_NN_Loads.pkl'))


# Training data ---------------------------------------------------------------
# Indicator if total time n
#load_IDs = Train_data # 0.015 --> 5
# load_IDs = ['108', '001', '231', '079', '251']
load_IDs = df_NN_loads['Train'][0]

# Training - X                                                                                 
load_Nodes_X = [23] # Indicator of dimension d

# Training - Y
load_Nodes_Y = [42]

# Combine it all
Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y]


# Testing Data ----------------------------------------------------------------


# Indicator if total time m
#load_IDss = Test_data # 20
load_IDss = df_NN_loads['Test'][0]


# Testing - X*  (Same as X)                                                                             
load_Nodes_Xs = load_Nodes_X

# Testing - Y* (Same as Y)
load_Nodes_Ys = load_Nodes_Y  

# Combine it all
Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys]



#Creation of sub-vecors W -----------------------------------------------------
# Length of sub-vectors
length_subvec = 25

# Overlaping parameter (number of new values in sub-vector)
length_step = 5
length_step_test = 1
hidden_dim = 20

W_par=[length_subvec, length_step, length_step_test, hidden_dim]

# Model Optimization Y/N
optimize_model = 1

# Creation of kernel ----------------------------------------------------------
# Scale factor for each sensor

# Create Epoch / Batch / Learning rate ----------------------------------------
Epochs = 5

# Train Batch Size
train_batch = 25

# Learning rate
learning_rate = 1e-5

Hyper_par = [Epochs, train_batch, learning_rate]


if False:
    NNR(W_par, 
        Hyper_par, 
        Train_par, 
        Test_par)

#%% Varying learning rate

# # Create Epoch / Batch / Learning rate ----------------------------------------
# Epochs = 100

# # Train Batch Size
# train_batch = 25

# for learning_rate in np.linspace(1e-5, 1e-4, 4):

#     NNR(W_par, 
#         [Epochs, train_batch, learning_rate], 
#         Train_par, 
#         Test_par)

#%% Varying hidden layer dims

# # Create Epoch / Batch / Learning rate ----------------------------------------
Epochs = 2

# Train Batch Size
train_batch = 25

for hidden_dim in [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50]:

    NNR(W_par, 
        [Epochs, train_batch, learning_rate], 
        Train_par, 
        Test_par)
    
#%% Varying Epochs

# Create Epoch / Batch / Learning rate ----------------------------------------
# Epochs = 5

# # Train Batch Size
# train_batch = 25

# learning_rate = 1e-5

# # for Epochs in list(range(5,100, 5)):

# NNR(W_par, 
#     [Epochs, train_batch, learning_rate], 
#     Train_par, 
#     Test_par)

