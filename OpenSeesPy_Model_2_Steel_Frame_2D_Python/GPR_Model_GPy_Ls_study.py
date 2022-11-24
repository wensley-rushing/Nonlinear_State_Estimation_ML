# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:59:39 2022

@author: s163761
"""

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

# For GPy
import GPy
GPy.plotting.change_plotting_library('matplotlib')

import pylab as pb

import DamageTools

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

folder_figure_save = r'output_files\new_step\test'

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
# Training data ---------------------------------------------------------------
# Train_data, Test_data = random_str_list(Index_Results, Train_procent = .015)

# # Indicator if total time n
# #load_IDs = Train_data # 0.015 --> 5
# load_IDs = ['182',  '086',  '247',  '149',  '052']#,  '094',  '250',  '138',  
# #             # '156',  '251',  '248',  '073',  '163',  '025',  '258',  '249',  
# #             # '130',  '098',  '040',  '078',  '297',  '012']


# # Training - X                                                                                 
# load_Nodes_X = [40] # Indicator of dimension d

# # Training - Y
# load_Nodes_Y = [20]

# # Combine it all
# Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y]



# # Testing Data ----------------------------------------------------------------

# # Indicator if total time m
# #load_IDss = Test_data # 20
# load_IDss = ['292', '023']
   
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
# W_par=[length_subvec, length_step, length_step_test]


# # Creation of kernel ----------------------------------------------------------
# # Scale factor for each sensor
# sigma2_ks = 1

# # Length sacle
# tau2_ks = 1

# # Error scale for both WW and SS
# sigma2_error = 0
# Ker_par=[sigma2_ks, tau2_ks, sigma2_error]




#%% Gaussian Process Model for Regression
def GPR(sub_folder_Ls_case,
        W_par=[25, 5, 1], #[length_subvec, length_step], 
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
    start_time_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(global_tic_0))
    #print(f'Start time: {start_time}') MOVED
    
    
    
    #%% Understanding inputs
    
    # Creation of Ws
    length_subvec = W_par[0]
    length_step = W_par[1]
    length_step_test = W_par[2]
    #print(f'Sub-vector parameters: Length = {length_subvec}, Step = {length_step}') MOVED
    
    # Creation of kernel (Hyper-parameters)
    sigma2_ks = Ker_par[0]
    tau2_ks = Ker_par[1]
    sigma2_error = Ker_par[2]
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
    # Create Sub-folder for plots (1 folder per node)
    sub_folder_plots = os.path.join( sub_folder_Ls_case, f'Pred_node{load_Nodes_Ys[0]}_IN{len(load_IDs)}_OUT{len(load_IDss)}_Time{start_time_name}')
      
    # Create applicabe sub-folder per each node
    os.mkdir(os.path.join(folder_figure_save, sub_folder_plots))
    
    
    Transcript.start(os.path.join(folder_figure_save, sub_folder_plots, '00_logfile.txt'))
    # print("inside file")
    print(f'Start time: {start_time}')
    print(f'Sub-vector parameters: Length = {length_subvec}, Step = {length_step}')
    print(f'Hyper-parameters: Scale_Factor = {sigma2_ks}, Length_Factor = {tau2_ks}, Error_Factor = {sigma2_error} \n')
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
    
    
                           
    # Testing - X* 
    df_ZXs, df_ZYs = load_to_w(load_IDss, load_Nodes_Xs, load_Nodes_Ys, len_sub_vector=length_subvec , step_size=length_step_test)        
    
    Xs = df_ZXs[df_ZXs.columns[0]]['Z']
    if len(list(df_ZXs.columns)) > 1:
        for Node in list(df_ZXs.columns)[1:]:
            Xs = np.append(Xs,df_ZXs[Node]['Z'], axis=1)
            
        
    df_Xs_list = pd.DataFrame(columns = [0], index = list(range(0,len(load_IDss))))
    for i in df_Xs_list.index.tolist():
        df_Xs_list[0][i] = np.array(df_ZXs[df_ZXs.columns[0]]['Z_list'][i])
        
        if len(list(df_ZXs.columns)) > 1:
            for Node in list(df_ZXs.columns)[1:]:
                df_Xs_list[0][i] = np.append(df_Xs_list[0][i], df_ZXs[Node]['Z_list'][i], axis=1)
    
        
    Ys = df_ZYs[df_ZYs.columns[0]]['Yi']
    Ys = np.array(Ys).reshape(-1,1)
    
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
    ''' OLD
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
    '''
    
    print('Subvector length:', length_subvec)
    # Based only on X values (Kernel 0 given total_dim=1)
    len_idx = 0
    ker = GPy.kern.RBF(input_dim=length_subvec, variance=1., lengthscale=1., active_dims=list(range((len_idx)*length_subvec,(len_idx+1)*length_subvec,1)))
    
    print('Number of sensors:', len(df_ZX.columns))
    if len(df_ZX.columns) > 1:
        for ker_activate_id in range(1,len(df_ZX.columns)):
            len_idx += 1
            ker += GPy.kern.RBF(input_dim=length_subvec, variance=1., lengthscale=1., active_dims=list(range((len_idx)*length_subvec,(len_idx+1)*length_subvec,1)))
    
    
    # Constrain Kernel
    if False:
        print('Kernel constrain')
        
        for i in [0,1]:
            if len(df_ZX.columns) > 1:
                for i in range(1,len(df_ZX.columns)):
                    ker[f'.*rbf_{i}.variance'].constrain_fixed()
                    ker[f'.*rbf_{i}.lengthscale'].constrain_fixed()
    
    
    #print(ker)
    
    
    #------------------------------------------------------------------------------
    ker_toc = time.time()
    ker_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ker_toc))
    #print(f'END: Determine Kernel @ {ker_time}')
    print(f'Duration [sec]: {round((ker_toc-ker_tic),4)} - [min]: {round((ker_toc-ker_tic)/60,4)} - [hrs]: {round((ker_toc-ker_tic)/60/60,4)} \n')
    
    #%% Define Model / Optimize Model
    
    model_tic = time.time()
    model_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_tic))
    print(f'Determine Model @ {model_time}')
    #------------------------------------------------------------------------------
    
    model = GPy.models.GPRegression(X,Y,ker)
    
    # Constrain Model
    if False:
        print('Model constrain \n')
        model['.*Gaussian_noise.variance'].constrain_fixed()
    
    
    
    print('Non-optimized model', model, '\n')
    
    
  
    
    if optimize_model == 1:
        
        opt_tic = time.time()
        opt_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(opt_tic))
        print(f'--Optimize Model @ {opt_time}')
        #--------------------------------------------------------------------------
     
        model_optimizer_idx = 0
        max_iters = 1000
        
        # Default optimization: L-BFGS-B (Scipy implementation
        model_optimizer = model.optimize(messages=True, ipython_notebook=False, max_iters=max_iters)
        #print(model_optimizer)
        # status == 'Converged' or 'ErrorABNORMAL_TERMINATION_IN_LNSRCH'
        
        if True:
            if model_optimizer.status.find('onverge') == -1:
                print('Optimize using scg')
                model_optimizer = model.optimize(optimizer='scg', 
                                                  messages=True, ipython_notebook=False, max_iters=max_iters)
                #print(model_optimizer)
                # status == 'converged - relative reduction in objective'
                
            if model_optimizer.status.find('onverge') == -1:
                print('Optimize using lbfgs')
                model_optimizer = model.optimize(optimizer='lbfgs', 
                                                  messages=True, ipython_notebook=False, max_iters=max_iters)
                #print(model_optimizer)
                
            if model_optimizer.status.find('onverge') == -1:
                print('Optimize using tnc')
                model_optimizer = model.optimize(optimizer='tnc', 
                                                  messages=True, ipython_notebook=False, max_iters=max_iters)
                #print(model_optimizer)
            
            # model_optimizer_idx += 1
        print('Optimized model \n', model)
        
        
        
    #------------------------------------------------------------------------------    
    model_toc = time.time()
    model_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_toc))
    
    print(f'Duration [sec]: {round((model_toc-model_tic),4)} - [min]: {round((model_toc-model_tic)/60,4)} - [hrs]: {round((model_toc-model_tic)/60/60,4)} \n')
    
    #%% Plot of Kernel
    if True:
        length_WW = len(X)
        length_ss = len(Xs)
        
        # Citerium if Xs kernel is 'small enough' then plot XXs, otherwise only plot X
        if length_ss < 15000: 
            ker_X = np.append(X,Xs, axis=0)
            plot_fulle_ker = True
        else:
            ker_X = X
            plot_fulle_ker = False
            
        
        ker_var = [ker['.*rbf.variance'][0]]
        ker_lengh_scale = [ker['.*rbf.lengthscale'][0]]
        #for i in [1,2]:
        if len(df_ZX.columns) > 1:
            for i in range(1,len(df_ZX.columns)):
                ker_var.append( ker[f'.*rbf_{i}.variance'][0] )
                ker_lengh_scale.append( ker[f'.*rbf_{i}.lengthscale'][0] )
        
        model_noise = model['.*Gaussian_noise.variance'][0]
        
        # Rounding inputs
        ker_var = list(np.around(np.array(ker_var),2))
        ker_lengh_scale = list(np.around(np.array(ker_lengh_scale),2))
        model_noise = round(model_noise,2)
        
        #---------------------------------------------------------------------------
        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots(1, figsize=(20*cm, 18*cm))
        #plt.figure()
        
        #plt.matshow()
        #plt.colorbar()
        
        plt.imshow(ker.K(ker_X) , cmap = 'autumn' , interpolation = 'nearest' )
        plt.colorbar();
        
        
        
        # Test in figure
        font_size = 10; a_trans = 0.8
        plt.text(length_WW/2, length_WW/2, r'$K(W,W)$', 
                 fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
        
        if plot_fulle_ker == True:
            plt.text(length_WW + length_ss/2, length_WW/2, r'$K(W,W^{*})$', 
                     fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
            plt.text(length_WW/2, length_WW + length_ss/2, r'$K(W^{*},W)$', 
                     fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
            plt.text(length_WW + length_ss/2, length_WW + length_ss/2, r'$K(W^{*},W^{*})$', 
                     fontsize=font_size, color ='black', alpha=a_trans, va='center', ha='center')
            
            # Plot lines
            plt.axvline(x=length_WW, ls='--', linewidth=1, color='black')
            plt.axhline(y=length_WW, ls='--', linewidth=1, color='black')
            
            fig.suptitle( 'Kernel Heat Map - Training and Testing data' )
        else:
            fig.suptitle( 'Kernel Heat Map - Only Training data' )
        
        # General in figure
        
        
        ax.set_title(f' General: $l$ = {length_subvec}, step = {length_step} \n' +
                     f' $\sigma^2_k$ = {ker_var}, $\u03C4^2_k$ = {ker_lengh_scale}, $\sigma^2_\epsilon$ = {model_noise} \n' +
                     f' Input: {len(load_IDs)}, Nodes {load_Nodes_X} \n Output: {len(load_IDss)}, Nodes {load_Nodes_Y}', 
                     x=0, y=1, ha='left', va='bottom', fontsize=7)
        #plt.show()q
        
        plt.savefig(os.path.join(folder_figure_save,sub_folder_plots,
                                 f'KernelOpt_l{length_subvec}_step{length_step}_time{start_time_name}.png'))
        plt.close()
        
    
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
        
        Xs_list = df_Xs_list[0][i]
        
        mus_EQ = model.predict(Xs_list)[0]
        sigma_iEQ = model.predict(Xs_list)[1]**.5
        
        
     
            
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
        
        sigma_i_temp = sigma_iEQ
        
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
                     f' $\sigma^2_k$ = {ker_var}, $\u03C4^2_k$ = {ker_lengh_scale}, $\sigma^2_\epsilon$ = {model_noise} \n' +
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
        
        ax[1].fill_between(x_temp, (2*sigma_i_temp).flatten(), (-2*sigma_i_temp).flatten(),
                 alpha=0.8, label=u'\u00B1 2 STD', color='moccasin')
        
        ax[1].plot(x_temp, mus_temp, 
                 alpha=0.8, label='Predicted', color='tab:orange')
        
               
        #plt.xlabel('time [s]')
        ax[1].set_ylabel('Standard deviation')
        ax[1].grid()
        ax[1].legend()
        
        
        if optimize_model == 1:
            if model_optimizer.status.find ('onverged') == -1:
                model_status = 'Failed'
            else:
                model_status = 'Converged'
        else:
            model_status = 'Not optimized'
        
        ax[1].set_title(f'Opt. Status: {model_status}, Error: RMSE = {round(RMSE[-1],2)}, SMSE = {round(SMSE[-1],2)}, MAE = {round(MAE[-1],2)}, MAPE = {round(MAPE[-1],2)}, TRAC = {round(TRAC[-1],2)}', 
                     x=0, y=0.97, ha='left', va='bottom', fontsize=10)   
        
        plt.savefig(os.path.join(folder_figure_save,sub_folder_plots,
                          f'ACC{int_to_str3([idx])[0]}_l{length_subvec}_step{length_step}_node{load_Nodes_Ys[0]}_time{start_time_name}.png'))
        plt.close()
        
        
        # Save all Values in DF
        df_values['T'][int_to_str3([idx])[0]] = acc
        df_values['TR'][int_to_str3([idx])[0]] = y_true.tolist()
        df_values['P'][int_to_str3([idx])[0]] = y_pred.tolist()
    
    #%% OLD (Shift in time for large number of outputs...)
    if False: 
        mus = model.predict(Xs)[0]
        sigma_i = model.predict(Xs)[1]**.5
        
        #% Plot Predictions and errors
        if True:
            
            columns = []
            for IDss in load_IDss:
                columns.append(IDss + f'_{load_Nodes_Y[0]}')
            
            df_error = pd.DataFrame(columns = columns, index = ['RMSE', 'SMSE', 'MAE','MAPE', 'TRAC'])
            
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
                sigma_i_temp = sigma_i[temp:temp+len(x_temp)]
                
                ax[0].plot(x_temp, mus_temp, 
                         alpha=0.8, label='Predicted')
                
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
                             f' $\sigma^2_k$ = {ker_var}, $\u03C4^2_k$ = {ker_lengh_scale}, $\sigma^2_\epsilon$ = {model_noise} \n' +
                             f' Input: {len(load_IDs)}, Nodes {load_Nodes_X} \n Output: {len(load_IDss)}, Nodes {load_Nodes_Y}', 
                                x=0, y=0.97, ha='left', va='bottom', fontsize=10)
                model_optimizer
                plt.xlabel('time [s]')
                fig.tight_layout()
                #plt.xlim(2000,3000)
            
            
                #% Error estimation ----------------------------------------------
                
                
                RMSE = []
                SMSE = []
                MAE = []
                MAPE = []
                TRAC = []
                for i in range(len(mus_temp)):
                    y_true = np.array(acc[length_subvec:len(acc):length_step][:i]).reshape(-1,1)
                    y_pred = mus_temp[:i].reshape(-1,1)
                    
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
                
                ax[1].fill_between(x_temp, (2*sigma_i_temp).flatten(), (-2*sigma_i_temp).flatten(),
                         alpha=0.8, label=u'\u00B1 2 STD', color='moccasin')
                
                ax[1].plot(x_temp, mus_temp, 
                         alpha=0.8, label='Predicted', color='tab:orange')
                
                       
                #plt.xlabel('time [s]')
                ax[1].set_ylabel('Standard deviation')
                ax[1].grid()
                ax[1].legend()
                
                if model_optimizer.status.find('onverge') == -1:
                    model_status = 'Failed'
                else:
                    model_status = 'Converged'
                
                ax[1].set_title(f'Opt. Status: {model_status}         Error: RMSE = {round(RMSE[-1],2)}, SMSE = {round(SMSE[-1],2)}, MAE = {round(MAE[-1],2)}, MAPE = {round(MAPE[-1],2)}, TRAC = {round(TRAC[-1],2)}', 
                             x=0, y=0.97, ha='left', va='bottom', fontsize=10)
                
                
                
                
                
                
                temp += len(x_temp)
                
                plt.savefig(os.path.join(folder_figure_save,sub_folder_plots,
                                         f'Predict_ACC_EQ{int_to_str3([idx])[0]}_l{length_subvec}_step{length_step}_node{load_Nodes_Ys[0]}_time{start_time_name}.png'))
                #plt.close()
        
        #return df_error
    
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
#%% Input
# Training data ---------------------------------------------------------------
Train_data, Test_data = random_str_list(Index_Results, Train_procent = .015)

# Indicator if total time n
#load_IDs = Train_data # 0.015 --> 5
load_IDs = ['108', '001', '231', '079', '251']
# load_IDs = ['052']

# Training - X                                                                                 
load_Nodes_X = [23] # Indicator of dimension d

# Training - Y
load_Nodes_Y = [42]

# Combine it all
Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y]



# Testing Data ----------------------------------------------------------------





# Indicator if total time m
#load_IDss = Test_data # 20
load_IDss = ['012', '277']
#load_IDss = ['023']  
# load_IDss = int_to_str3(Index_Results.index.tolist())
# for i in load_IDs:
#     load_IDss.remove(i)

# Testing - X*  (Same as X)                                                                             
load_Nodes_Xs = load_Nodes_X

# Testing - Y* (Same as Y)
load_Nodes_Ys = load_Nodes_Y  

# Combine it all
Test_par=[load_IDss, load_Nodes_Xs, load_Nodes_Ys]



#Creation of sub-vecors W -----------------------------------------------------
# Length of sub-vectors
length_subvec = 100

# Overlaping parameter (number of new values in sub-vector)
length_step = 5
length_step_test = 1

W_par=[length_subvec, length_step, length_step_test]

# Model Optimization Y/N
optimize_model = 1

# Creation of kernel ----------------------------------------------------------
# Scale factor for each sensor
sigma2_ks = 1

# Length sacle
tau2_ks = 1

# Error scale for both WW and SS
sigma2_error = 0
Ker_par=[sigma2_ks, tau2_ks, sigma2_error]


if False:
    GPR(W_par, 
                Ker_par, 
                Train_par, 
                Test_par)

#%% RUN Analysis

#------------------------------------------------------------------------------ 
# Diff_Nodes = [20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]

# #for load_Nodes_X_el in [23]:
# for load_Nodes_Y_el in Diff_Nodes[8:]: # Pred_Node

#     load_Nodes_X = [23]# [load_Nodes_X_el]
#     load_Nodes_Y = [load_Nodes_Y_el]
#     print(load_Nodes_X, load_Nodes_Y)
    
#     GPR(W_par=[length_subvec, length_step, length_step_test], 
#             Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
#             Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
#             Test_par=[load_IDss, load_Nodes_X, load_Nodes_Y])
   

#%%  Different datasets and different nodes to predict

# df_datasets = pd.read_pickle(folder_structure + '/GM_datasets_5_earthquakes.pkl')
# df_datasets = pd.read_pickle(folder_structure + '/GM_datasets_20_random_earthquakes.pkl')

# df_datasets = pd.read_pickle(folder_structure + '/GM_datasets_duration_impl.pkl')



# for i in range(df_datasets.shape[0]):
#     load_IDs = int_to_str3(df_datasets.loc[i, 'Train sets'])
#     load_IDss = int_to_str3(df_datasets.loc[i, 'Test sets'])

#     Diff_Nodes = [22, 32, 42]
    
#     #for load_Nodes_X_el in [23]:
#     for j  in Diff_Nodes:  
    
#         load_Nodes_X = [23]# [load_Nodes_X_el]
#         load_Nodes_Y = [j]
#         print(load_Nodes_X, load_Nodes_Y)
        
#         GPR(W_par=[length_subvec, length_step, length_step_test], 
#                 Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
#                 Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
#                 Test_par=[load_IDss, load_Nodes_X, load_Nodes_Y])

#%%  One dataset and different nodes to predict

# Analysis: 5 Random Earthquakes (same used for coloured matrix and boxplots)
#           Train on node 23, predict 22, 32, 42


Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
index_list = Index_Results.index.tolist()

load_IDs = ['052', '086', '149', '182', '247']
load_IDss = int_to_str3(index_list)

for k in load_IDs:
    load_IDss.remove(k)

# L_parameter_values = [5, 10, 15, 20, 25, 30]
L_parameter_values = [35,40,45,50,70]
S_parameter_values = [5]

Diff_Nodes = [22, 32, 42]

for length_subvec in L_parameter_values:
    for length_step in S_parameter_values:
        
        # Create Sub-folder for L, s values (1 folder per case)
        sub_folder_Ls_case = f'L{length_subvec}_s{length_step}'
        
        os.mkdir(os.path.join(folder_figure_save, sub_folder_Ls_case))
        
        
        #for load_Nodes_X_el in [23]:
        for i in Diff_Nodes:  
        
            load_Nodes_X = [23] # [load_Nodes_X_el]
            load_Nodes_Y = [i]
            
            GPR(sub_folder_Ls_case,
                W_par=[length_subvec, length_step], 
                    Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
                    Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
                    Test_par=[load_IDss, load_Nodes_X, load_Nodes_Y])
            
#%% Linear / non-linear study
# import random

# df_datasets = pd.read_pickle(folder_structure + '/00_EQ_List.pkl')    
# df_datasets = pd.read_pickle(folder_structure + '/00_EQ_List_01.pkl')

# train_LN = 'L'
# test_LN = 'N'

# if train_LN =='L':
#     # load_IDs = int_to_str3(random.sample(df_datasets[23]['L'], k=10))
#     load_IDs = int_to_str3(df_datasets[23]['L'])
# elif train_LN =='N':
#     # load_IDs = int_to_str3(random.sample(df_datasets[23]['N'], k=10))
#     load_IDs = int_to_str3(df_datasets[23]['N'])
    

#%%
        
# Diff_Nodes = [22, 32, 42]

# #for load_Nodes_X_el in [23]:
# for i in Diff_Nodes:  
    
#     if test_LN =='L':
#         load_IDss = []
#         load_IDss = int_to_str3(df_datasets[i]['L'])
#     elif test_LN =='N':
#         load_IDss = []
#         load_IDss = int_to_str3(df_datasets[i]['N'])
    
#     load_Nodes_X = [23] # [load_Nodes_X_el]
#     load_Nodes_Y = [i]
    
#     GPR(W_par=[length_subvec, length_step, length_step_test], 
#             Ker_par=[sigma2_ks, tau2_ks, sigma2_error], 
#             Train_par=[load_IDs, load_Nodes_X, load_Nodes_Y], 
#             Test_par=[load_IDss, load_Nodes_X, load_Nodes_Y])

    
    
sys.exit()


