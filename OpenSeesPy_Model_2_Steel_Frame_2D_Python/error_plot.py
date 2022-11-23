# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:19:07 2022

@author: gabri
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os


#%% Folder structure

folder_accs = r'output_files\ACCS'

folder_output = r'output_files\error_plot'

folder_structure = r'output_files'

folder_figure_save = r'output_files\error_plot\figures'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = Structure.Nodes[0]

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
    df_ZX = pd.DataFrame(columns = columns , index = ['ACCS', 'Z', 'Z_list', 'Time_subvec'])
    for head in df_ZX.columns:
        df_ZX[head]['ACCS'] = []
        df_ZX[head]['Z'] = []
        df_ZX[head]['Z_list'] = []
        df_ZX[head]['Time_subvec'] = []
        
        
    # Create Overall Dataframe Y
    columns = np.array(load_Nodes_Y) # k ---> (p)
    #index = np.array(load_IDs) # i
    df_ZY = pd.DataFrame(columns = columns , index = ['ACCS', 'Yi','Yi_list','Time_subvec'])
    for head in df_ZY.columns:
        df_ZY[head]['ACCS'] = []
        df_ZY[head]['Yi'] = []
        df_ZY[head]['Yi_list'] = []
        df_ZY[head]['Time_subvec'] = []
    
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
                    
                    # time = time_Accs[:,0]
                    accs = time_Accs[:,load_Nodes_X_id[j]+1].tolist()
                    time = np.arange(0,len(accs))*0.02
                    
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
                        t = []
                        z = []
                        # Range of sub-vector elements in z_ik
                        for idx in range(i-l,i):
                            t.append(time[idx])
                            z.append(accs[idx])
                            
                        df_ZX[load_Nodes_X[j]]['Z'].append( z )
                        df_ZX[load_Nodes_X[j]]['Time_subvec'].append( t )
                        
                        Zi.append(z)
                    df_ZX[load_Nodes_X[j]]['Z_list'].append( Zi )
                        #print(z)
                        
                # Load Accelerations in nodes Y
                for j in range(len(load_Nodes_Y)):
                    # time = time_Accs[:,0]
                    accs = time_Accs[:,load_Nodes_Y_id[j]+1].tolist()
                    time = np.arange(0,len(accs))*0.02
                    #accs = list(range(100,1100,100))
                    
                    df_ZY[load_Nodes_Y[j]]['ACCS'].append( accs )
                    
                    df_ZY[load_Nodes_Y[j]]['Yi'].extend( accs[l-1::move_step] )
                    df_ZY[load_Nodes_Y[j]]['Time_subvec'].extend( time[l-1::move_step] )
                    df_ZY[load_Nodes_Y[j]]['Yi_list'].append( accs[l-1::move_step] )
                
    return df_ZX, df_ZY     

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

L_parameter_values = [1, 10, 20, 25, 40]

S_parameter_values = [1,2,3,4,5]

nodes =[42]
GM_indxs = np.arange(0,301)

# nodes = [42]
# GM_indxs = [0,1,2,3]

save_results = True

for node in nodes:
    
    
    if save_results: 
        # Create subfolder
    
        node_folder = os.path.join(folder_output, f'node_{node}')
            
        os.mkdir(node_folder)
    
    # Start analysis
    
    df = pd.DataFrame(columns = ['GM index', 'RMSE', 'Tails', 'Tails relevance'])
    
    
    # Set parameters
    
    for length_subvec in L_parameter_values:
        for length_step in S_parameter_values:
    
            if save_results: 
                # Create subfolder
                
                Ls_case_folder = os.path.join(node_folder, f'L{length_subvec}_s{length_step}')
                os.mkdir(Ls_case_folder)
            
            
            for gm_index in GM_indxs:
                
                gm = int_to_str3([gm_index])        
        
                
                
                
                
                # Get subvectors for acceleration and time
                
                df_ZX, df_ZY = load_to_w(gm,[node],[],length_subvec,length_step)
                
                # Get entire signal: x_acc is time! 
                
                acc = df_ZX[node]['ACCS'][0]
                x_acc = np.arange(0,len(acc))*0.02
                
                n_subvec = len(df_ZX.loc['Z'].tolist()[0])
                
                # Get reduced signal: it includes only the predicted points: two different methods
                
                y_true_red = np.array(acc[length_subvec -1 :len(acc):length_step]).reshape(-1,1)
                y_true_red_2 = []
                x_red = [] # time vector for reduced y vector
                
                for i in range(0, n_subvec):
                    
                    a = df_ZX.loc['Z'].tolist()[0][i][-1]
                    t = df_ZX.loc['Time_subvec'].tolist()[0][i][-1]
                    
                    y_true_red_2.append(a)  
                    x_red.append(t)
                    
                    
                    if y_true_red_2[i] != y_true_red[i]:
                        print('Error 1 in the script!!!')
                        
                if len(y_true_red_2) != n_subvec or len(x_red) != len(y_true_red_2):
                    print('Error 2 in the script!!!')
                
                
                rmse = []
                err = []
                plot_curve = []
                x_plot = []
                j = 0
                flag = 0
                
                tail_start = []
                tail_end = []
                
                # For cycle: i is the index for the extended acceleration vector; j is used for the reduced acceleration vector
                
                for i in range(0,len(acc)):   
                    
                    if flag == 2:
                        tail_end.append(acc[i])
                    
                    
                    if x_acc[i] == x_red[0]: # if we are at the first value of the reduced vector we can start making estimations and measure
                        flag += 1            # the error
                        
                        
                    if flag == 1:  # measure the error
                        if x_acc[i] == x_red[j]: # Make estimation: the two points match
                            
                            err.append((acc[i] - y_true_red_2[j])**2)
                            plot_curve.append(y_true_red_2[j])
                            j += 1
        
                            if  x_acc[i] == x_red[-1]:  # if we are at the last value of the reduced vector we cannot make any other estimation
                                flag = 2
                            
                        else: # Make estimation: the two points do not match, we need to make linear regression
                            
                            inter = np.interp(x_acc[i] ,[x_red[j-1], x_red[j]], [y_true_red_2[j-1], y_true_red_2[j]]) 
                            err.append((acc[i] - inter)**2)
                            plot_curve.append(inter)
                            
                        x_plot.append(x_acc[i])
                    
                    # Points which have been excluded both from the error calculation and from the reduced vector    
                    
                    if flag == 0: 
                        tail_start.append(acc[i])
                    
                        
                        
                        
                rmse.append(math.sqrt(sum(err)/len(err)))
                
                
                if np.mean(tail_start)>0.25: 
                    tail_start_relevance = 1
                else: 
                    tail_start_relevance = 0
                
                if np.mean(tail_end)>0.25: 
                    tail_end_relevance = 1
                else: 
                    tail_end_relevance = 0
                    
                tail = [[tail_start],[tail_end]]
                tail_relevance = [tail_start_relevance, tail_end_relevance]
                
                df.loc[gm_index] = gm[0], rmse[0], tail, tail_relevance
                
                plt.figure()
                plt.plot(x_acc, acc, label = 'Entire sign')
                # plt.plot(x_red, y_true_red_2, label = 'Py pred', linestyle = '-', alpha = 0.7)
                plt.plot(x_plot, plot_curve, label = 'Reduced sign', alpha = 0.6)
                plt.suptitle(f'Node {node}', y=1.)
                plt.title(f'GM ID: {gm[0]} - RMSE = {round(rmse[0],2)}', loc='Left', fontsize = 9)
                plt.grid()
                plt.legend()
                if save_results: 
                    plt.savefig(os.path.join(Ls_case_folder,'GM' + gm[0] + f'_Node_{node}_L={length_subvec}_s={length_step}.png'))
                plt.close()
            
            if save_results: 
                df.to_pickle(os.path.join(Ls_case_folder, '00_RMSE_Error.pkl'))
                    
        
        




