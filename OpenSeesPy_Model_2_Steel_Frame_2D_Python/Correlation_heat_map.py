# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:19:08 2022

@author: gabri

Resources:
    
    Theory:
    https://en.wikipedia.org/wiki/Convolution
    https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
    
    Normalization:
    https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import math
from scipy import signal

def load_acc(load_IDs, load_Nodes_X):
    load_Nodes_X_id = []
    for i in range(len(load_Nodes_X)):
        load_Nodes_X_id.append( struc_nodes.index(load_Nodes_X[i]) )
        
    # load_Nodes_Y_id = []
    # for i in range(len(load_Nodes_Y)):
    #     load_Nodes_Y_id.append( struc_nodes.index(load_Nodes_Y[i]) )
        
    
    # Create Overall Dataframe X
    columns = np.array(load_Nodes_X) # k ---> (p)
    #index = np.array(load_IDs) # i
    df_ZX = pd.DataFrame(columns = columns , index = ['ACCS', 'Z', 'Z_list', 'Time'])
    for head in df_ZX.columns:
        df_ZX[head]['ACCS'] = []
        df_ZX[head]['Z'] = []
        df_ZX[head]['Z_list'] = []
        df_ZX[head]['Time'] = []
        
        
    # Create Overall Dataframe Y
    # columns = np.array(load_Nodes_Y) # k ---> (p)
    #index = np.array(load_IDs) # i
    # df_ZY = pd.DataFrame(columns = columns , index = ['ACCS', 'Yi','Yi_list','Time_subvec'])
    # for head in df_ZY.columns:
    #     df_ZY[head]['ACCS'] = []
    #     df_ZY[head]['Yi'] = []
    #     df_ZY[head]['Yi_list'] = []
    #     df_ZY[head]['Time_subvec'] = []
    
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
                    df_ZX[load_Nodes_X[j]]['Time'].append( time )
                    
                    
                    # Create sum vector from accelerations in node
                    
                #     # length of sub-vector 1 <= L <= len (x) 
                #     l = len_sub_vector 
    
                #     # 1 <= step <=  L-1  --!! IF STEP >= L NO OVERLAP !!--
                #     move_step = step_size 
                #     #(if step = 0 then the vector never changes...)
                #     #(if step = 1 then the vector takes 1 step [1 new value + spits out 1 value] )
                #     #(if step = L then the vector takes L steps NO OVERLAP [L new values + spits out L values] )
                #     # Range of sub-vectors z_ik (i = last element in sub-vector)
                #     Zi = []
                #     for i in range(l,len(accs)+1, move_step):
                #         #print(f'i={i}--')
                #         t = []
                #         z = []
                #         # Range of sub-vector elements in z_ik
                #         for idx in range(i-l,i):
                #             t.append(time[idx])
                #             z.append(accs[idx])
                            
                #         df_ZX[load_Nodes_X[j]]['Z'].append( z )
                #         df_ZX[load_Nodes_X[j]]['Time_subvec'].append( t )
                        
                #         Zi.append(z)
                #     df_ZX[load_Nodes_X[j]]['Z_list'].append( Zi )
                #         #print(z)
                        
                # # Load Accelerations in nodes Y
                # for j in range(len(load_Nodes_Y)):
                #     # time = time_Accs[:,0]
                #     accs = time_Accs[:,load_Nodes_Y_id[j]+1].tolist()
                #     time = np.arange(0,len(accs))*0.02
                #     #accs = list(range(100,1100,100))
                    
                #     df_ZY[load_Nodes_Y[j]]['ACCS'].append( accs )
                    
                #     df_ZY[load_Nodes_Y[j]]['Yi'].extend( accs[l-1::move_step] )
                #     df_ZY[load_Nodes_Y[j]]['Time_subvec'].extend( time[l-1::move_step] )
                #     df_ZY[load_Nodes_Y[j]]['Yi_list'].append( accs[l-1::move_step] )
                
    return df_ZX   

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

def sort_df_lag(df_lag):
    lags_plot = []
    
    for i in df_lag.columns.tolist():
        if len(i) < 6:
            lags_plot.append( int(i[4]) )
        elif len(i) == 6:
            lags_plot.append( int(i[4:6]) )
        else:
            lags_plot.append( int(i[4:7]) )
    
    lags_plot.sort(reverse=True)
    
    lags_plot_str = []
    
    for i in lags_plot: 
        lags_plot_str.append (str(f'Lag {i}'))
    
    df_lag = df_lag[lags_plot_str]
    return df_lag
 
#%% Folders structure

folder_accs = r'output_files\ACCS'

folder_output = r'output_files\correlation'

folder_structure = r'output_files'




#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]

struc_nodes = Structure.Nodes[0]

GM_indxs = np.arange(0,2)

print_text = True
plot_bar = False
plot_heat = True

hm_nolag = pd.DataFrame( index = np.sort(struc_nodes)[::-1], columns = struc_nodes)

hm_lag = pd.DataFrame( index = np.sort(struc_nodes)[::-1], columns = struc_nodes)



# for node_sensor in np.sort(struc_nodes)[::-1]:
for node_sensor in [30]:
    # for node2 in struc_nodes:
        for node2 in [10]:
        
            
            df_lag = pd.DataFrame( index = ['N. GMs', 'Conf w/o Lag', 'Conf w/ Lag'], columns = ['All'])
            df_lag.loc['N. GMs', 'All'] = 0
            df_lag.loc['Conf w/o Lag', 'All'] = 0
            df_lag.loc['Conf w/ Lag', 'All'] = 0
            
            # df_lag = pd.DataFrame( index = ['N. GMs', 'Conf w/o Lag', 'Conf w/ Lag'])
            
            corr_nolag = []
            corr_lag = []
            
            for gm_index in GM_indxs:
                
                gm = int_to_str3([gm_index])  
                
                print('\n\n' + gm[0] + '\n\n')
                
                df_ZX_sens = load_acc(gm, struc_nodes)
                
                acc_sens = df_ZX_sens[node_sensor]['ACCS'][0]
                norm = np.linalg.norm(acc_sens)
                acc_sens = acc_sens / norm
                
                time_sens = df_ZX_sens[node_sensor]['Time'][0]
                
                acc_pred = df_ZX_sens[node2]['ACCS'][0]
                norm = np.linalg.norm(acc_pred)
                acc_pred = acc_pred / norm
                time_pred = df_ZX_sens[node2]['Time'][0]
                
                correlation_scalar = np.correlate(acc_sens, acc_pred)
                
                correlation_full = np.correlate(acc_sens, acc_pred, mode = 'full')
                
                lags = signal.correlation_lags(len(acc_sens), len(acc_pred), mode="full")
                
                max_corr = max(correlation_full)
                
                lag = lags[np.argmax((correlation_full))]
                
                stringer = str(f'Lag {lag}')
                
                if lag != 0 and print_text:       
                    
                    print('\n\nSignal must be shifted - GM:' + str(gm_index) + ' Sensors = ' + str(node_sensor) +  
                          ' Prediction = ' + str(node2) + 
                          ':\n [lag, correlation]' + f'[0, {correlation_scalar}] - max correlation for [{lag}, {max_corr}]' )
                    
                
                if stringer not in df_lag:                     
                    df_lag = df_lag.reindex(columns = df_lag.columns.tolist() + [stringer])
                    df_lag.loc['N. GMs', stringer] = 0
                    df_lag.loc['Conf w/o Lag', stringer] = 0
                    df_lag.loc['Conf w/ Lag', stringer] = 0
            
                df_lag.loc['N. GMs', stringer] = df_lag[stringer]['N. GMs'] + 1 
                df_lag.loc['Conf w/o Lag', stringer] = df_lag[stringer]['Conf w/o Lag'] + correlation_scalar
                df_lag.loc['Conf w/ Lag', stringer] = df_lag[stringer]['Conf w/ Lag'] + (max_corr)
                
                df_lag.loc['N. GMs', 'All'] = df_lag['All']['N. GMs'] + 1 
                df_lag.loc['Conf w/o Lag', 'All'] = df_lag['All']['Conf w/o Lag'] + correlation_scalar
                df_lag.loc['Conf w/ Lag', 'All'] = df_lag['All']['Conf w/ Lag'] + (max_corr)
                
                
                corr_nolag.append(correlation_scalar)
                corr_lag.append(max_corr)
            
                
            hm_nolag.loc[node_sensor, node2] = np.mean(corr_nolag)
            hm_lag.loc[node_sensor, node2] = np.mean(corr_lag)
            # print(f'Row {node_sensor}, column {node2}')
            # print(f'Previous study: \nw/o lag = ')


        #%% plot  
        # if plot_bar:
            
        #     # get lags included in the plot
            
        #     df_lag = sort_df_lag(df_lag)
            
        #     plt.figure(figsize=(10, 10))
            
        #     x_offset = np.arange(0, len(df_lag.columns.tolist()) ,1)
        #     # y_lab = np.arange(1, df_lag.max(axis=1)[0]+1,20)
        #     labels = []

            
            
        #     for i in range(0, len(df_lag.columns.tolist())):
                
        #         string = df_lag.columns.tolist()[i]
        #         err_no_lag = np.round(df_lag.loc['Conf w/o Lag', string] / df_lag.loc['N. GMs', string],2)
        #         err_with_lag = np.round(df_lag.loc['Conf w/ Lag', string] / df_lag.loc['N. GMs', string],2)
        #         plt.bar(x_offset[i], df_lag.loc['N. GMs', string], width=0.8, color = 'g' )
        #         plt.text(x_offset[i]-0.1, df_lag.loc['N. GMs', string] + 0.5, s = int(df_lag.loc['N. GMs', string]) )
        #         labels.append( str(string + '\n\nCorrelation:' +f'\nw/o lag: {err_no_lag}' + f'\nw/ lag: {err_with_lag}'))
            
        #     plt.xticks(x_offset, labels)
            
        #     plt.title(f'Sensor in node {node_sensor}, compared to node {node2}')
            
        #     # plt.yticks(y_lab)
        #     plt.grid(axis = 'y')
            
            
            
            
        #     plt.ylabel('N. GMs')
                
                
                
            
                    

if plot_heat:
    
        
    
    plt.figure(figsize =(10, 7))
    plt.pcolor(hm_nolag.values.tolist())
    plt.yticks(np.arange(0.5, len(hm_nolag.index), 1), hm_nolag.index)
    plt.xticks(np.arange(0.5, len(hm_nolag.columns), 1), hm_nolag.columns)
    # plt.title('Train set: 5 random GMs \nTest set: 296 GMs', loc='Left', fontsize = 9)
    plt.colorbar(label = 'Mean Correlation')
    
    # Lines
    # plt.axvline(x=4, ls='--', linewidth=1, color='black')
    # plt.axvline(x=8, ls='--', linewidth=1, color='black')
    
    # plt.axhline(y=4, ls='--', linewidth=1, color='black')
    # plt.axhline(y=8, ls='--', linewidth=1, color='black')
    
    # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    
    
    # # Loop over data dimensions and create text annotations.
    for i in range(len(hm_nolag.index)):
        for j in range(len(hm_nolag.columns)):
            if round(hm_lag.iloc[i,j],2) >  0.8: #0.8:
                text = plt.text(j+0.5, i+0.5, round(hm_nolag.iloc[i,j],2),
                                ha="center", va="center", color="k", fontsize='small')#, transform = ax.transAxes)
            else:
                text = plt.text(j+0.5, i+0.5, round(hm_nolag.iloc[i,j],2),
                                ha="center", va="center", color="w", fontsize='small')#, transform = ax.transAxes)
            
    
    plt.title('Nodes signal correlation: without lag')
    plt.ylabel('Node')
    plt.xlabel('Node')
    plt.show()
    
    # plt.savefig(os.path.join(folder_data, f'ErrorMap_{error_text}_train{train}_IN{num_in}_OUT{num_out}.png'))
    plt.close()           
    
    
    plt.figure(figsize =(10, 7))
    plt.pcolor(hm_lag.values.tolist())
    plt.yticks(np.arange(0.5, len(hm_lag.index), 1), hm_lag.index)
    plt.xticks(np.arange(0.5, len(hm_lag.columns), 1), hm_lag.columns)
    # plt.title('Train set: 5 random GMs \nTest set: 296 GMs', loc='Left', fontsize = 9)
    plt.colorbar(label = 'Mean Correlation')
    
    # Lines
    # plt.axvline(x=4, ls='--', linewidth=1, color='black')
    # plt.axvline(x=8, ls='--', linewidth=1, color='black')
    
    # plt.axhline(y=4, ls='--', linewidth=1, color='black')
    # plt.axhline(y=8, ls='--', linewidth=1, color='black')
    
    # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    
    
    # # Loop over data dimensions and create text annotations.
    for i in range(len(hm_lag.index)):
        for j in range(len(hm_lag.columns)):
            if round(hm_lag.iloc[i,j],2) >  0.8: #0.8:
                text = plt.text(j+0.5, i+0.5, round(hm_lag.iloc[i,j],2),
                                ha="center", va="center", color="k", fontsize='small')#, transform = ax.transAxes)
            else:
                text = plt.text(j+0.5, i+0.5, round(hm_lag.iloc[i,j],2),
                                ha="center", va="center", color="w", fontsize='small')#, transform = ax.transAxes)
            
    
    plt.title('Nodes signal correlation: with lag')
    plt.ylabel('Node')
    plt.xlabel('Node')
    plt.show()
    
    # plt.savefig(os.path.join(folder_data, f'ErrorMap_{error_text}_train{train}_IN{num_in}_OUT{num_out}.png'))
    plt.close()   
