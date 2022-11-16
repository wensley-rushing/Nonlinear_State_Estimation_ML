# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:30:28 2022

@author: gabri
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:54:45 2022

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


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



plot_best_sets = True
output_directory = 'output_files'



df_eq = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )
Index_Results = pd.read_pickle( os.path.join(output_directory, '00_Index_Results.pkl') )
index_list = Index_Results.index.tolist()

df_HV = pd.read_pickle( os.path.join(output_directory, 'GM_datasets_5_earthquakes.pkl') ) # read the high variance dataset for comparison


df_datasets = pd.DataFrame(columns = ['Train sets', 'Test sets', 'Variance train set', 'Train eq. duration','Tot. duration'])
        
Train_data = [52, 86, 149, 182, 247]
Test_data = index_list

for k in Train_data:
    Test_data.remove(k)


       
tot_duration = 0
train_duration = []
acc_set = []
period_set = []
        

for j in Train_data:
    for k in df_eq.index:
        if j == k:
            acc_set.append(df_eq.loc[j, 'Peak acc'])
            period_set.append(df_eq.loc[j, 'Peak T'])
            train_duration.append(df_eq.loc[j, 'Input time'][-1])
            tot_duration = tot_duration + df_eq.loc[j, 'Input time'][-1]
                    
                    
var = [np.var(acc_set) , np.var(period_set)]
        
    

    
df_datasets.loc[0] = int_to_str3(Train_data), Test_data, var, train_duration, tot_duration
    
    
    
            
plt.figure()
plt.scatter(df_eq.loc[:,'Peak acc'], df_eq.loc[:,'Peak T'], marker='x')
plt.scatter(acc_set, period_set, marker='x')
plt.title(Train_data, fontsize = '8', loc = 'left')
plt.xlabel('Amplitude')
plt.ylabel('Period')
plt.show()

    
        

    
df_datasets.to_pickle(output_directory + '/GM_datasets_5_random_earthquakes.pkl')


