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


df_structure = pd.read_pickle( os.path.join(output_directory, '00_Structure.pkl') )
struc_periods = list(df_structure.Periods[0])

df_eq = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )
Index_Results = pd.read_pickle( os.path.join(output_directory, '00_Index_Results.pkl') )
index_list = Index_Results.index.tolist()

df_HV = pd.read_pickle( os.path.join(output_directory, 'GM_datasets_5_earthquakes.pkl') ) # read the high variance dataset for comparison


df_datasets = pd.DataFrame(columns = ['Train sets', 'Test sets', 'Variance train set', 'Train eq. duration','Tot. duration'])
        
sets = [
        # [15,91,98,154,158,165,176,177,187,210,215,230,240,276,287,291,296,299,], 
        # [1,2,3,11,13,16,17,20,26,30,34,36,47,48,49,63,65,66,69,70,72,73,76,78,82,88,102,109,116,122,126,131,132,135,138,
        #  140,146,152,157,166,193,199,200,211,236,241,242,243,246,248,252,253,254,258,259,262,263,],
        # [7,9,10,15, 24,41,43,61, 79, 81, 85,90,91, 98, 99, 100,101, 105, 111,123,141,143,145,150,154,158,159,
        #  161,165, 176,177,180, 187,188,194,195,201,210,213,214,215,217,220, 221,222,225, 227,230,234,237,239, 240,269,274,
        #  275,276,279,287,290,291,296,299],
        # [210, 37, 150, 42, 145, 43, 41, 143, 40, 214] 
        # [103, 165, 57, 43, 187, 154, 220, 37, 213, 45]
        [11]
        
        ]
        





for Train_data in sets: 
    # Test_data = index_list
    
    # for k in Train_data:
    #     Test_data.remove(k)
    
    
           
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
            
        
    
        
    # df_datasets.loc[0] = int_to_str3(Train_data), Test_data, var, train_duration, tot_duration
        
        
        
                
    plt.figure()
    plt.scatter(df_eq.loc[:,'Peak acc'], df_eq.loc[:,'Peak T'], marker='x')
    plt.scatter(acc_set, period_set, marker='x')
    plt.title(Train_data[0:5], fontsize = '8', loc = 'left')
    for period in struc_periods:
        plt.axhline(period, linewidth=0.8, linestyle = '--')
        plt.text(4.9, period ,f'{round(period,2)}', fontsize='small')
    
    plt.xlabel('Amplitude')
    plt.ylabel('Period')
    plt.show()

    
        

    
# df_datasets.to_pickle(output_directory + '/GM_datasets_5_random_earthquakes.pkl')


