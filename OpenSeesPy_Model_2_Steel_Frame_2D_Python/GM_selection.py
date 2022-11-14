# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:54:45 2022

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random

plot_all_sets = False
plot_best_sets = True
output_directory = 'output_files'


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




def getCurrentMemoryUsage():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())

def random_loads(Index_Results, Train_procent = 0.07):
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
    
    return train_list, test_list

output_directory = 'output_files'

df_datasets = pd.DataFrame(columns = ['Train sets', 'Test sets', 'Variance train set'])

def best_set(n_set):
    
    var_set = [0, 0, 0]
    for i in range(n_set):
        
        Train_data_temp, Test_data_temp = random_loads(Index_Results, Train_procent = set_dim)
        
        acc_set_temp = []
        period_set_temp = []
        
        print('\n')    
        print(f'Set {i}:', Train_data_temp)
        
        for j in Train_data_temp:
            for k in df_eq.index:
                if j == k:
                    acc_set_temp.append(df_eq.loc[j, 'Peak acc'])
                    period_set_temp.append(df_eq.loc[j, 'Peak T'])
                    
                    
        var_temp = [np.var(acc_set_temp) , np.var(period_set_temp)]
        
        print(f'Set {i} variance [amplitude, period]:', var_temp)
        
        if plot_all_sets:        
            plt.figure()
            plt.scatter(df_eq.loc[:,'Peak acc'], df_eq.loc[:,'Peak T'], marker='x')
            plt.scatter(acc_set_temp, period_set_temp, marker='x')
            plt.title('Set n. .%i' %i)
            plt.xlabel('Amplitude')
            plt.ylabel('Period')
        
        if var_temp[0] > var_set[0]:
            var_set = var_temp
            Train_data = Train_data_temp
            Test_data = Test_data_temp
            acc_set = acc_set_temp
            period_set = period_set_temp
            
    return Train_data, Test_data, var_set, acc_set, period_set

df_eq = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )
Index_Results = pd.read_pickle( os.path.join(output_directory, '00_Index_Results.pkl') )


var_df = [df_eq.var()[1], df_eq.var()[2]] # list: acceleration variance, period variance

print(f'Dataframe variance [amplitude, period]:', var_df)




#%% 

n_set = 5  # how many sets for each iteration: select the best output set out of 5 sets 
output_sets = 3 # how many sets as output
set_dim = 0.05 # number of earthquakes  out of the 301 per each dataset: 20 eart = 6.6% of 301, 15 eart =  5.98%




Train_sets = []
Test_sets = []
var_sets = []

for n in range(output_sets):
    
    var_set = [0, 0, 0]

    while var_set[0] < 0.7 or var_set[1] < 0.07:
        Train_data, Test_data, var_set, acc_set, period_set = best_set(n_set)
        
        
    Train_sets.append(Train_data)
    Test_sets.append(Test_data)
    var_sets.append(var_set)
    
    df_datasets.loc[n] = Train_data, Test_data, var_set
    
    print(f'\n Set {n}, final set variance [amplitude, period]:', var_set)
    
    if plot_best_sets:        
        plt.figure()
        plt.scatter(df_eq.loc[:,'Peak acc'], df_eq.loc[:,'Peak T'], marker='x')
        plt.scatter(acc_set, period_set, marker='x')
        plt.suptitle('Set n. %i' %(n+1),fontsize = '12' )
        plt.title(Train_data, fontsize = '8', loc = 'left')
        plt.xlabel('Amplitude')
        plt.ylabel('Period')
        plt.show()
    
# df_datasets.to_pickle(output_directory + '/GM_datasets.pkl')


