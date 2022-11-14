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



def random_set_loads(Index_Results, train_elements):
    # Remove all non valid alanysis
    Index_Results.drop(Index_Results['OK=0'][Index_Results['OK=0']!=0],axis=1, inplace=True)
    
    index_list = Index_Results.index.tolist()
    
    train_list = random.sample(index_list, train_elements)
    test_list = index_list
    for i in train_list:
        test_list.remove(i)
    
    return train_list, test_list

output_directory = 'output_files'

df_datasets = pd.DataFrame(columns = ['Train sets', 'Test sets', 'Variance train set', 'Train eq. duration','Tot. duration'])

def best_set(n_set):
    
    var_set = [0, 0]
    
    for i in range(n_set):
        
        Train_data_temp, Test_data_temp = random_set_loads(Index_Results, set_dim)
        
        tot_duration = 0
        train_duration = []
        acc_set_temp = []
        period_set_temp = []
        
        print('\n')    
        print(f'Set {i}:', Train_data_temp)
        
        for j in Train_data_temp:
            for k in df_eq.index:
                if j == k:
                    acc_set_temp.append(df_eq.loc[j, 'Peak acc'])
                    period_set_temp.append(df_eq.loc[j, 'Peak T'])
                    train_duration.append(df_eq.loc[j, 'Input time'][-1])
                    tot_duration = tot_duration + df_eq.loc[j, 'Input time'][-1]
                    
                    
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
            
    return Train_data, Test_data, var_set, acc_set, period_set, train_duration, tot_duration

df_eq = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )
Index_Results = pd.read_pickle( os.path.join(output_directory, '00_Index_Results.pkl') )


var_df = [df_eq.var()[1], df_eq.var()[2]] # list: acceleration variance, period variance

print(f'Dataframe variance [amplitude, period]:', var_df)




#%% 

n_set = 20  # how many sets for each iteration in the function 'best_set': select the best output set out of 5 sets 
output_sets = 1 # how many sets as output
set_dim = 20 # number of earthquakes for each train dataset




Train_sets = []
Test_sets = []
var_sets = []
duration = 0

for n in range(output_sets):
    
    var_set = [0, 0, 0]

    while var_set[0] < 0.6 or var_set[1] < 0.07 or duration > 840:
        Train_data, Test_data, var_set, acc_set, period_set, train_duration , tot_duration = best_set(n_set)
        
        
    Train_sets.append(Train_data)
    Test_sets.append(Test_data)
    var_sets.append(var_set)
    
    df_datasets.loc[n] = Train_data, Test_data, var_set, train_duration, tot_duration
    
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
    
    
#%% check: earthquakes included both in TRAIN and TEST

for index in range(0, df_datasets.shape[0]):
    for i in df_datasets.loc[index, 'Train sets']:
        for j in df_datasets.loc[index, 'Test sets']:
            if i == j:
                print('MATCH test and train for index', i)
        

    
df_datasets.to_pickle(output_directory + '/GM_datasets_duration_impl.pkl')


