# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:41:52 2022

@author: s202277
"""

import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt


output_directory = r'output_files'
df_datasets = pd.read_pickle(output_directory + '/GM_datasets.pkl')


df_datasets_red = pd.DataFrame(columns = ['Train sets', 'Test sets', 'Variance train set'])
df_eq = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )

# for i in df_datasets.loc[0, 'Train sets']:
#     for j in df_datasets.loc[0, 'Test sets']:
#         if i == j:
#             print('MATCH test and train for index', i)

#%%
for i in [0,1,3]:
    
    remove = -1
    
    Train_data_red = df_datasets.loc[i, 'Train sets'][:remove]
    # Test_data_red = df_datasets.loc[i, 'Test sets']
    # Test_data_red.append(df_datasets.loc[i, 'Train sets'][remove])
    
    Test_data_red = []
    
    acc_set_red = []
    period_set_red = []
    
    Test_data_red = list(range(1,301 + 1))
    for j in Train_data_red:
        for k in df_eq.index:
            if j == k:
                acc_set_red.append(df_eq.loc[j, 'Peak acc'])
                period_set_red.append(df_eq.loc[j, 'Peak T'])
                Test_data_red.remove(j)

    var_set_red = [ round(np.var(acc_set_red),3) , round(np.var(period_set_red),3)]
    
    df_datasets_red.loc[i] = Train_data_red, Test_data_red, var_set_red
    
    
    acc_set = []
    period_set = []
    
    for j in df_datasets.loc[i, 'Train sets']:
        for k in df_eq.index:
            if j == k:
                acc_set.append(df_eq.loc[j, 'Peak acc'])
                period_set.append(df_eq.loc[j, 'Peak T'])
                
    var_set = [ round(np.var(acc_set),3) , round(np.var(period_set),3)]
    
    
    
    
    fig = plt.figure(figsize = (10,7))
    plt.subplot(2,1,1)
    plt.scatter(df_eq.loc[:,'Peak acc'], df_eq.loc[:,'Peak T'], marker='x')
    plt.scatter(acc_set, period_set, marker='x')
    plt.suptitle('Set n. %i' %(i+1),fontsize = '14' )
    plt.title('Dataset with ' +str(np.shape(acc_set)[0]) +' earthquakes \nVariance [amplitude, period] =' 
              +str(var_set), fontsize = '10', loc = 'left')
    plt.xlabel('Amplitude')
    plt.ylabel('Period')
    
    plt.subplot(2,1,2)
    plt.scatter(df_eq.loc[:,'Peak acc'], df_eq.loc[:,'Peak T'], marker='x')
    plt.scatter(acc_set_red, period_set_red, marker='x')
    # plt.title(var_set_red, fontsize = '10', loc = 'left')
    # plt.title('Reduced set, \nvariance [amplitude, period] = '+str(var_set_red), fontsize = '10', loc = 'left')
    plt.title('Reduced set with ' +str(np.shape(acc_set)[0]+remove) +' earthquakes \nVariance [amplitude, period] =' 
              +str(var_set_red), fontsize = '10', loc = 'left')
    plt.xlabel('Amplitude')
    plt.ylabel('Period')
    
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    
    plt.savefig(os.path.join(output_directory, 'Figures', 'GM_selection', f'Set_{i}.png'))
    plt.show()
    
    
    df_datasets_red.to_pickle(output_directory + '/GM_datasets_red.pkl')