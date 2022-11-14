# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:27:05 2022

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random


data_directory = 'output_files\Testing\Box_plots'

errors = ['SMSE', 'TRAC']


test_vec = [
            'Test 1\n5 random GMs\nL=25 s=5\ntrain node [23]',
            'Test 2\n5 random GMs\nL=10 s=3\ntrain node [23]',
            'Test 3\n5 high var GMs\nL=25 s=5\ntrain node [23]',
            'Test 4\n5 high var GMs\nL=10 s=3\ntrain node [23]',
            'Test 5\n20 high var GMs\nL=25 s=5\ntrain node [23]',
            'Test 6\n20 high var GMs\nL=10 s=3\ntrain node [23]',
            'Test 7\n20 high var GMs\nL=25 s=5\ntrain node [33]',
            ]

n_test = len(test_vec)

nodes = [22]

for i in nodes: 
    
    df_TRAC = pd.DataFrame(columns=[])
    df_SMSE = pd.DataFrame(columns=[])
    
    j = 0
    
    for test_lab in test_vec: # test label as defined previously
        
        df = pd.DataFrame(columns=[])
        
        test = 'test_' + test_lab[5]   # test index ex. test_1, test_2 .....
        
        file_path = os.path.join(data_directory, f'node_{i}', f'00_Error_{test}.pkl')
        
        df = pd.read_pickle(file_path)
        
        df_TRAC[test_lab] = df.loc['TRAC']
        
        df_SMSE[test_lab] = df.loc['SMSE']
    
        
#%%



for j in range(0, n_test):
    


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))
      
    
       
    
axes[0].boxplot(df_TRAC.iloc[:,range(0, n_test)])
axes[0].set_title('TRAC error')

# plot box plot
axes[1].boxplot(df_SMSE.iloc[:,range(0, n_test)])
axes[1].set_title('SMSE error')

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(n_test)])
    

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(n_test)],
          xticklabels=[df_SMSE.columns][0])
plt.suptitle('Node ' +str(i), fontsize = '14' )

plt.subplots_adjust(left=0.1,
                bottom=0.1,
                right=0.9,
                top=0.9,
                wspace=0.4,
                hspace=0.5)

plt.show()
fig.savefig(os.path.join(data_directory, 'Figures', f'Node_{i}.png'))
plt.close()  
        
        # axes[0].boxplot(df_TRAC.iloc[:,j])
        # axes[0].set_title('TRAC error')
        
        # # plot box plot
        # axes[1].boxplot(df_SMSE.iloc[:,j])
        # axes[1].set_title('SMSE error')
        
        # # adding horizontal grid lines
        # for ax in axes:
        #     ax.yaxis.grid(True)
        #     ax.set_xticks([1])
            
        
        # # add x-tick labels
        # plt.setp(axes, xticks=[1],
        #           xticklabels=[df_SMSE.columns[j]])
        # plt.suptitle('Node ' +str(i), fontsize = '14' )
        
        # plt.subplots_adjust(left=0.1,
        #                 bottom=0.1,
        #                 right=0.9,
        #                 top=0.9,
        #                 wspace=0.4,
        #                 hspace=0.5)
        
        
        
        # j +=1
        


