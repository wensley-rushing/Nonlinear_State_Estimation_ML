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
import sys

data_directory = 'output_files\Testing\Box_plots'


plot_box = False

test_vec = [
            'Test 1\n5 random GMs\nL=25 s=5\ntrain node [23]',
            'Test 2\n5 random GMs\nL=10 s=3\ntrain node [23]',
            'Test 3\n5 high var GMs\nL=25 s=5\ntrain node [23]',
            'Test 4\n5 high var GMs\nL=10 s=3\ntrain node [23]',
            'Test 5\n20 high var GMs\nL=25 s=5\ntrain node [23]',
            'Test 6\n20 high var GMs\nL=10 s=3\ntrain node [23]',
            'Test 7\n20 high var GMs\nL=25 s=5\ntrain node [33]',
            'Test 8\n20 high var GMs\nL=10 s=3\ntrain node [33]',
            'Test 9\n10 high energy GMs\nL=25 s=5\ntrain node [23]',
            'Test 10\n10 high error GMs\nL=25 s=5\ntrain node [23]'
            ]

n_test = len(test_vec)

nodes = [22]

for i in nodes: 
    
    df_TRAC = pd.DataFrame(columns=[])
    df_SMSE = pd.DataFrame(columns=[])
     
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 20))
    
    j = 0
    
    for test_lab in test_vec: # test label as defined previously
        
        df = pd.DataFrame(columns=[])
        
        if test_lab[6] == '\n':
            test = 'test_' + test_lab[5]   # test index ex. test_1, test_2 .....
        else:
            test = 'test_' + test_lab[5] + test_lab[6] 
            
                    
        file_path = os.path.join(data_directory, f'node_{i}', f'00_Error_{test}.pkl')
        
        df = pd.read_pickle(file_path)
        
        df_TRAC[test_lab] = df.loc['TRAC']
        
        df_SMSE[test_lab] = df.loc['SMSE']
    
        if plot_box:
        
            # TRAC boxplot
            axes[0].boxplot(df.loc['TRAC'].values.reshape(-1,1), widths=0.25, positions=[j])# labels = test_lab)        
            axes[0].set_title('TRAC error',  y=1.05, fontweight="bold", fontsize = '12')
            axes[0].text(x=(j+0.5)/(len(test_vec)) , y=1, s=f"({round(df.loc['TRAC'].values.reshape(-1,1).mean(),2)})", 
                          va='bottom', ha='center', transform = axes[0].transAxes, fontsize = '12')
            
            # SMSE boxplot
            axes[1].boxplot(df.loc['SMSE'].values.reshape(-1,1), widths=0.25, positions=[j]) #labels = test_lab)   
            axes[1].set_title('SMSE error',  y=1.05, fontweight="bold", fontsize = '12')
            axes[1].text(x=(j+0.5)/(len(test_vec)) , y=1, s=f"({round(df.loc['SMSE'].values.reshape(-1,1).mean(),2)})", 
                          va='bottom', ha='center', transform = axes[1].transAxes, fontsize = '12')
           
            # RMSE boxplot
            
            axes[2].boxplot(df.loc['RMSE'].values.reshape(-1,1), widths=0.25, positions=[j]) #labels = test_lab)   
            axes[2].set_title('RMSE error',  y=1.05, fontweight="bold", fontsize = '12')
            axes[2].text(x=(j+0.5)/(len(test_vec)) , y=1, s=f"({round(df.loc['RMSE'].values.reshape(-1,1).mean(),2)})", 
                          va='bottom', ha='center', transform = axes[2].transAxes, fontsize = '12')
            
        
        
           
    
    
            # adding horizontal grid lines
            for ax in axes:
                ax.yaxis.grid(True)
                # ax.set_xticks([y+1 for y in range(n_test)])
                
            
            # add x-tick labels
            plt.setp(axes, xticks=[y for y in range(n_test)],
                      xticklabels=[df_SMSE.columns][0])
            plt.suptitle('Node ' +str(i), y=0.95, fontsize = '20', fontweight="bold" )
            
            plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
                
            plt.show()
            # fig.savefig(os.path.join(data_directory, 'Figures', f'Node_{i}.png'))
            plt.close() 
        
        j += 1 
            


