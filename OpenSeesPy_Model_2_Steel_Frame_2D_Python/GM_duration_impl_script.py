# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:52:24 2022

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random


output_directory = r'output_files'

df_eq = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )
df_datasets = pd.read_pickle(output_directory + '/GM_datasets.pkl')


duration = np.zeros([df_datasets.shape[0],1])

GM_labels = np.zeros([df_datasets.shape[0],1])

for index in range(0, df_datasets.shape[0]):
    
    GM_labels[index] = []
    
    load_IDs = df_datasets.loc[index, 'Train sets']
    for i in load_IDs:
        duration_temp = df_eq.loc[i, 'Input time'][-1]
        duration[index] = duration[index] + duration_temp 
        GM_labels[index].append(df_eq.loc[i, 'Ground motion'])