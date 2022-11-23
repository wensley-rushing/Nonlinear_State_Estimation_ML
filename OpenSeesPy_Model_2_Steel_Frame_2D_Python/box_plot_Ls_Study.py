# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:21:34 2022

@author: s202277
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import sys

data_directory = r'output_files\18_tests\Ls_study'

# L_parameter_values = [5, 10, 15, 20, 25, 30]

# S_parameter_values = [3,4,5,6,7]

# Diff_Nodes = [22, 32, 42]

L_parameter_values = [5]

S_parameter_values = [3]

Diff_Nodes = [22]

for l in L_parameter_values:
    for s in S_parameter_values:
        
        folder_path = os.path.join(data_directory, f'L{l}_s{s}')
        print(folder_path)
        
        sub_folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            
        for i in range(0,len(Diff_Nodes)):
            for rdirs, dirs, files in os.walk(os.path.join(folder_path, sub_folders[i])):
                    
                    for file in files:
                        if file.endswith("Error.pkl"):
                            df = pd.read_pickle(os.path.join(folder_path, sub_folders[i], file))
                            
                            
