# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 13:55:12 2023

@author: gabri
"""

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

folder_database = os.path.join('output_files_All')


folder_results = os.path.join('output_files', '18_tests', '03_Report NN - Test_17_V3',
                             'Pred_node42_IN20_OUT20_Time2022-12-14_14-35-13')

database = pd.read_pickle(os.path.join(folder_database, '00_Index_Results.pkl'))
 
Basis = pd.read_pickle(os.path.join(folder_results, '00_Basis.pkl'))

# Error = pd.read_pickle(os.path.join(folder_results, '00_Error.pkl'))

# Values = pd.read_pickle(os.path.join(folder_results, '00_Values.pkl'))

GMs_table_train = pd.DataFrame(columns=['ID', 'GM label', 'Energy', 'Drift', 'Damage'])
IDs_train = Basis.loc[0, 'IN_EQs']


GMs_table_test = pd.DataFrame(columns=['ID', 'GM label', 'Energy', 'Drift', 'Damage'])
IDs_test = Basis.loc[0, 'OUT_EQs']

for ID in IDs_train:
    
    # ID = int_to_str3(ID)[0]
    
    for index in database.index: 
        
        flag = 0
        
        ID_data = int_to_str3([index])[0]
        
        
        
        if ID_data == ID:
            
            
            GMs_table_train.loc[ID] = [index, database.loc[index, 'Ground motion'], database.loc[index, 'E - glob'],
                         database.loc[index, 'Gl Drift'], database.loc[index, 'Gl Drift - class']]
            
            flag = 1
        
        if flag == 1:
            break
                

for ID in IDs_test:
    
    # ID = int_to_str3(ID)[0]
    
    for index in database.index: 
        
        flag = 0
        
        ID_data = int_to_str3([index])[0]
        
        
        
        if ID_data == ID:
            
            
            GMs_table_test.loc[ID] = [index, database.loc[index, 'Ground motion'], database.loc[index, 'E - glob'],
                         database.loc[index, 'Gl Drift'], database.loc[index, 'Gl Drift - class']]
            
            flag = 1
        
        if flag == 1:
            break
        
        
        

        
   