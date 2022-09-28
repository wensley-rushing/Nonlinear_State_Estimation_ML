# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# inizialization - before FOR 

gm_idx = 0
df = pd.DataFrame(columns = ['Ground motion', 'Load factor', 'E - glob', 'Gl Drift', 'Gl Drift - class', 'Element ID', 'E - elem', 'Plastic def. ele.'])

# start for 


# interstorey drift

n_floors = len(time_drift_disp[0]) - 2 # do not count the columns for time and base node 
drift = []

for i in range (0, n_floors):
    drift[i] = abs(time_drift_disp[-1, i+2]) / H1 # residual drift
    
max_drift = max(drift)*100 # residual drift in percentage

if max_drift <= 0.01:
    drift_cl = 'IO'     # immediate occupancy
elif max_drift <= 0.04:
    drift_cl = 'LS'     # life safety
else:
    drift_cl = 'C'      # collapse


    

    



# fill up the dataframe

gm_idx = loadfactor_idx + 1
df.loc[gm_idx-1] = [ground_motion, loadfactor, Energy_G, max_drif, drift_cl, ele_ID, Energy_L, max_plastic_deform ]



# export

df.to_csv(r'el_centro_dataframe.csv')  # export dataframe to cvs