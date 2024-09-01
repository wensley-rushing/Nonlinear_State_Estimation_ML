# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:28:16 2022

@author: gabri
"""

import opensees.openseespy as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np



from Model_definition_2D_frame import createModel
from gravityAnalysis import runGravityAnalysis

output_directory = 'output_files'

time_topDisp = np.loadtxt(output_directory+'/2_groundmotion_top_disp.out')

Pushover_topDisp_dyn = time_topDisp[:,1]
Pushover_reactions_dyn = np.loadtxt(output_directory+'/2_groundmotion_base_reactions.out')
total_base_reaction_dyn = -np.sum(Pushover_reactions_dyn,axis=1)

Pushover_topDisp = np.loadtxt(output_directory+'/001_Pushover_top_disp.out')
Pushover_reactions = np.loadtxt(output_directory+'/001_Pushover_base_reactions.out')
total_base_reaction = -np.sum(Pushover_reactions,axis=1)


plt.figure()
plt.plot(np.insert(Pushover_topDisp_dyn, 0, 0),np.insert(total_base_reaction_dyn, 0, 0)/1000)   #inserts 0 at the beginning
plt.plot(np.insert(Pushover_topDisp, 0, 0),np.insert(total_base_reaction, 0, 0)/1000)
plt.title('Pushover curve')
plt.xlabel('displacement top (m)')
plt.ylabel('total base shear (kN)')
plt.grid()
plt.show()