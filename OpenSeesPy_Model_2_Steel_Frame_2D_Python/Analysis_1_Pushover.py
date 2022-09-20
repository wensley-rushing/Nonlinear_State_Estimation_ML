# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""

#sketch of the structure  with node and element numbering

#
# 20		2021     	  21	

# | |					 | |	
# | |					 | |	
# 1020 				     1121
# | |					 | |	
# | |					 | |	

# 10				      11	

# nodes: 10, 11, 20, 21
# elements: 1020 (column), 1121 (column), 2021 (beam)
#import os
#import sys

#file_dir = os.path.dirname(__file__)
#sys.path.append(file_dir)


import openseespy.opensees as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np



from Model_definition_2D_frame import createModel
from gravityAnalysis import runGravityAnalysis





# turn on/off the plots by setting these to True or False
plot_model = True
plot_defo_gravity = True
plot_defo_Pushover= True






# =============================================================================
# Units
# =============================================================================

m = 1
cm = 0.01*m
mm = 0.001*m

N = 1
kN = 1e3*N

Pa = 1
MPa = 1e6*Pa
GPa = 1e9*Pa

kg = 1

# =============================================================================
# Input parameters
# =============================================================================


H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 1000 *kg 	  #kg		lumped mass at top corner nodes 





# =============================================================================
# # call function to create the model
# =============================================================================

createModel(H1,L1,M)



if plot_model:
    plt.figure()
    opsv.plot_model()
    plt.title('model')
    plt.show()  



# =============================================================================
# Run gravity Analysis
# =============================================================================
runGravityAnalysis()


if plot_defo_gravity:
    plt.figure()
    opsv.plot_defo(sfac = 10000) 
    plt.title('deformed shape - gravity analysis')
    plt.show()  


# wipe analysis objects and set pseudo time to 0
ops.wipeAnalysis()
ops.loadConst()
ops.loadConst('-time', 0.0)
 
    

# =============================================================================
# Pushover analysis
# =============================================================================

# Define Recorders
output_directory = 'output_files'
ops.recorder('Node', '-file', output_directory+'/1_Pushover_top_disp.out',
             '-node', 20,  '-dof', 1,  'disp')
ops.recorder('Node', '-file', output_directory+'/1_Pushover_base_reactions.out',
             '-node', 10,11,  '-dof', 1,  'reaction')


# Define static load
# ------------------

ops.timeSeries('Linear', 1)     #create timeSeries with tag 1

#pattern(   'Plain', patternTag, tsTag)
ops.pattern('Plain', 1,        1 )

#load   (nodeTag, *loadValues)    loadValues in x direction, y dir, moment
ops.load(20     , *[1,0,0]  )  # load in x direction in node 20




#Define max displacement and displacement increment

Dmax  = 0.40*m; 	# 0.40m   maximum displacement of pushover. It could also be for example 0.1*$H1
Dincr = 0.004*m; 	# 4mm     increment of pushover



# ---- Create Analysis
ops.constraints('Plain')      		    #objects that handles the constraints
ops.numberer('Plain')					#objects that numbers the DOFs
ops.system('FullGeneral')				#objects for solving the system of equations
ops.test('NormDispIncr', 1e-12, 100)    #convergence test, defines the tolerance and the number of iterations
ops.algorithm('Newton') 				#algorithm for solving the nonlinear equations


#integrator('DisplacementControl',   nodeTag, dof, incr)
ops.integrator('DisplacementControl',20,      1 ,  Dincr)

ops.analysis('Static')    #creates a static analysis


# ---- Analyze model
Nsteps = int(Dmax/Dincr); # number of pushover analysis steps
print(f"number of steps Pushover: {Nsteps}")


    
ok = ops.analyze(Nsteps)


if ok == 0: print("-----------------Pushover analysis successfully run--------------------")


plt.figure()
opsv.plot_defo(sfac = 1) 
plt.title('deformed shape - Pushover analysis')
plt.show()  




#%%

# =============================================================================
# plot pushover results
# =============================================================================





ops.wipe() # to close recorders

Pushover_topDisp = np.loadtxt(output_directory+'/1_Pushover_top_disp.out')
Pushover_reactions = np.loadtxt(output_directory+'/1_Pushover_base_reactions.out')

total_base_reaction = -np.sum(Pushover_reactions,axis=1)



if plot_defo_Pushover:
    plt.figure()
    plt.plot(np.insert(Pushover_topDisp, 0, 0),np.insert(total_base_reaction, 0, 0)/1000)   #inserts 0 at the beginning
    plt.title('Pushover curve')
    plt.xlabel('displacement top (m)')
    plt.ylabel('total base shear (kN)')
    plt.grid()
    plt.show()














