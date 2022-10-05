# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""

import openseespy.opensees as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np



from Model_definition_beam_test_horizontal import createModel
from gravityAnalysis import runGravityAnalysis
from DamageTools import Yielding_point





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

section_n = 5   # number of sections in 1 element



# =============================================================================
# # call function to create the model
# =============================================================================

createModel(H1, L1, M)


if plot_model:
    plt.figure()
    opsv.plot_model()
    plt.title('model')
    plt.show()  


   

# =============================================================================
# Pushover analysis
# =============================================================================

# Define Recorders


# Global P - top displ curve

output_directory = 'output_files'
ops.recorder('Node', '-file', output_directory+'/001_pushover_beam_glob_disp.out',
             '-node', 11,  '-dof', 2,  'disp')
ops.recorder('Node', '-file', output_directory+'/001_pushover_beam_glob_force.out',
             '-node', 10,  '-dof', 2,  'reaction')


# Local M-curvature curves

output_directory = 'output_files'

# Columns

ops.recorder('Element', '-file', output_directory+'/001_pushover_beam_force.out',
             '-ele', 1011,  'section',  'force')
ops.recorder('Element', '-file', output_directory+'/001_pushover_beam_deform.out',
             '-ele', 1011,  'section',  'deformation')

# Define static load
# ------------------

ops.timeSeries('Linear', 1)     #create timeSeries with tag 1

#pattern(   'Plain', patternTag, tsTag)
ops.pattern('Plain', 1,        1 )

#load   (nodeTag, *loadValues)    loadValues in x direction, y dir, moment
ops.load(11     , *[0,1,0]  )  # load in x direction in node 20




#Define max displacement and displacement increment

Dmax  = 0.06*m; 	# 0.40m   maximum displacement of pushover. It could also be for example 0.1*$H1
Dincr = 0.004*m; 	# 4mm     increment of pushover



# ---- Create Analysis
ops.constraints('Plain')      		    #objects that handles the constraints
ops.numberer('Plain')					#objects that numbers the DOFs
ops.system('FullGeneral')				#objects for solving the system of equations
ops.test('NormDispIncr', 1e-12, 100)    #convergence test, defines the tolerance and the number of iterations
ops.algorithm('Newton') 				#algorithm for solving the nonlinear equations


#integrator('DisplacementControl',   nodeTag, dof, incr)
ops.integrator('DisplacementControl',11,      2 ,  Dincr)

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
# capacities calculation
# =============================================================================

ops.wipe() # to close recorders

# Global 

Pushover_topDisp = np.loadtxt(output_directory+'/001_pushover_beam_glob_disp.out')
total_base_reaction = np.loadtxt(output_directory+'/001_pushover_beam_glob_force.out')
total_base_reaction = -(total_base_reaction)/1000

if plot_defo_Pushover:
    plt.figure()
    plt.plot(np.insert(Pushover_topDisp, 0, 0),np.insert(total_base_reaction, 0, 0))   #inserts 0 at the beginning
    plt.title('Pushover test: beam \n Global result')
    plt.xlabel('displacement top (m)')
    plt.ylabel('total base shear (kN)')
    plt.grid()
    plt.show()



#%% Local 

col_moment = np.loadtxt(output_directory+'/001_pushover_col_force.out')
col_moment = abs(col_moment[:, 1::2])/1000 # select even columns - corresponding to moments

col_curv = np.loadtxt(output_directory+'/001_pushover_col_deform.out')
col_curv = abs(col_curv[:, 1::2]) # select even columns - corresponding to curvatures

max_M = max(col_moment.max(axis=0))

for i in range(0, section_n):
    
    if max(col_moment[:,i]) == max_M:                    # select critical cross section
        print('Section [' + str(i+1) + '] is critical')
        crit_sec = i
        
        
    # plt.figure()
    # plt.plot(col_curv[:,i] ,col_moment[:,i])
    # plt.xlabel('Curvature [-]')
    # plt.ylabel('Moment [kNm]')
    # plt.title('Column moment - curvature plot \n in Section [' + str(i+1) + ']')
    # plt.grid()
    # plt.show()

curv_y, M_y, curv_u, M_u = Yielding_point(col_curv[:,crit_sec], col_moment[:,crit_sec])

print('Column: ult(%.4f  %.4f)  yiled(%.4f  %.4f)' %(curv_u, M_u, curv_y, M_y))  

plt.figure()
plt.plot(np.insert(col_curv[:,crit_sec], 0, 0),np.insert(col_moment[:,crit_sec], 0, 0))   #inserts 0 at the beginning
plt.plot(curv_y,M_y, marker="x", markersize=7, markeredgecolor="red")
plt.plot(curv_u, M_u, marker="x", markersize=7, markeredgecolor="green")
plt.plot([0,curv_y, curv_u], [0,M_y,M_u], c='red')
plt.title('Pushover test: column')
plt.xlabel('Curvature [-]')
plt.ylabel('Moment [kNm]')
plt.grid()
plt.show()