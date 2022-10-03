# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""

import openseespy.opensees as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np



from Model_definition_column_test import createModel
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
M = 6000 *kg 	  #kg		lumped mass at top corner nodes 

section_n = 5   # number of sections in 1 element



# =============================================================================
# # call function to create the model
# =============================================================================

createModel(H1, M)


if plot_model:
    plt.figure()
    opsv.plot_model()
    plt.title('model')
    plt.show()  


   

# =============================================================================
# Pushover analysis
# =============================================================================

# Define Recorders

# Global P-delta curve

output_directory = 'output_files'
ops.recorder('Node', '-file', output_directory+'/001_capacities_glob_top_disp.out',
             '-node', 20,  '-dof', 1,  'disp')
ops.recorder('Node', '-file', output_directory+'/001_capacities_glob_base_reactions.out',
             '-node', 10,  '-dof', 1,  'reaction')


# Local M-curvature curves

# # Beams
# ops.recorder('Element', '-file', output_directory+'/001_capacities_loc_force_beam.out',
#              '-ele', beam_vec[0],  'section', 'force')
# ops.recorder('Element', '-file', output_directory+'/001_capacities_loc_deform_beam.out',
#              '-ele', beam_vec[0],  'section', 'deformation')

# Columns
ops.recorder('Element', '-file', output_directory+'/001_capacities_loc_force_col.out',
             '-ele', 1020,  'section',  'force')
ops.recorder('Element', '-file', output_directory+'/001_capacities_loc_deform_col.out',
             '-ele', 1020,  'section',  'deformation')

# Define static load
# ------------------

ops.timeSeries('Linear', 1)     #create timeSeries with tag 1

#pattern(   'Plain', patternTag, tsTag)
ops.pattern('Plain', 1,        1 )

#load   (nodeTag, *loadValues)    loadValues in x direction, y dir, moment
ops.load(20     , *[1,0,0]  )  # load in x direction in node 20




#Define max displacement and displacement increment

Dmax  = 0.6*m; 	# 0.40m   maximum displacement of pushover. It could also be for example 0.1*$H1
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
# capacities calculation
# =============================================================================

ops.wipe() # to close recorders

#%% Local

# beam_moment = np.loadtxt(output_directory+'/001_capacities_loc_force_beam.out')
# beam_moment = abs(beam_moment[:, 1::2])/1000 # select even columns - corresponding to moments

# beam_curv = np.loadtxt(output_directory+'/001_capacities_loc_deform_beam.out')
# beam_curv = abs(beam_curv[:, 1::2]) # select even columns - corresponding to curvatures



# file: 001_capacities_loc_force_col

col_moment = np.loadtxt(output_directory+'/001_capacities_loc_force_col.out')
col_moment = abs(col_moment[:, 1::2])/1000 # select even columns - corresponding to moments

col_curv = np.loadtxt(output_directory+'/001_capacities_loc_deform_col.out')
col_curv = abs(col_curv[:, 1::2]) # select even columns - corresponding to curvatures


for i in range(0, section_n-3):
    
    
    # fig, axs = plt.subplots(1, 2)
    
    # axs[0].plot(beam_curv[:,i] ,beam_moment[:,i])
    # axs[0].set(xlabel='Curvature [-]', ylabel='Moment [kNm]')
    # axs[0].set_title('Moment curvature for beam [' + str(beam_vec[0]) + '] in section [' + str(i+1) + ']')
    # axs[0].grid()
    
    # axs[1].plot(col_curv[:,i] ,col_moment[:,i])
    # axs[1].set(xlabel='Curvature [-]', ylabel='Moment [kNm]')
    # axs[1].set_title('Moment curvature for column [' + str(col_vec[0]) + '] in section [' + str(i+1) + ']')
    # axs[1].grid()
    
    # fig.tight_layout()
    # plt.show()

      
    # x = [beam_curv[0,i],beam_curv[1,i]]
    # y = [beam_moment[0,i],beam_moment[1,i]]
    # K_i = abs(y[0]-y[1]) / abs(x[0]-x[1])

    # A_c = np.trapz(beam_moment[:,i], x=beam_curv[:,i])

    # curv_u_b = beam_curv[-1, i]
    # M_u_b = beam_moment[-1, i]

    # curv_y_b = abs((2*A_c - (M_u_b*curv_u_b)) / ((K_i*curv_u_b) - M_u_b))
    # M_y_b = K_i*curv_y_b
    
    
    # print('Beam: ult(%.4f  %.4f)  yiled(%.4f  %.4f)' %(curv_u_b, M_u_b, curv_y_b, M_y_b))
   
    x = [col_curv[0,i],col_curv[1,i]]
    y = [col_moment[0,i],col_moment[1,i]]
    
    K_i = abs(y[0]-y[1]) / abs(x[0]-x[1])

    A_c = np.trapz(col_moment[:,i], x=col_curv[:,i])

    curv_u_c = col_curv[-1, i]
    M_u_c = col_moment[-1, i]

    curv_y_c = abs((2*A_c - (M_u_c*curv_u_c)) / ((K_i*curv_u_c) - M_u_c))
    M_y_c = K_i*curv_y_c


    print('Column: ult(%.4f  %.4f)  yiled(%.4f  %.4f)' %(curv_u_c, M_u_c, curv_y_c, M_y_c))   

    fig, axs = plt.subplots(1, 2)
    
    # axs[0].plot(beam_curv[:,i] ,beam_moment[:,i])
    # axs[0].scatter(curv_y_b, M_y_b, marker="x", color="red")
    # axs[0].scatter(curv_u_b, M_u_b, marker="x", color="green")
    # axs[0].set(xlabel='Curvature [-]', ylabel='Moment [kNm]')
    # axs[0].set_title('Moment curvature for beam [' + str(beam_vec[0]) + '] in section [' + str(i+1) + ']')
    # axs[0].grid()
    
    axs[1].plot(col_curv[:,i] ,col_moment[:,i])
    axs[1].set(xlabel='Curvature [-]', ylabel='Moment [kNm]')
    axs[1].scatter(curv_y_c, M_y_c, marker="x", color="red")
    axs[1].scatter(curv_u_c, M_u_c, marker="x", color="green")
    axs[1].set_xlim([0, 0.015])
    axs[1].set_ylim([0, 110])
    # axs[1].set_title('Moment curvature for column [' + str(col_vec[0]) + '] in section [' + str(i+1) + ']')
    axs[1].grid()
    
    fig.tight_layout()
    plt.show()
    
    
#%%






# =============================================================================
# plot pushover results
# =============================================================================




# ops.wipe() # to close recorders

# Pushover_topDisp = np.loadtxt(output_directory+'/001_Pushover_top_disp.out')
# Pushover_reactions = np.loadtxt(output_directory+'/001_Pushover_base_reactions.out')

# total_base_reaction = -np.sum(Pushover_reactions,axis=1)/1000



# F_max  = abs(max(total_base_reaction))
# F_max_index = np.where(total_base_reaction == F_max)[0][0]

# delta_y = 0.8*F_max/slope
# delta_u = Pushover_topDisp[F_max_index]





if plot_defo_Pushover:
    plt.figure()
    plt.plot(np.insert(Pushover_topDisp, 0, 0),np.insert(total_base_reaction, 0, 0))   #inserts 0 at the beginning
    plt.plot(D_y, F_y, marker="x", markersize=7, markeredgecolor="red", markerfacecolor="red")
    plt.plot(D_u, F_u, marker="x", markersize=7, markeredgecolor="red", markerfacecolor="red")
    plt.title('Pushover curve')
    plt.xlabel('displacement top (m)')
    plt.ylabel('total base shear (kN)')
    plt.grid()
    plt.show()















