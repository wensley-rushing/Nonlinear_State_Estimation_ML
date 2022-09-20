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



import openseespy.opensees as ops
#import openseespy.postprocessing.ops_vis as opsv
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np



from Model_definition_2D_frame import createModel
from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord



# turn on/off the plots by setting these to True or False
plot_model = True
plot_defo_gravity = True
plot_modeshapes = True





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
g = 9.81


# =============================================================================
# Input parameters
# =============================================================================


H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 1000 *kg 	  #kg		lumped mass at top corner nodes 
dampRatio = 0.02


loadfactor = 50

load_file = 'el_centro.AT2'
load_dat_file = 'el_centro.dat'



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
# Compute structural periods after gravity
# =============================================================================

omega_sq = ops.eigen('-fullGenLapack', 2); # eigenvalue mode 1 and 2
omega = np.array(omega_sq)**0.5 # circular natural frequency of modes 1 and 2

periods = 2*np.pi/omega

print(f'First period T1 = {round(periods[0],3)} seconds')


# =============================================================================
# Plot modeshapes
# =============================================================================

if plot_modeshapes:
    plt.figure()
    opsv.plot_mode_shape(1, sfac=1)
    plt.title('mode shape 1')
    plt.show()

    plt.figure()
    opsv.plot_mode_shape(2, sfac=1)
    plt.title('mode shape 2')
    plt.show()


# =============================================================================
# Assign Rayleigh Damping
# =============================================================================

# ---- Rayleigh damping constants for SDOF system (omega1_ray = omega2_ray):

    
omega1_ray = omega[0]
omega2_ray = omega[0]


alphaM = 2.0*dampRatio*omega1_ray*omega2_ray/(omega1_ray+omega2_ray)
betaKcurr = 0
betaKinit = 0
betaKcomm = 2.0*dampRatio/(omega1_ray+omega2_ray)


ops.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)


# =============================================================================
# Dynamic analysis
# =============================================================================



#read record
dt, nPts = ReadRecord(load_file, load_dat_file)







# Define Recorders
output_directory = 'output_files'
ops.recorder('Node', '-file', output_directory+'/2_groundmotion_top_disp.out',
             '-time', '-node', 20,  '-dof', 1,  'disp')
ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_def.out',
             '-ele', 1020,  'section', 1,  'deformation')
ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_force.out',
             '-ele', 1020,  'section', 1,  'force')




# Define dynamic load (ground motion) in the horizontal direction
# ------------------
#time series with tag 1
ops.timeSeries('Path', 1,'-filePath' ,load_dat_file, '-dt', dt,'-factor', loadfactor*g)


#pattern(   'UniformExcitation', patternTag, dir,'-accel', accelSeriesTag)
ops.pattern('UniformExcitation', 1,          1,  '-accel', 1)


# ---- Create Analysis
ops.constraints('Plain')      		    #objects that handles the constraints
ops.numberer('Plain')					#objects that numbers the DOFs
ops.system('FullGeneral')				#objects for solving the system of equations
ops.test('NormDispIncr', 1e-8, 100)    #convergence test, defines the tolerance and the number of iterations
ops.algorithm('Newton') 				#algorithm for solving the nonlinear equations


#integrator('DisplacementControl',   nodeTag, dof, incr)
ops.integrator('Newmark',0.5, 0.25)  # integration method  gamma=0.5   beta=0.25

ops.analysis('Transient')    #creates a dynamic analysis


# ---- Analyze model


current_time = 0      	# start the time of the analysis
ok = 0				    # as long as ok remains 0.0 it means that the analysis converges
maxT =  (1+nPts)*dt;    # final time of the analysis


while ok == 0 and current_time<maxT:
    ok = ops.analyze(1,dt)
    current_time = ops.getTime()



if ok == 0: print("-----------------Dynamic analysis successfully completed--------------------")
else: print(f"-----------------Analysis FAILED at time {current_time}--------------------")


#%%

# =============================================================================
# plot analysis results
# =============================================================================





ops.wipe() # to close recorders

time_topDisp = np.loadtxt(output_directory+'/2_groundmotion_top_disp.out')
sectionDef = np.loadtxt(output_directory+'/2_groundmotion_section_def.out')
sectionForce = np.loadtxt(output_directory+'/2_groundmotion_section_force.out')
                          
              
plt.figure()
plt.plot(time_topDisp[:,0],time_topDisp[:,1])
plt.title('dynamic analysis')
plt.xlabel('time (s)')
plt.ylabel('displacement top (m)')
plt.grid()
plt.show()

plt.figure()
plt.plot(sectionDef[:,1],sectionForce[:,1]/1000)
plt.title('dynamic analysis - section base column')
plt.xlabel('curvature')
plt.ylabel('Moment (kN)')
plt.grid()
plt.show()










