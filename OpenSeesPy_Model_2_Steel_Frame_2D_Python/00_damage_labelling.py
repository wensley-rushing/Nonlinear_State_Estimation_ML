# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""




import openseespy.opensees as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np


from gravityAnalysis import runGravityAnalysis
from Model_definition_2D_frame import createModel
from Module_analysis import pushover_analysis
from ReadRecord import ReadRecord

import DamageTools


# turn on/off the plots by setting these to True or False
plot_model = True
plot_modeshapes = True
plot_defo_Pushover = True
plot_dynamic = True

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

## Define the model ----------

st = 2 # n. storeys number

# Acceleration nodes to be recorded
if st == 2:
    ACC_Nodes = [20,30]
elif st == 1:
    ACC_Nodes = [20]
    

H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 1000 *kg 	  #kg		lumped mass at top corner nodes 
dampRatio = 0.02


Loadfactors = [1, 3, 5, 7, 10, 13, 16, 20]

load_file = 'el_centro.AT2'
load_dat_file = 'el_centro.dat'


## Pushover analysis ----------

node_supp1 = 10
node_supp2 = 11
node_load = 30

nodes = [node_supp1, node_supp2, node_load]

Dmax = 0.40*m
Dincr = 0.004*m


# =============================================================================
# # call function to create the model + plot
# =============================================================================

createModel(H1,L1,M, st)

if plot_model:
    plt.figure()
    opsv.plot_model()
    plt.title('model')
    plt.show()  


# =============================================================================
# Run gravity Analysis
# =============================================================================
runGravityAnalysis()

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
# Run Pushover Test
# =============================================================================

delta_y, delta_u, p_curve_x, p_curve_y = pushover_analysis(H1, L1, M, Dmax, Dincr, nodes, plot_defo_Pushover)


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



for loadfactor in Loadfactors:
    
    j = Loadfactors.index(loadfactor)

    #read record
    
    dt, nPts = ReadRecord(load_file, load_dat_file)
    
    
    # Define Recorders
    output_directory = 'output_files'
    
    ops.recorder('Node', '-file', output_directory+'/2_groundmotion_top_disp.out',
                 '-time', '-node', max(ACC_Nodes),  '-dof', 1,  'disp')
    ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_def.out',
                  '-ele', 1020,  'section', 1,  'deformation')
    ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_force.out',
                  '-ele', 1020,  'section', 1,  'force')
    
    ops.recorder('Node', '-file', output_directory+'/2_groundmotion_base_reactions.out',
                 '-node', node_supp1, node_supp2,  '-dof', 1,  'reaction')
    
    
    
    
    # Acceleration recordings
    for acc_nodes in ACC_Nodes:
        ops.recorder('Node', '-file', output_directory+'/2_Acc_x_' + str(acc_nodes) +'.out',
                     '-time', '-node', acc_nodes,  '-dof', 1,  'accel')
        
    
    
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
    
    
    # Plot
    
    time_topDisp = np.loadtxt(output_directory+'/2_groundmotion_top_disp.out')

    gm_base_reaction_dy = np.loadtxt(output_directory+'/2_groundmotion_base_reactions.out')
    
    if plot_dynamic:
    
        plt.figure()
        plt.plot(time_topDisp[:,1],gm_base_reaction_dy)   #inserts 0 at the beginning
        plt.plot(np.insert(p_curve_x, 0, 0),np.insert(p_curve_y, 0, 0))
        plt.title('Pushover curve')
        plt.xlabel('displacement top (m)')
        plt.ylabel('total base shear (kN)')
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.plot(time_topDisp[:,0],time_topDisp[:,1])
        plt.title('dynamic analysis')
        plt.xlabel('time (s)')
        plt.ylabel('displacement top (m)')
        plt.grid()
        #plt.show()
    
    # =============================================================================
    # Estimate Entropy
    # =============================================================================
    
    
    print('DAMAGE LABEL -----------------------------------------------------------')
    print('Load factor: ', loadfactor)
    print()
    
    for acc_nodes in ACC_Nodes:
        ACC_x = np.loadtxt(output_directory+'/2_Acc_x_' + str(acc_nodes) +'.out')[:,1]
    
        entropy = DamageTools.SampEn(ACC_x, 2, 0.2*np.std(ACC_x))
        print('Entropy Node_%d: ' % acc_nodes, round(entropy,4))












