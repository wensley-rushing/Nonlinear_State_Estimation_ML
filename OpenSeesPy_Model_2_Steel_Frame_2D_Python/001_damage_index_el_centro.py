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
import pandas as pd


from Model_definition_2D_frame import createModel
from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord



# turn on/off the plots by setting these to True or False
plot_model = True
plot_defo_gravity = False
plot_modeshapes = False



delta_y = 0.121771097
delta_u = 0.4


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

st = 2

if st == 2:
    ACC_Nodes = [20,30]
elif st == 1:
    ACC_Nodes = [20]

H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 1000 *kg 	  #kg		lumped mass at top corner nodes 
dampRatio = 0.02

df = pd.DataFrame(columns = ['Load factor', 'Damage index', 'Entropy'])


loadfactor_idx = 0
for loadfactor in [3,4,5,7,10,20,30,40]:
    loadfactor_idx = loadfactor_idx + 1
    
    if loadfactor_idx > 1:
        plot_model = False  
    

    
    load_file = 'el_centro.AT2'
    load_dat_file = 'el_centro.dat'
    
    
    
    # =============================================================================
    # # call function to create the model
    # =============================================================================
    
    createModel(H1,L1,M, st)
    
    if plot_model:
        plt.figure()
        opsv.plot_model()
        plt.title('model')
        #plt.show()  
    
    
    
    # =============================================================================
    # Run gravity Analysis
    # =============================================================================
    runGravityAnalysis()
    
    if plot_defo_gravity: 
        plt.figure()
        opsv.plot_defo(sfac = 10000) 
        plt.title('deformed shape - gravity analysis')
        #plt.show()  
    
    
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
        #plt.show()
    
        plt.figure()
        opsv.plot_mode_shape(2, sfac=1)
        plt.title('mode shape 2')
        #plt.show()
    
    
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
                 '-time', '-node', max(ACC_Nodes),  '-dof', 1,  'disp')
    ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_def.out',
                 '-ele', 1020,  'section', 1,  'deformation')
    ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_force.out',
                 '-ele', 1020,  'section', 1,  'force')
    
    
    ops.recorder('Node', '-file', output_directory+'/2_groundmotion_base_reactions.out',
             '-node', 10,11,  '-dof', 1,  'reaction')

    
    
    for nodes in ACC_Nodes:
        ops.recorder('Node', '-file', output_directory+'/2_Acc_x_' + str(nodes) +'.out',
                     '-time', '-node', nodes,  '-dof', 1,  'accel')
    
    
    
    
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
    plt.title('dynamic analysis \n GM: ' + load_file)
    plt.xlabel('time (s)')
    plt.ylabel('displacement top (m)')
    plt.grid()
    #plt.show()
    
    plt.figure()
    plt.plot(sectionDef[:,1],sectionForce[:,1]/1000)
    plt.title('dynamic analysis - section base column \n GM: ' + load_file)
    plt.xlabel('curvature')
    plt.ylabel('Moment (kN)')
    plt.grid()
    #plt.show()
    
    
    
    # =============================================================================
    # Labelling
    # =============================================================================

    delta_max = abs(max(time_topDisp[:,1]))

    damage_idx = delta_max/delta_y

    print('------------------DAMAGE LABEL ------------------')
    print('Load factor: ', loadfactor)

    if damage_idx < 1:
        print('Damage index: ', round(damage_idx,4), '<1 structure is undamaged!')
    else:
        print('Damage index: ', round(damage_idx,4), '>1 structure is damaged!')
    
    
    
    #%% Estimate entropy
     
    import DamageTools
    
    print('DAMAGE LABEL -----------------------------------------------------------')
    print('Load factor: ', loadfactor)
    print()
    
    
    Entropy = []
    Entropy_labels = []
    
    for nodes in ACC_Nodes:
        time_Acc_x_XX = np.loadtxt(output_directory+'/2_Acc_x_' + str(nodes) +'.out')
    
        ACC_x_XX = time_Acc_x_XX[:,1]
    
        entropy = DamageTools.SampEn(ACC_x_XX, 2, 0.2*np.std(ACC_x_XX))
        Entropy.append(entropy)
        
        Entropy_labels.append('E_'+str(nodes)) 
        
        print('Entropy Node_%d: ' % nodes, round(entropy,4))

    
    df.loc[loadfactor_idx-1] = [loadfactor, damage_idx, Entropy]

#%%
DF_labels = ['Damage index']
DF_labels.extend(Entropy_labels)

DF = pd.DataFrame(columns=DF_labels)
for i in range(len(df)):
    vec = []
    
    vec.append(df['Damage index'][i])
    vec.extend(df['Entropy'][i])
    
    DF.loc[i] = vec
    
corr = DF.corr()

#%%

markers = ['x', 'o']
colors = ['red', 'green']


plt.figure()
for i in range(len(df['Entropy'][0])):
    label_i = Entropy_labels[i]+'corr= %.2f' % corr.iloc[0,i+1]
    plt.scatter(DF['Damage index'],DF.iloc[:,i+1],  color=colors[i], marker=markers[i], label=label_i)
plt.title('Entropy \n GM: ' + load_file)
plt.xlabel('Damage index')
plt.ylabel('Entropy')
plt.legend()
plt.grid()


fig, axs = plt.subplots(1, 2)
for i in range(len(df['Entropy'][0])):
    axs[i].scatter(DF['Damage index'],DF.iloc[:,i+1],  color=colors[i], marker=markers[i], label=label_i)
    axs[i].set_title(Entropy_labels[i]+'corr= %.2f' % corr.iloc[0,i+1])
    axs[i].set(xlabel='Damage index', ylabel='Entropy')
    axs[i].grid()
fig.suptitle('Entropy \n GM: ' + load_file)
fig.tight_layout()


DF.to_csv(r'el_centro_dataframe.csv')
 
# UnDamaged: 
# loadlevel: 1
#  E = 210          E = 150
#  20:  0.1496      :0.1325
#  30:  0.1220      :0.1009


# UnDamaged: 
# loadlevel: 5
#  E = 210          E = 150
#  20:  0.1496      :0.1325
#  30:  0.1220      :0.1009
    
    
# UnDamaged: 
# loadlevel: 7
#  E = 210          E = 150
#  20:  0.1496      :0.1326
#  30:  0.1222      :0.1009
    
    
    
# Damaged: 
# loadlevel: 10
#  E = 210          E = 150
#  20:  0.1664        :0.134
#  30:  0.1349        :0.1019
    
    
    
# Damaged: 
# loadlevel: 25
#  E = 210          E = 150
#  20:  0.2731        :0.2069
#  30:  0.2311        :0.1772
    
    
    
# Damaged: 
# loadlevel: 50
#  E = 210          E = 150
#  20:  0.2855        :0.2337
#  30:  0.2731        :0.2175

import sys
sys.exit()









