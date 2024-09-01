# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 20:22:15 2022

@author: larsk
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""

#%% IMPORTS

# Import OpenSees
import opensees.openseespy as ops
import opsvis as opsv

# Import analysis (OpenSees)
from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord


# Import plots and math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import os


# Import time-keeping
import time

# Wipe before anything...
ops.wipe()

#%% Enable/Disable plots
# turn on/off the plots by setting these to True or False

# General structure
plot_model = True
plot_defo_gravity = True
plot_modeshapes = True

# Dynamic analysis
plot_dynamic_analysis = True

#%% UNITS
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


#%% INPUTS
# =============================================================================
# Input parameters
# =============================================================================
delta_y = 0.02


# Define structure and nodes/elements
define_model = '3x3'

if define_model == '1x1':
    from Model_definition_1x1_frame import createModel
    
    # Global DI
    support_nodes = [10, 11]
    top_nodes = [20, 21]
    drift_nodes = [11, 21]
    
    #Local DI
    id_element = [1020]
    
elif define_model == '2x1':
    from Model_definition_2x1_frame import createModel
    
    # Global DI
    support_nodes = [10, 11]
    top_nodes = [30, 31]
    drift_nodes = [11, 21, 31]
    
    #Local DI
    id_element = [1020]
    
elif define_model == '3x3':
    from Model_definition_3x3_frame import createModel
    
    # Global DI
    support_nodes = [10, 11, 12, 13]
    top_nodes = [40, 41, 42, 43]
    drift_nodes = [13, 23, 33, 43]
    
    #Local DI
    id_element = [1020, 2030]
#------------------------------------------------------------------------------
    
# Geometry
H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 6000 *kg 	  #kg		lumped mass at top corner nodes 
dampRatio = 0.02



#%% Initialization 

# Create Dataframe for results
gm_idx = 0
df = pd.DataFrame(columns = ['OK=0', 'Ground motion', 'Load factor',           #General
                             'E - glob', 'Gl Drift', 'Gl Drift - class',       # Global
                             'Element ID', 'E - elem', 'Plastic def. ele.'])   # Local
 
#%% Gravity analysis

# =============================================================================
# # call function to create the model
# =============================================================================

node_vec, el_vec, col_vec = createModel(H1,L1,M)

beam_vec = [i for i in el_vec if i not in col_vec]


if plot_model:
    plt.figure()
    opsv.plot_model()
    plt.title('Structural Model')
    #plt.show()  



# =============================================================================
# Run gravity Analysis
# =============================================================================

runGravityAnalysis(beam_vec)

if plot_defo_gravity: 
    plt.figure()
    opsv.plot_defo(sfac = 10000)
    plt.title('Deformed shape - Gravity analysis')
    #plt.show()  


# wipe analysis objects and set pseudo time to 0
ops.wipeAnalysis()
ops.loadConst('-time', 0.0)



# =============================================================================
# Compute structural periods after gravity
# =============================================================================

omega_sq = ops.eigen('-fullGenLapack', 3); # eigenvalue mode 1, 2 and 3
omega = np.array(omega_sq)**0.5 # circular natural frequency of modes 1 and 2

periods = 2*np.pi/omega

print(f'Period(s) of structure [s]: {np.around(periods, decimals = 4)}')
print()


# =============================================================================
# Plot modeshapes
# =============================================================================

if plot_modeshapes:
    for i in range(len(periods)):
        plt.figure()
        opsv.plot_mode_shape(i+1, sfac=100)
        plt.title(f'Mode shape {i+1}')
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
    
#%% Database
ops.database('File', 'DataBase\\test_db')
# Created the copy
ops.save(199)

#%% Import data    
# Import multiple loads

# Getting the work directory of loads .AT1 or .AT2 files
folder_loads = os.path.join(os.getcwd(), 'import_loads\\TT')
#r'C:\Users\larsk\Danmarks Tekniske Universitet\Thesis_Nonlinear-Damage-Detection\OpenSeesPy_Model_2_Steel_Frame_2D_Python\load_files'

# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_loads):  
    for file in files:
        if file.endswith(".AT1") or file.endswith(".AT2"):
            #print(os.path.join(rdirs, file))
            #print(idx)
            #print(file)
            
            file_name = file[:-4]
            #print('Loading: ' + file_name)
            
            file_dat = file_name + '.dat'
            # print(file_dat)
            
            
            load_file = file
            load_dat_file = file_dat


            #Move up in folder structure
            #from pathlib import Path
            #p = Path(__file__).parents[0]
            
            #print(p)
            # /absolute/path/to/two/levels/up
            #idx += 1
                               
            load_file = os.path.join(folder_loads, file)
            load_dat_file = os.path.join(folder_loads, file_dat)

            # load_file = 'el_centro.AT2'
            # load_dat_file = 'el_centro.dat'
            # file_name = 'el_centro'
            
            
            
            loadfactor_idx = 0
            for loadfactor in [1]:
                loadfactor_idx = loadfactor_idx + 1
                
                print('Load: ' + file_name + ' -- Loadfactor: %.2f' %(loadfactor))
                            
                ops.restore(199)
                # =============================================================================
                # Dynamic analysis
                # =============================================================================
                
                #read record
                dt, nPts = ReadRecord(load_file, load_dat_file)
                
            
            
                # Define Recorders
                output_directory = 'output_files'
                
                
                # Base reaction recorder
                ops.recorder('Node', '-file', output_directory+'/2_Reaction_Base.out',
                             '-node', *support_nodes,  '-dof', 1,  'reaction')
            
                # create recorder files for displacements (interstory drift - Global)
                #for nodes in drift_nodes:
                ops.recorder('Node', '-file', output_directory+'/2_Dsp_Drift_Nodes.out',
                             '-time', '-node', *drift_nodes,  '-dof', 1,  'disp')
                
                
                # Plastic deformations (element)
                ops.recorder('Element', '-file', output_directory+'/2_Plastic_Def.out',
                              '-time', '-ele', *id_element, 'plasticDeformation ')
                
                # Section deformation () (local)
                ops.recorder('Element', '-file', output_directory+'/2_Section_Force.out',
                             '-ele', *id_element,  'section', 1,  'force')
                
                ops.recorder('Element', '-file', output_directory+'/2_Section_Def.out',
                             '-ele', *id_element,  'section', 1,  'deformation')
                
                
                
                # Recorder files for all nodes
                #for nodes in ACC_Nodes:
                #    ops.recorder('Node', '-file', output_directory+'/2_Acc_x_' + str(nodes) +'.out',
                #                 '-time', '-node', nodes,  '-dof', 1,  'accel')
                
                
                
                
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
                #ops.setTime(0)
                
                while ok == 0 and current_time<maxT:
                    ok = ops.analyze(1,dt)
                    current_time = ops.getTime()
                    #print(current_time)
                
                
                
                if ok == 0: print("-----------------Dynamic analysis successful--------------------")
                else: print(f"-----------------Dynamic analysis FAILED @ time {current_time}--------------------"); ok = 1
                

                

                # =============================================================================
                # plot analysis results
                # =============================================================================
                
                # ops.remove('recorders') # to close recorders
                # ops.remove('timeSeries',1) # to close timeSeries
                # ops.remove('loadPattern',1) # to close timeSeries
                
                #ops.wipe() # Instead of ops.wipe() use:
                ops.wipeAnalysis()
                ops.remove('recorders')
                
                ops.remove('loadPattern', 1)
                ops.remove('timeSeries', 1)
                
                
                # Retrive data
                base_shear = np.loadtxt(output_directory+'/2_Reaction_Base.out')
                total_base_shear = -np.sum(base_shear,axis=1)
                
                time_drift_disp = np.loadtxt(output_directory+'/2_Dsp_Drift_Nodes.out')
                
                #time_topDisp = np.loadtxt(output_directory+'/2_groundmotion_top_disp.out')
                #sectionDef = np.loadtxt(output_directory+'/2_groundmotion_section_def.out')
                #sectionForce = np.loadtxt(output_directory+'/2_groundmotion_section_force.out')
                
                
                if plot_dynamic_analysis:
                    # Top displacement over time
                    plt.figure()
                    plt.plot(time_drift_disp[:,0],time_drift_disp[:,len(drift_nodes)])
                    plt.title('Dynamic analysis \n GM: ' + file_name + ' -  Loadfactor: ' + str(loadfactor))
                    plt.xlabel('Time (s)')
                    plt.ylabel('Top displacement node %.0f (m)' %(drift_nodes[-1]))
                    plt.grid()
                    #plt.show()
                    
                    # Hysterises loop Global (Base shear vs. top disp)
                    plt.figure()
                    plt.plot(time_drift_disp[:,len(drift_nodes)],total_base_shear/1000)
                    plt.title('Dynamic analysis - Structure \n GM: ' + file_name + ' -  Loadfactor: ' + str(loadfactor))
                    plt.xlabel('Roof displacement (m)')
                    plt.ylabel('Total base shear (kN)')
                    plt.grid()
                    #plt.show()
                    
                 
                    
                #%% Energy (Global)
                
                # Damage index information of interest
                DI_x = time_drift_disp[:,len(drift_nodes)] #Deformation (curvature) 
                DI_y = total_base_shear/1000 #Force (Moment)
                
                
                Energy_G = np.trapz(DI_y, x=DI_x)
                print('Energy - Global: %.4f' %(Energy_G))
                
                corr = np.corrcoef(DI_x, DI_y)
                Corr = corr[0][1]
                print('Correlation: %.4f' %(Corr))
                    
                    
                # Interstorey drift (Global)
        
                n_floors = len(time_drift_disp[0]) - 2 # do not count the columns for time and base node 
                
                drift = []
                inter_drift = []
                inter_time_drift = []
                for i in range (0, n_floors):
                    drift.append(abs(time_drift_disp[-1, i+2]) / H1) # residual drift
                    inter_drift.append( (abs(time_drift_disp[-1, i+2])  -  abs(time_drift_disp[-1, i+1])) / H1 ) # residual drift
                    inter_time_drift.append( abs(max(time_drift_disp[:,i+2]-time_drift_disp[:,i+1], key=abs)) / H1 ) # residual drift
                    
                max_drift = max(drift)*100 # residual drift in percentage
                max_inter_drift = max(inter_drift)*100
                max_inter_time_drift = max(inter_time_drift)*100
                
                if max_inter_drift < 0.2:
                    drift_cl = 'No damage'     
                elif max_inter_drift <= 0.5:
                    drift_cl = 'Repairable'     
                elif max_inter_drift < 1.5:
                    drift_cl = 'Irreparable'      
                elif max_inter_drift < 2.5:
                    drift_cl = 'Severe'
                elif max_inter_drift > 2.5:
                    drift_cl = 'Collapse'
                    
                    
                if max_inter_time_drift < 0.2:
                    drift_time_cl = 'No damage'     
                elif max_inter_time_drift <= 0.5:
                    drift_time_cl = 'Repairable'     
                elif max_inter_time_drift < 1.5:
                    drift_time_cl = 'Irreparable'      
                elif max_inter_time_drift < 2.5:
                    drift_time_cl = 'Severe'
                elif max_inter_time_drift > 2.5:
                    drift_time_cl = 'Collapse'
                
                print('Max drift: ' + str(round(max_drift,4)))
                print('Max inter. drift: ' + str(round(max_inter_drift,4))  + ' - Class: ' + drift_cl)
                print('Max inter. time drift: ' + str(round(max_inter_time_drift,4))  + ' - Class: ' + drift_time_cl)
                    
                    
                    
                #%% Local ---------------
                
                # Moment curvature curce
                
                element_section_forces = np.loadtxt(output_directory+'/2_Section_Force.out')
                element_section_defs = np.loadtxt(output_directory+'/2_Section_Def.out')
                
                Energy_L = []
                for el_id in range(len(id_element)):
                    
                    plt.figure()
                    plt.plot(element_section_defs[:,(el_id*2)+1],element_section_forces[:,(el_id*2)+1]/1000)
                    plt.title('Dynamic analysis - Element ' + str(id_element[el_id]) + ' \n GM: ' + file_name + ' -  Loadfactor: ' + str(loadfactor) )
                    plt.xlabel('Curvature (-)')
                    plt.ylabel('Moment (kNm)')
                    plt.grid()
                    
                    
                    # Damage index information of interest
                    DI_x = element_section_defs[:,(el_id*2)+1] #Deformation (curvature) 
                    DI_y = element_section_forces[:,(el_id*2)+1]/1000 #Force (Moment)
                    
                    
                    Energy_l = np.trapz(DI_y, x=DI_x)
                    print('Energy - Element %.0f: %.4f' %(id_element[el_id],Energy_l))
                    Energy_L.append(Energy_l)
                    
                    corr = np.corrcoef(DI_x, DI_y)
                    Corr = corr[0][1]
                    print('Correlation - Element %.0f: %.4f' %(id_element[el_id],Corr))
                    
                    
                    #max_plasic_deforms = plastic_deform[-1:].tolist()
                    #max_plasic_deform = max(max_plasic_deforms[0][1:], key=abs)
                    #print('Max plastic defomation, el_%.0f: %0.4f' %(id_element[el_id], max_plasic_deform))
    
                # Energy (Local)
            
                
                # Plastic deformation 
                plastic_deform = np.loadtxt(output_directory+'/2_Plastic_Def.out')
                Max_plastic_deform = []
                for el_id in range(len(id_element)):
                    plt.figure()
                    for i in range((el_id*3)+1,(el_id*3)+4):
                        plt.plot(plastic_deform[:,0],plastic_deform[:,i], label=str(i))
                    plt.title('Plastic Deformation - Element ' + str(id_element[el_id]) + ' \n GM: ' + file_name + ' -  Loadfactor: ' + str(loadfactor) )
                    plt.xlabel('Time (s)')
                    plt.ylabel('Plastic deformation (-)')
                    plt.legend()
                    plt.grid()
                    
                    
                    max_plastic_deforms = plastic_deform[-1:].tolist()
                    max_plastic_deform = max(max_plastic_deforms[0][1:], key=abs)
                    print('Max plastic defomation, el_%.0f: %0.4f' %(id_element[el_id], max_plastic_deform))
                    Max_plastic_deform.append(max_plastic_deform)
            
                #%% Record data in the dataframe
                df.loc[gm_idx] = [ok, file_name, loadfactor,                    # General
                                  Energy_G, max_inter_drift, drift_cl,          # Global
                                  id_element, Energy_L, Max_plastic_deform]     # Local
                
                gm_idx += 1
                
                
                

# export dataframe

df.to_csv(r'el_centro_dataframe.csv')  # export dataframe to cvs

sys.exit()
#%% ??
    # delta_max = abs(max(time_topDisp[:,1]))

    # damage_idx = delta_max/delta_y

    # print('------------------DAMAGE LABEL ------------------')
    # print('Load factor: ', loadfactor)

    # if damage_idx < 1:
    #     print('Damage index: ', round(damage_idx,4), '<1 structure is undamaged!')
    # else:
    #     print('Damage index: ', round(damage_idx,4), '>1 structure is damaged!')
    
    
    
    #%% Estimate entropy
     
    # import DamageTools
    
    # print('DAMAGE LABEL -----------------------------------------------------------')
    # print('Load factor: ', loadfactor)
    # print()
    
    
    # Entropy = []
    # Entropy_labels = []
    
    # for nodes in ACC_Nodes:
    #     time_Acc_x_XX = np.loadtxt(output_directory+'/2_Acc_x_' + str(nodes) +'.out')
    
    #     ACC_x_XX = time_Acc_x_XX[:,1]
    
    #     entropy = DamageTools.SampEn(ACC_x_XX, 2, 0.2*np.std(ACC_x_XX))
    #     Entropy.append(entropy)
        
    #     Entropy_labels.append('E_'+str(nodes)) 
        
    #     print('Entropy Node_%d: ' % nodes, round(entropy,4))

    
    # df.loc[loadfactor_idx-1] = [loadfactor, damage_idx, Entropy]

#%%
# DF_labels = ['Damage index']
# DF_labels.extend(Entropy_labels)

# DF = pd.DataFrame(columns=DF_labels)
# for i in range(len(df)):
#     vec = []
    
#     vec.append(df['Damage index'][i])
#     vec.extend(df['Entropy'][i])
    
#     DF.loc[i] = vec
    
# corr = DF.corr()

#%%

# markers = ['x', 'o']
# colors = ['red', 'green']


# plt.figure()
# for i in range(len(df['Entropy'][0])):
#     label_i = Entropy_labels[i]+'corr= %.2f' % corr.iloc[0,i+1]
#     plt.scatter(DF['Damage index'],DF.iloc[:,i+1],  color=colors[i], marker=markers[i], label=label_i)
# plt.title('Entropy \n GM: ' + load_file)
# plt.xlabel('Damage index')
# plt.ylabel('Entropy')
# plt.legend()
# plt.grid()


# fig, axs = plt.subplots(1, 2)
# for i in range(len(df['Entropy'][0])):
#     axs[i].scatter(DF['Damage index'],DF.iloc[:,i+1],  color=colors[i], marker=markers[i], label=label_i)
#     axs[i].set_title(Entropy_labels[i]+'corr= %.2f' % corr.iloc[0,i+1])
#     axs[i].set(xlabel='Damage index', ylabel='Entropy')
#     axs[i].grid()
# fig.suptitle('Entropy \n GM: ' + load_file)
# fig.tight_layout()


# DF.to_csv(r'el_centro_dataframe.csv')

# import sys
# sys.exit()









