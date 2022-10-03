# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""

#%% IMPORTS

# Import OpenSees
import openseespy.opensees as ops
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


#%% #%% Enable/Disable plots
# turn on/off the plots by setting these to True or False

# General structure
plot_model = False
plot_defo_gravity = False
plot_modeshapes = False

# Dynamic analysis
plot_dynamic_analysis = False


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
    

H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 6000 *kg 	  #kg		lumped mass at top corner nodes 
dampRatio = 0.02



#%% Initialization 

# Create Dataframe for results
gm_idx = 0
df = pd.DataFrame(columns = ['OK=0', 'Ground motion', 'Load factor', 
                             'E - glob', 'Gl Drift', 'Gl Drift - class', 
                             'Element ID', 'Section ID (E el.)', 'E el.', 'Max. Pla. def. el.', 'Res. Pla. def. el.'])
 
#%% Time - tik
global_tic_0 = time.time()

#%% Gravity Analysis

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

omega_sq = ops.eigen('-fullGenLapack', 3); # eigenvalue mode 1, 2, and 3
omega = np.array(omega_sq)**0.5 # circular natural frequency of modes 1 and 2

periods = 2*np.pi/omega

print(f'Period(s) of structure [s]: {np.around(periods, decimals = 4)}')


# =============================================================================
# Plot modeshapes
# =============================================================================

if plot_modeshapes:
    for i in range(int(len(periods))):
        plt.figure()
        opsv.plot_mode_shape(i+1, sfac=100)
        plt.title(f'mode shape {i+1}')
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
ops.database('File', 'DataBase\\3x3-Initial')
# Created the copy
ops.save(199)

#%% Time - General
local_last = time.time()
global_tic_1 = time.time()
print('General analysis: %.4f [s]' %(global_tic_1 - global_tic_0 ))
print()

#sys.exit()
#%% Import data    
# Import multiple loads

# Getting the work directory of loads .AT1 or .AT2 files
folder_loads = os.path.join(os.getcwd(), 'import_loads\\TT')
#r'C:\Users\larsk\Danmarks Tekniske Universitet\Thesis_Nonlinear-Damage-Detection\OpenSeesPy_Model_2_Steel_Frame_2D_Python\load_files'

# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_loads):
    for file in files:
        if rdirs == folder_loads and ( file.endswith(".AT1") or file.endswith(".AT2") ):
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
                
                #%% DYNAMIC ANALYSIS
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
                              '-time', '-ele', *id_element, 'plasticDeformation ') # [Axial, RotationalI, RotationalJ]
                
                # Section deformation () (local)
                ops.recorder('Element', '-file', output_directory+'/2_Section_Force.out',
                             '-ele', *id_element,  'section',   'force')      # [Fx, Mx]
                
                ops.recorder('Element', '-file', output_directory+'/2_Section_Def.out',
                             '-ele', *id_element,  'section',   'deformation') # [axial-strain, curvature]
                
                
                
                # Recorder files for all nodes
                if len(str(gm_idx)) == 1:
                    mod_gm_idx = f'00{gm_idx}'
                elif len(str(gm_idx)) == 2:
                    mod_gm_idx = f'0{gm_idx}'
                        
                ops.recorder('Node', '-file', output_directory+f'/ACCS/ID_{mod_gm_idx}_Time_Node_Accs.out',
                            '-time', '-node', *node_vec,  '-dof', 1,  'accel')
                
                
                
                
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
                
                
                if ok == 0: print("Dynamic analysis: SUCCESSFULL")
                else: print(f"!! -- Analysis FAILED @ time {current_time} -- !!"); ok = -1
                

            
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
                    
                #%% Global Damage Index 
                print('-- Global Damage Index:')
                #%% --Energy (Global)
                
                # Damage index information of interest
                DI_x = time_drift_disp[:,-1] #Deformation at roof top (curvature) 
                DI_y = total_base_shear/1000 #Force (Moment)
                
                
                Energy_G = np.trapz(DI_y, x=DI_x)
                print('---- Energy - Global: %.4f' %(Energy_G))
                
                #corr = np.corrcoef(DI_x, DI_y)
                #Corr = corr[0][1]
                #print('Correlation: %.4f' %(Corr))
                    
                    
                #%% --Interstorey drift (Global)
        
                n_floors = len(time_drift_disp[0]) - 2 # do not count the columns for time and base node 
                
                drift = []
                inter_drift = []
                inter_time_drift = []
                for i in range (0, n_floors):
                    drift.append(abs(time_drift_disp[-1, i+2]) / (3*H1)) # Residual drift
                    inter_drift.append( (abs(time_drift_disp[-1, i+2])  -  abs(time_drift_disp[-1, i+1])) / H1 ) # Residual Inter drift
                    inter_time_drift.append( abs(max(time_drift_disp[:,i+2]-time_drift_disp[:,i+1], key=abs)) / H1 ) # Max Inter drift
                    
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
                
                print('---- Max drift: ' + str(round(max_drift,4)))
                print('---- Max inter. drift: ' + str(round(max_inter_drift,4))  + ' - Class: ' + drift_cl)
                print('---- Max inter. time drift: ' + str(round(max_inter_time_drift,4))  + ' - Class: ' + drift_time_cl)
                    
                
                #%% --Park-Ang (Global)
                # DI = (Dm - Dy)/(Du - Dy) + beta E/(Fy Du)
                # Based on Article: 1.Park-Ang Damage Index-Based Framework ...
                
                PA_beta = 0.15
                
                PA_Dm = abs(max(time_drift_disp[:,-1], key=abs)) # Maximal roof displacement
                PA_E = Energy_G
                
                PA_Dy = 0.5*PA_Dm # Yiels deformation (Estimated)
                PA_Du = 1.5*PA_Dm # Ultimate deformation (Estimated)
                PA_Fy = 10000 # Yield strengh (Estimated)
                
                PA_G = (PA_Dm - PA_Dy)/(PA_Du - PA_Dy) + PA_beta*PA_E/(PA_Fy*PA_Du)
                
                
                
                # Table 1
                if PA_G < 0.11:
                    PA_G_cl = 'Minor'
                elif PA_G < 0.4:
                    PA_G_cl = 'Moderate'
                elif PA_G < 0.77:
                    PA_G_cl = 'Severe'
                elif PA_G >= 0.77:
                    PA_G_cl = 'Collapse'
               
                
                
                print('---- Park-Ang Global: %.4f -- Damage Level: ' %(PA_G) + PA_G_cl)
                
                
                    
                #%% Local - Damage Index
                print('-- Local Damage Index:')
                #%% --Energy + Moment-Curvature Plot (Local)
                
                # Moment / Curvature (NO TIME)
                element_section_forces = np.loadtxt(output_directory+'/2_Section_Force.out') # [Fx, Mx]
                element_section_defs = np.loadtxt(output_directory+'/2_Section_Def.out') # [Axial Strain, curvature]
                
                Energy_L = []
                Energy_L_sec = []
                
                
                # Park-Ang
                # DI = (Dm - Dy)/(Du - Dy) + beta E/(Fy Du) , Dy = 0
                # Based on Article: 2. Performance-based earthquake engineering design of ...
                PA_beta = 0.15
                
                #PA_Dm = abs(max(time_drift_disp[:,-1], key=abs)) # Maximal roof displacement
                #PA_E = Energy_G
                
                PA_Dy = 0 # Yiels deformation (Estimated)
                PA_Du = 100 # Ultimate deformation (Estimated)
                PA_Fy = 1000 # Yield strengh (Estimated)
                
                PA_L = []
                PA_L_sec = []
               
                
               
                # Assume same number of integration points for ALL elements
                # Number if elements
                num_el = len(id_element)
                # Number of sections in total
                num_sec = int(element_section_forces.shape[1])
                # Number of integration points with 2 walues as output [Fx, Mx]
                num_int = int( num_sec/(num_el*2) )
                
                
                for el_id in range(int(num_el)):
                    
                    Energy_l_sec = []
                    PA_l_sec = []
                    for sec_id in range(int(num_int)):

                        # Damage index information of interest
                        DI_x = element_section_defs[:,(sec_id*2)+1 + (2*num_int)*el_id] #Deformation (curvature) 
                        DI_y = element_section_forces[:,(sec_id*2)+1 + (2*num_int)*el_id]/1000 #Force (Moment)
                        
                        
                        Energy_l = np.trapz(DI_y, x=DI_x)
                        #print('Energy - Element %.0f, Section %.0f: %.4f' %(id_element[el_id], sec_id+1, Energy_l))
                        Energy_l_sec.append(Energy_l)
                        
                        
                        #'''
                        #Park-Ang
                        #'''
                        PA_Dm = abs(max(DI_x, key=abs))
                        PA_l = (PA_Dm - PA_Dy)/(PA_Du - PA_Dy) + PA_beta*Energy_l/(PA_Fy*PA_Du)
                        PA_l_sec.append(PA_l)
                        
                        
                        
                        
                    Energy_L.append( max(Energy_l_sec) )
                    Energy_L_sec.append( Energy_l_sec.index( max(Energy_l_sec) ) + 1 )
                    
                    print('---- Max energy - Element %.0f, Section %.0f: %.4f' %(id_element[el_id], Energy_L_sec[el_id], Energy_L[el_id]))
                    
                    if plot_dynamic_analysis:
                        plt.figure()
                        plt.plot(DI_x,DI_y)
                        plt.title('Dynamic analysis: Max E - Element ' + str(id_element[el_id]) + ' Section ' + str(Energy_L_sec[el_id]) + 
                                  ' \n GM: ' + file_name + ' -  Loadfactor: ' + str(loadfactor) )
                        plt.xlabel('Curvature (-)')
                        plt.ylabel('Moment (kNm)')
                        plt.grid()  
                        
                        
                        
                    PA_L.append( max(PA_l_sec) )
                    PA_L_sec.append( PA_l_sec.index( max(PA_l_sec) ) + 1 )
                    
                    # Table 1
                    if PA_L[el_id] < 0.1:
                        PA_L_cl = 'No Damage' 
                    elif PA_L[el_id] < 0.2:
                        PA_L_cl = 'Minor'
                    elif PA_L[el_id] < 0.5:
                        PA_L_cl = 'Moderate'
                    elif PA_L[el_id] < 1:
                        PA_L_cl = 'Severe'
                    elif PA_L[el_id] >= 1:
                        PA_L_cl = 'Collapse'
                    
                    
                    
                    
                    print('---- Max PA_L - Element %.0f, Section %.0f: %.4f' %(id_element[el_id], PA_L_sec[el_id], PA_L[el_id]) + ' Damage: ' + PA_L_cl)
                    
    
                #%% --Plastic Deformation (Local)
            
                # Plastic deformation (+ Time)
                plastic_deform = np.loadtxt(output_directory+'/2_Plastic_Def.out')
                
               
                # Assume same number of integration points for ALL elements
                
                # Number if elements
                num_el = len(id_element)
                # Number of values in total, excludeing time
                num_val = 3
                
                # Initialize
                Res_plastic_deform = []
                Max_plastic_deform = []
                
                
                for el_id in range(num_el):
                    
                    if plot_dynamic_analysis:
                        plt.figure()
                        plt.plot(plastic_deform[:,0],plastic_deform[:,(el_id*3)+1], label= 'Axial Deformation')
                        plt.plot(plastic_deform[:,0],plastic_deform[:,(el_id*3)+2], label= 'Rotation Node: ' + str(id_element[el_id])[:2])
                        plt.plot(plastic_deform[:,0],plastic_deform[:,(el_id*3)+3], label= 'Rotation Node: ' + str(id_element[el_id])[2:])
                            
                        plt.title('Plastic Deformation - Element ' + str(id_element[el_id]) + ' \n GM: ' + file_name + ' -  Loadfactor: ' + str(loadfactor) )
                        plt.xlabel('Time (s)')
                        plt.ylabel('Plastic deformation (-)')
                        plt.legend()
                        plt.grid()
                    
                    
                    # Permanent plastic deformation (At time = end) - Residual
                    res_plastic_deforms = plastic_deform[-1,(el_id*3)+1:(el_id*3)+4].tolist()
                    res_plastic_deform = max(res_plastic_deforms, key=abs)
                    print('---- Residual plastic defomation, Element %.0f: %0.4e' %(id_element[el_id], res_plastic_deform))
                    Res_plastic_deform.append(res_plastic_deform)
                    
                    
                    # Maximal plastic deformation (At time = t)
                    max_plastic_deforms = [max(plastic_deform[:,(el_id*3)+1]), min(plastic_deform[:,(el_id*3)+1]),
                                           max(plastic_deform[:,(el_id*3)+2]), min(plastic_deform[:,(el_id*3)+2]),
                                           max(plastic_deform[:,(el_id*3)+3]), min(plastic_deform[:,(el_id*3)+3])]
                    max_plastic_deform = max(max_plastic_deforms, key=abs)
                    print('---- Max plastic defomation, Element %.0f: %0.4e' %(id_element[el_id], max_plastic_deform))
                    Max_plastic_deform.append(max_plastic_deform)
                    
                    
                
                    
            
            #%% Record data in the dataframe
            
                
                df.loc[gm_idx] = [ok, file_name, loadfactor, 
                                  Energy_G, max_inter_drift, drift_cl,
                                  id_element, Energy_L_sec, Energy_L, Max_plastic_deform, Res_plastic_deform]
                gm_idx += 1
                
                #%% Time - Dynmic loops
                
                local_toc = time.time()
                   
                print('Lap time: %.4f [s]' %(local_toc - local_last ))              
                print('Total duration: %.4f [s]' %(local_toc - global_tic_0 ))
                print()
                
                local_last = time.time()
#%% Time  Estimations
global_toc = time.time()

print('Total for %.0f analyses: %.4f [s]' %(df.shape[0], global_toc - global_tic_0 ))
total_analyses = 300
print('Estimat for %.0f analyses: %.4f [s]' %(total_analyses, (global_toc - global_tic_0)/df.shape[0]*total_analyses ))
print('-- Minutes: %.4f [min]' %( (global_toc - global_tic_0)/df.shape[0]*total_analyses/60 ))
print('-- Hours: %.4f [hrs]' %( (global_toc - global_tic_0)/df.shape[0]*total_analyses/60760 ))

#%% Export dataframe

df.to_csv(output_directory + r'/00_Index_Results.csv')  # export dataframe to cvs
df.to_pickle(output_directory + "/00_Index_Results.pkl") 
#unpickled_df = pd.read_pickle("./dummy.pkl")  

Structure = pd.DataFrame(columns = ['Nodes'])
Structure['Nodes'] = node_vec
Structure.to_pickle(output_directory + "/00_Structure.pkl") 

 
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









