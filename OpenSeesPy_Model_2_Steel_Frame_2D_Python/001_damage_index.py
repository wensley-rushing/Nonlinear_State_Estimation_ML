# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:29:09 2022

@author: lucag
"""



import openseespy.opensees as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


#from Model_definition_2D_frame import createModel
#from Model_definition_3x3_frame import createModel
#from Model_definition_2x1_frame import createModel

#from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord



# turn on/off the plots by setting these to True or False
plot_model = True
#plot_modeshapes = False
plot_dynamic_analysis = True


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
delta_y = 0.02

define_model = '1x1'

if define_model == '1x1':
    from Model_definition_1x1_frame import createModel
    
    # Global DI
    support_nodes = [10, 11]
    top_nodes = [20, 21]
    drift_nodes = [11, 21]
    
    #Local DI
    id_element = [1020, 1121]
    
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
    id_element = [1020]
#------------------------------------------------------------------------------
    

H1 = 3.5*m        # height first floor
L1 = 5.5*m        #m      length first span 
M = 1000 *kg 	  #kg		lumped mass at top corner nodes 
dampRatio = 0.02

df = pd.DataFrame(columns = ['Load factor', 'Damage index', 'Entropy'])


loadfactor_idx = 0
for loadfactor in [10,60]:
    loadfactor_idx = loadfactor_idx + 1
    print()
    print('Loadfactor: %.2f' %(loadfactor))
    
    if loadfactor_idx > 1:
        plot_model = False  
    

    
    load_file = 'el_centro.AT2'
    load_dat_file = 'el_centro.dat'
    
    
    # Import multiple loads
    if 1 > 2:
        import os
        
        # Getting the work directory of loads .AT1 or .AT2 files
        folder_loads = os.path.join(os.getcwd(), 'load_files')
        #r'C:\Users\larsk\Danmarks Tekniske Universitet\Thesis_Nonlinear-Damage-Detection\OpenSeesPy_Model_2_Steel_Frame_2D_Python\load_files'
        
        # r=root, d=directories, f = files
        for rdirs, dirs, files in os.walk(folder_loads):
            for file in files:
                if file.endswith(".AT1") or file.endswith(".AT2"):
                    #print(os.path.join(rdirs, file))
                    
                    #print(file)
                    
                    file_name = file[:-4]
                    #print(file_name)
                    
                    file_dat = file_name + '.dat'
                    #print(file_dat)
                    
                    
                    load_file = file
                    load_dat_file = file_dat
    
    
    # Move up in folder structure
    #from pathlib import Path
    #p = Path(__file__).parents[0]
    
    #print(p)
    # /absolute/path/to/two/levels/up
    
    
    
    # =============================================================================
    # # call function to create the model
    # =============================================================================
    
    node_vec, el_vec = createModel(H1,L1,M)

    
    if plot_model:
        plt.figure()
        opsv.plot_model()
        plt.title('model')
        #plt.show()  
    
    
    
    # =============================================================================
    # Run gravity Analysis
    # =============================================================================
    '''
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
    '''
     
    
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
    '''
    if plot_modeshapes:
        plt.figure()
        opsv.plot_mode_shape(1, sfac=1)
        plt.title('mode shape 1')
        #plt.show()
    
        plt.figure()
        opsv.plot_mode_shape(2, sfac=1)
        plt.title('mode shape 2')
        #plt.show()
    
    '''
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
    
    # ops.recorder('Node', '-file', output_directory+'/2_groundmotion_top_disp.out',
    #              '-time', '-node', max(ACC_Nodes),  '-dof', 1,  'disp')
    # ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_def.out',
    #              '-ele', 1020,  'section', 1,  'deformation')
    # ops.recorder('Element', '-file', output_directory+'/2_groundmotion_section_force.out',
    #              '-ele', 1020,  'section', 1,  'force')
    
    
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
    
    
    
    #%% 
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
    
    
    
    
    if plot_dynamic_analysis:
        ops.wipe() # to close recorders
        
        # Retrive data
        base_shear = np.loadtxt(output_directory+'/2_Reaction_Base.out')
        total_base_shear = -np.sum(base_shear,axis=1)
        
        time_drift_disp = np.loadtxt(output_directory+'/2_Dsp_Drift_Nodes.out')
        
        #time_topDisp = np.loadtxt(output_directory+'/2_groundmotion_top_disp.out')
        #sectionDef = np.loadtxt(output_directory+'/2_groundmotion_section_def.out')
        #sectionForce = np.loadtxt(output_directory+'/2_groundmotion_section_force.out')
                                  
        # Top displacement over time
        plt.figure()
        plt.plot(time_drift_disp[:,0],time_drift_disp[:,len(drift_nodes)])
        plt.title('Dynamic analysis \n GM: ' + load_file)
        plt.xlabel('Time (s)')
        plt.ylabel('Top displacement node %.0f (m)' %(drift_nodes[-1]))
        plt.grid()
        #plt.show()
        
        # Hysterises loop Global (Base shear vs. top disp)
        plt.figure()
        plt.plot(time_drift_disp[:,len(drift_nodes)],total_base_shear/1000)
        plt.title('Dynamic analysis - Structure \n GM: ' + load_file)
        plt.xlabel('Roof displacement (m)')
        plt.ylabel('Total base shear (kN)')
        plt.grid()
        #plt.show()
        
        
        #%% Energy (Global)
        
        # Damage index information of interest
        DI_x = time_drift_disp[:,len(drift_nodes)] #Deformation (curvature) 
        DI_y = total_base_shear/1000 #Force (Moment)
        
        
        Energy_g = np.trapz(DI_y, x=DI_x)
        print('Energy - Global: %.4f' %(Energy_g))
        
        corr = np.corrcoef(DI_x, DI_y)
        Corr = corr[0][1]
        print('Correlation: %.4f' %(Corr))
        
        
        
        
        
        #%% Local ---------------
        element_section_forces = np.loadtxt(output_directory+'/2_Section_Force.out')
        element_section_defs = np.loadtxt(output_directory+'/2_Section_Def.out')
        
        for el_id in range(len(id_element)):
            
            plt.figure()
            plt.plot(element_section_defs[:,(el_id*2)+1],element_section_forces[:,(el_id*2)+1]/1000)
            plt.title('Dynamic analysis - Element ' + str(id_element[el_id]) + ' \n GM: ' + load_file )
            plt.xlabel('Curvature (-)')
            plt.ylabel('Moment (kNm)')
            plt.grid()
            
            
            # Damage index information of interest
            DI_x = element_section_defs[:,(el_id*2)+1] #Deformation (curvature) 
            DI_y = element_section_forces[:,(el_id*2)+1]/1000 #Force (Moment)
            
            
            Energy_l = np.trapz(DI_y, x=DI_x)
            print('Energy - Element %.0f: %.4f' %(id_element[el_id],Energy_g))
            
            corr = np.corrcoef(DI_x, DI_y)
            Corr = corr[0][1]
            print('Correlation - Element %.0f: %.4f' %(id_element[el_id],Corr))
            
            
            #max_plasic_deforms = plastic_deform[-1:].tolist()
            #max_plasic_deform = max(max_plasic_deforms[0][1:], key=abs)
            #print('Max plastic defomation, el_%.0f: %0.4f' %(id_element[el_id], max_plasic_deform))
    
        #%% Energy (Local)


    # =============================================================================
    # Labelling
    # =============================================================================

    
    #%% Plastic deformation (Local)
    plastic_deform = np.loadtxt(output_directory+'/2_Plastic_Def.out')
    
    for el_id in range(len(id_element)):
        plt.figure()
        for i in range((el_id*3)+1,(el_id*3)+4):
            plt.plot(plastic_deform[:,0],plastic_deform[:,i], label=str(i))
        plt.title('Plastic Deformation - Element ' + str(id_element[el_id]) + ' \n GM: ' + load_file )
        plt.xlabel('Time (s)')
        plt.ylabel('Plastic deformation (-)')
        plt.legend()
        plt.grid()
        
        
        max_plasic_deforms = plastic_deform[-1:].tolist()
        max_plasic_deform = max(max_plasic_deforms[0][1:], key=abs)
        print('Max plastic defomation, el_%.0f: %0.4f' %(id_element[el_id], max_plasic_deform))

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









