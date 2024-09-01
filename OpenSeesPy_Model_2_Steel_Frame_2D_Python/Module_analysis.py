# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:50:45 2022

@author: Gabriele and Lars
"""


import opensees.openseespy as ops
import opsvis as opsv


import matplotlib.pyplot as plt
import numpy as np


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

def pushover_analysis(H1, L1, M, Dmax, Dincr, nodes, plot_defo_Pushover):

    # nodes = [node_supp1, node_supp2, node_load]

    # Define Recorders
    
    output_directory = 'output_files'
    ops.recorder('Node', '-file', output_directory+'/1_Pushover_top_disp.out',
                 '-node', nodes[2],  '-dof', 1,  'disp')
    ops.recorder('Node', '-file', output_directory+'/1_Pushover_base_reactions.out',
                 '-node', nodes[0],nodes[1],  '-dof', 1,  'reaction')


    # Define static load
    # ------------------

    ops.timeSeries('Linear', 1)     #create timeSeries with tag 1

    #pattern(   'Plain', patternTag, tsTag)
    ops.pattern('Plain', 1,        1 )

    #load   (nodeTag, *loadValues)    loadValues in x direction, y dir, moment
    ops.load(nodes[2]     , *[1,0,0]  )  # load in x direction in node 20


    # ---- Create Analysis
    ops.constraints('Plain')      		    #objects that handles the constraints
    ops.numberer('Plain')					#objects that numbers the DOFs
    ops.system('FullGeneral')				#objects for solving the system of equations
    ops.test('NormDispIncr', 1e-12, 100)    #convergence test, defines the tolerance and the number of iterations
    ops.algorithm('Newton') 				#algorithm for solving the nonlinear equations


    #integrator('DisplacementControl',   nodeTag, dof, incr)
    ops.integrator('DisplacementControl', nodes[2],      1 ,  Dincr)

    ops.analysis('Static')    #creates a static analysis


    # ---- Analyze model
    Nsteps = int(Dmax/Dincr); # number of pushover analysis steps
    print(f"number of steps Pushover: {Nsteps}")


        
    ok = ops.analyze(Nsteps)

    

    
    if ok == 0: print("-----------------Pushover analysis successfully run--------------------")
        
    
    delta_y = 0.2
    delta_u = 0.5

    plt.figure()
    opsv.plot_defo(sfac = 1) 
    plt.title('deformed shape - Pushover analysis')
    plt.show()  




    #%%

    # =============================================================================
    # plot pushover results
    # =============================================================================





    ops.wipe() # to close recorders

    p_curve_x = np.loadtxt(output_directory+'/1_Pushover_top_disp.out')
    Pushover_reactions = np.loadtxt(output_directory+'/1_Pushover_base_reactions.out')
    
    p_curve_y = -np.sum(Pushover_reactions,axis=1)/1000
    
    x = [p_curve_x[0],p_curve_x[1]]
    y = [p_curve_y[0],p_curve_y[1]]
    slope = abs(y[0]-y[1]) / abs(x[0]-x[1])
    
    F_max  = abs(max(p_curve_y))
    F_max_index = np.where(p_curve_y == F_max)[0][0]
    
    delta_y = (0.8)*(F_max) / (slope)
    delta_u = p_curve_x[F_max_index]
    

    if plot_defo_Pushover:
        plt.figure()
        plt.plot(np.insert(p_curve_x, 0, 0),np.insert(p_curve_y, 0, 0))   #inserts 0 at the beginning
        plt.plot(delta_y, 0.8*F_max, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        plt.plot(delta_u, F_max, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        plt.title('Pushover curve')
        plt.xlabel('displacement top (m)')
        plt.ylabel('total base shear (kN)')
        plt.grid()
        plt.show()

    
    return delta_y, delta_u, p_curve_x, p_curve_y

