# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:19:29 2022

@author: lucag
"""



import openseespy.opensees as ops

N = 1
m = 1



def runGravityAnalysis(beam_vec):
    
    roof_beam = beam_vec[-3:len(beam_vec)]
    
    floor_beam = [i for i in beam_vec if i not in roof_beam]
    
    
    ops.timeSeries('Linear', 100)     #create timeSeries with tag 100
    
    #pattern(   'Plain', patternTag, tsTag)
    ops.pattern('Plain', 100,        100 )
    
    #eleLoad('-ele', *eleTags, '-type', '-beamUniform', Wy)
    ops.eleLoad('-ele', *roof_beam, '-type', '-beamUniform', -7500*N/m)      # 140 N/m of uniformly distributed load on the beam
    ops.eleLoad('-ele', *floor_beam, '-type', '-beamUniform', -9500*N/m)      # 140 N/m of uniformly distributed load on the beam
    

    # ---- Create Analysis
    ops.constraints('Plain')      		    #objects that handles the constraints
    ops.numberer('Plain')					#objects that numbers the DOFs
    ops.system('FullGeneral')				#objects for solving the system of equations
    ops.test('NormDispIncr', 1e-12, 100)    #convergence test, defines the tolerance and the number of iterations
    ops.algorithm('Newton') 				#algorithm for solving the nonlinear equations


    # run analysis in 10 steps
    n_steps_gravity = 10
    
    ops.integrator('LoadControl',1/n_steps_gravity)
    
    ops.analysis('Static')    #creates a static analysis
    
    ok = ops.analyze(n_steps_gravity)
    

    if ok == 0: print("Gravity analysis: SUCCESSFULL")
    else: print("!! -- Gravity analysis: FAILED -- !!")
    
    
    return

