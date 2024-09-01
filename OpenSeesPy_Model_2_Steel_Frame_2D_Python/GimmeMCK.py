# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:41:51 2021

@author: lucag
"""

# https://portwooddigital.com/2020/05/17/gimme-all-your-damping-all-your-mass-and-stiffness-too/


import opensees.openseespy as ops
import numpy as np

def GimmeMCK():
    
    
    
    ops.wipeAnalysis()

    ops.system('FullGeneral') 
    ops.numberer('Plain')
    ops.constraints('Plain')        
    ops.algorithm('Newton')

    ops.test('NormDispIncr', 1.*10**-7, 100)
    ops.integrator('CentralDifference')
    ops.analysis('Transient')    


    # Mass
    ops.integrator('GimmeMCK',1.0,0.0,0.0)
 
    ops.analyze(1,0.0)
     
    # Number of equations in the model
    N = ops.systemSize() # Has to be done after analyze
     
    M = ops.printA('-ret') # Or use ops.printA('-file','M.out')
    M = np.array(M) # Convert the list to an array
    M.shape = (N,N) # Make the array an NxN matrix
     
    # Stiffness
    ops.integrator('GimmeMCK',0.0,0.0,1.0)
    ops.analyze(1,0.0)
    K = ops.printA('-ret')
    K = np.array(K)
    K.shape = (N,N)
     
    # Damping
    ops.integrator('GimmeMCK',0.0,1.0,0.0)
    ops.analyze(1,0.0)
    C = ops.printA('-ret')
    C = np.array(C)
    C.shape = (N,N)
    
    ops.wipeAnalysis()
    
    return M, C, K, N

def extract_K():
    
    
    
    ops.wipeAnalysis()

    ops.system('FullGeneral') 
    ops.numberer('Plain')
    ops.constraints('Plain')        
    ops.algorithm('Linear')

    ops.test('NormDispIncr', 1.*10**-7, 100)
    ops.integrator('CentralDifference')
    ops.analysis('Transient')    
    
    # Stiffness
    ops.integrator('GimmeMCK',0.0,0.0,1.0)
    ops.analyze(1,0.0)

    # Number of equations in the model
    N = ops.systemSize() # Has to be done after analyze


    K = ops.printA('-ret')
    K = np.array(K)
    K.shape = (N,N)
     
    ops.wipeAnalysis()
    
    return K

