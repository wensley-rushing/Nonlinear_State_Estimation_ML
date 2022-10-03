# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:48:48 2022

DamageLabelTools
This modulus includes functions in orer quantify damages.

The functions include:
    CorrFuncs: Calculate correlation functions for given input time signals
    SampEn:    Calulate sample Entroty for given signals
    

@author: larsk
"""

#%% IMPORTS

import numpy as np
import matplotlib as plt



#%% Functions

'''
SamEn: Sample Entropy
Calulates the sample entrity from a given time signal X

SampEn is the negative natural logarithm of the probability 
 that if two sets of simultaneous data points of length m have distance <r, 
 then two sets of  simultaneous data points of length m+1 also have distance <r.


Inputs:
    X: Time signal X = [x1, x2, x3, x4, ... xN], Aray of float64
    m: Embedding dimension
    r: Chebyshev distance

        
Outputs:
    sample_entropy
'''


def SampEn(X, m, r):
    N = len(X)
    B = 0.0
    A = 0.0
    
    
    # Split time series and save all templates of length m
    xmi = np.array([X[i : i + m] for i in range(N - m)])
    xmj = np.array([X[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([X[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    sample_entropy = -np.log(A / B)
    return sample_entropy


#%%

'''
CorrFuncs: Corelation Functions
Calulate the correlation matrix R(tau) for a given time-lag, tau

The correlation matrix R is the ...


Inputs:
    X: [Array of float64], Time signal X = [x1, x2, x3, x4, ... xN] 
    m: [Integer], Increment of time-lag, tau = m dt. m in [0..N]

        
Outputs:
    R
'''

def CorrFuncs(X, m):
    N = len(X)  # Number of time stamps
    M = np.size(X,1) # Number of signals
    
    
   
    
    
def Yielding_point(x, y):
    
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    point_x = [x[0],x[1]]
    point_y = [y[0],y[1]]
    K_i = abs(y[0]-y[1]) / abs(x[0]-x[1])

    loc_min = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1
    loc_min = loc_min[-1]

    linear_y = np.arange(0, y[loc_min], 10)
    linear_x = linear_y/K_i

    dim = len(linear_y)

    # plt.figure()
    # plt.plot(x,y)
    # plt.plot(linear_x, linear_y)
    # plt.grid()
    # plt.show()

    for i in range(1, dim):  
        
        bilin_x = [0, linear_x[dim-i],x[loc_min]]
        bilin_y = [0, linear_y[dim-i], y[loc_min]]
        
        # plt.figure()
        # plt.plot(x,y)
        # plt.plot(bilin_x, bilin_y)
        # plt.title('Case %.0f: ' %(i))
        # plt.grid()
        # plt.show()
        
        # bilinear curve area
        
        A_tot_bilin = np.trapz(bilin_y, x=bilin_x)
        A_1 = np.trapz(bilin_y[:2], x=bilin_x[:2])
        A_2 = A_tot_bilin - A_1
        
        # real curve area
        
        A_tot_real = np.trapz(y[:loc_min+1], x=x[:loc_min+1])
       # A_3 = np.trapz(y[:loc_min+1], x=x[:loc_min+1])
        
        diff = A_tot_bilin - A_tot_real
        
        # print('Case %.0f: %.1f-%.1f = %.1f' %(i,A_tot_bilin, A_tot_real , diff))
        
        if abs(diff) < 0.1:
            print('Yielding point: (%.4f, %.1f)' %(linear_x[dim-i], linear_y[dim-i]))
            print('Ultimate resistance point: (%.4f, %.1f)' %(x[loc_min], y[loc_min]))
            break
        
    # print('Out of the loop')
    return linear_x[dim-i], linear_y[dim-i], x[loc_min], y[loc_min] # yielding deformation and moment 
                                                                        # and ultimate point (def and force)
    





