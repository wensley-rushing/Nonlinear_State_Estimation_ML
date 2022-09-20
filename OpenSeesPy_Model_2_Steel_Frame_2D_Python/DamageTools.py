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
    
    
   
    return sample_entropy