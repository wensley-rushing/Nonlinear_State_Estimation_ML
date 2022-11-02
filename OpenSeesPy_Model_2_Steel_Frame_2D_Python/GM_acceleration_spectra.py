# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:54:05 2022

@author: Gabriele and Lars

- Resp_type: Response type, choose between:
             - 'SA'  : Acceleration Spectra
             - 'PSA' : Pseudo-acceleration Spectra
             - 'SV'  : Velocity Spectra
             - 'PSV' : Pseudo-velocity Spectra
             - 'SD'  : Displacement Spectra

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import DamageTools


folder_loads = os.path.join(os.getcwd(), 'import_loads\\5_eartq')
plot_spectra = True

n = 0 # GM index

for rdirs, dirs, files in os.walk(folder_loads):
    
    
    
    # Parameters of the response spectra
    
    Resp_type = 'SA' # See above for the different options
    T = np.arange(0.01, 4, 0.01) # Time vector for the spectrum
    freq = 1/T # Frequenxy vector
    c = .01 # Damping factor
    
    # inizialize storing variables

    S_acc = np.zeros((len(files), len(T)))   # spectra matrix
    S_max = np.zeros(len(files)) # max acc vector - peak
    T_max = np.zeros(len(files)) # max period vector - peak positon
    
    
    #%% Loops for n GM
    
    for file in files:
        if rdirs == folder_loads and ( file.endswith(".AT1") or file.endswith(".AT2") ):
                                           
            load_file = os.path.join(folder_loads, file)        # get file path
            desc, npts, dt, time, inp_acc = DamageTools.processNGAfile(load_file)  # get GM data
            delta = 1/dt                        # Time step of the recording in Hz
            S_acc[n, :] = DamageTools.RS_function(inp_acc, delta, T, c, Resp_type = Resp_type)  # analysis - spectra
            
            S_max[n] = max(S_acc[n, :])
            T_max[n] = T[np.where(S_acc == S_max[n])[1][0]]
            
            #%% plot
            
            
            if plot_spectra: 
                
                fig = plt.figure(figsize = (6,8))
                plt.subplot(2,1,1)
                plt.plot(time, inp_acc) 
                plt.grid()
                plt.title('Earthquake ' + str(file))
                plt.ylabel('Acceleration (m/s\u00b2)')
                plt.xlabel('Time (s)')
                plt.xlim(time[0], time[-1])



                plt.subplot(2,1,2)
                plt.semilogy(T, S_acc[n,:])
                plt.title("Acceleration spectra")
                plt.scatter(T_max[n],S_max[n], marker = 'x', color = 'r')
                plt.grid()
                if Resp_type == 'SA':
                    plt.ylabel('Acceleration response (m/s\u00b2)')
                elif  Resp_type == 'SV':
                    plt.ylabel('Velocity response (m/s)')
                else:
                    plt.ylabel(Resp_type)
                plt.xlabel('Period (s)')
                plt.xlim(T[0],T[-1])
                
            
                plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
                
                
            
                
                
            
            #%%
            n += 1
            
            
            
            
          