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


def int_to_str3(list_int):
    
    '''
    Takes list of index (integers)
    Reurns list index in (string) 000 format
    E.g.:
        0   --> '000'
        20  --> '020'
        100 --> '100'
    '''
    
    list_str = []
    for i in list_int:
    
        i_str = str(i)
        
        if len(i_str) == 1:
            list_str.append( f'00{i_str}')
        elif len(i_str) == 2:
            list_str.append( f'0{i_str}')
        else:
            list_str.append( f'{i_str}')
            
    return list_str


folder_loads = os.path.join(os.getcwd(), 'import_loads\\Ground Motions')
output_directory = 'output_files'

df_structure = pd.read_pickle( os.path.join(output_directory, '00_Structure.pkl') )
struc_periods = list(df_structure.Periods[0])


plot_spectra = False



n = 0 # GM index

for rdirs, dirs, files in os.walk(folder_loads):
    
    
    
    # Parameters of the response spectra
    
    Resp_type = 'SA' # See above for the different options
    T = np.arange(0.01, 4, 0.01) # Time vector for the spectrum
    freq = 1/T # Frequenxy vector
    c = .01 # Damping factor
    
    # inizialize storing variables

    # S_acc = np.zeros((len(files), len(T)))   # spectra matrix
    # S_max = np.zeros(len(files)) # max acc vector - peak
    # T_max = np.zeros(len(files)) # max period vector - peak positon
    
    df =  pd.DataFrame(columns = ['Ground motion', 'dT', 'Peak acc', 'Peak T', 'Input time', 'Input acc', 'Spectra acc'])

    
    
    #Loops for n GM
    
    
    for file in files:
        if rdirs == folder_loads and ( file.endswith(".AT1") or file.endswith(".AT2") ):
                                           
            load_file = os.path.join(folder_loads, file)        # get file path
            desc, npts, dt, time, inp_acc = DamageTools.processNGAfile(load_file)  # get GM data
            delta = 1/dt                        # Time step of the recording in Hz
            S_acc = DamageTools.RS_function(inp_acc, delta, T, c, Resp_type = Resp_type)  # analysis - spectra
            
            # S_max[n] = max(S_acc[n, :])
            # T_max[n] = T[np.where(S_acc == S_max[n])[1][0]]
            
            df.loc[n] = [file[:-4], [dt], max(S_acc), T[np.where(S_acc == max(S_acc))[0][0]], time, inp_acc, S_acc]
            
            
            
            #%% plot
            
            if plot_spectra:
                
                i = n
                
                ID = int_to_str3([i])[0]
            
                fig = plt.figure(figsize = (10,12))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']), x=0.1, y=0.98, horizontalalignment='left', verticalalignment='top', fontweight='bold')
    
                ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
                ax2 = fig.add_axes([0.1, 0.425, 0.7, 0.25], sharex=ax1)
                ax3 = fig.add_axes([0.83, 0.425, 0.03, 0.25])
                ax4 = fig.add_axes([0.1, 0.08, 0.7, 0.25])
    
                #make time vector
                t = df.loc[i,'Input time']
    
    
                #plot waveform (top subfigure)    
                ax1.plot(t, df.loc[i,'Input acc'])
                ax1.grid()
                ax1.set_ylabel('Acc. [m/s\u00b2]')
                ax1.set_xlabel('Time [s]')
    
    
    
                #plot spectrogram (bottom subfigure)
                #spl2 = x
                Pxx, freqs, bins, im = ax2.specgram(df.loc[i,'Input acc'], Fs=1/df.loc[i,'dT'], cmap='jet_r') # remove cmap for different result
    
                # line colour is white
                # for periods in struc_periods:
                #     ax2.axhline(y = 1/periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--')
                #     ax2.text(t[0]-1.5, 1/periods, f'f$_{struc_periods.index(periods)+1}$', fontsize='small')
    
                mappable = ax2.images[0]
                plt.colorbar(mappable=mappable, cax=ax3, label='Amplitude [dB]')
                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Freq. [Hz]')
                
                # line colour is white
                for periods in struc_periods:
                    line = ax2.axhline(y = 1/periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--')
                    ax2.text(t[0]-1.5, 1/periods, f'{struc_periods.index(periods)+1}', fontsize='small')
                    line2 = ax4.axvline(x = periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--')
                    ax4.text(periods-0.01, df.loc[n, 'Peak acc']+0.1 ,f'{struc_periods.index(periods)+1}', fontsize='small')
                
                ax2.legend([line],['Struct. freq.'])
                ax4.legend([line2],['Struct. periods'])
    
                ax4.plot(T, df.loc[i,'Spectra acc'])
                ax4.grid(axis='y')
                ax4.set_ylabel('Acc. [m/s\u00b2]')
                ax4.set_xlabel('Period [s]')
    
                ax1.set_title(f'Ground motion')
                ax2.set_title(f'Spectogram')
                ax4.set_title(f'Acceleration spectrum')
                
                fig.savefig(os.path.join(output_directory, 'Figures', 'GM_spectra_ID', f'{ID}_{file[:-4]}.png'))
                
            
                
                
            
            
            n += 1
            
            

# Export dataframe

# df.to_pickle(output_directory + '/GM_spectra.pkl')
            
            
          