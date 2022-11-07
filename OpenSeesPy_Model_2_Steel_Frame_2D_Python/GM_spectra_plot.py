# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:31:56 2022

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import pickle

output_directory = 'output_files'
folder_loads = os.path.join(os.getcwd(), 'import_loads\\5_eartq')
plot_spectra = True


T = np.arange(0.01, 4, 0.01) # Time vector for the spectrum

   

df = pd.read_pickle( os.path.join(output_directory, 'GM_spectra.pkl') )    
df_structure = pd.read_pickle( os.path.join(output_directory, '00_Structure.pkl') )
struc_periods = list(df_structure.Periods[0])

plot_earth = ['BIGBEAR_DHP090', 'CHICHI_CHY101-N']
plot_index = []

for j in plot_earth:
    for i in df.index:
        if df.loc[i, 'Ground motion'] == j:
            plot_index.append(i) 
            if plot_spectra: 
                
                fig = plt.figure(figsize = (10,12))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']), x=0.1, y=1, horizontalalignment='left', verticalalignment='top', fontweight='bold')

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
                    line = ax2.axhline(y = 1/periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--', label = 'Struct. periods')
                    # ax2.text(t[0]-1.5, 1/periods, f'f$_{struc_periods.index(periods)+1}$', fontsize='small')
                
                ax2.legend([line],['Struct. periods'])
                

                ax4.plot(T, df.loc[i,'Spectra acc'])
                ax4.grid()
                ax4.set_ylabel('Acc. [m/s\u00b2]')
                ax4.set_xlabel('Period [s]')

                ax1.set_title(f'Ground motion')
                ax2.set_title(f'Spectogram')
                ax4.set_title(f'Acceleration spectrum')