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

from numpy.fft import fft, ifft


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


folder_loads = os.path.join(os.getcwd(), 'import_loads\\GM_Report')
output_directory = r'output_files'

df_structure = pd.read_pickle( os.path.join(output_directory, '00_Structure.pkl') )
struc_periods = list(df_structure.Periods[0])

folder_save_plots = r'output_files\Figures\GM_Report'


plot_spectra = False

plot_GM = True

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
            if True: # Tm
                i = n
                
                ID = int_to_str3([i])[0]
            
                fig = plt.figure(figsize = (14,10))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']) + f' - ID:{ID}' , fontweight='bold', fontsize = 16)
    
                ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.38]) #[left bottom width height]
                # ax2 = fig.add_axes([0.1, 0.425, 0.7, 0.25], sharex=ax1)
                # ax3 = fig.add_axes([0.83, 0.425, 0.03, 0.25])
                ax4 = fig.add_axes([0.1, 0.08, 0.8, 0.38])
    
                #make time vector
                t = df.loc[i,'Input time']
    
    
                #plot waveform (top subfigure)    
                ax1.plot(t, df.loc[i,'Input acc'])
                ax1.grid()
                ax1.set_ylabel('Acceleration [g]', fontsize = 16)
                ax1.yaxis.set_tick_params(labelsize=14)
                
                ax1.set_xlabel('Time [s]', fontsize = 16)
                ax1.xaxis.set_tick_params(labelsize=14)
                
                
                #-------------------------------------------
                amplitude = df.loc[i,'Input acc']
                samplingFrequency = 1/(t[1]-t[0])
                
                fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
                fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
                tpCount     = len(amplitude)
                values      = np.arange(int(tpCount/2))
                timePeriod  = tpCount/samplingFrequency
                frequencies = values/timePeriod

                a = 0.25 <= frequencies
                frequencies0 = frequencies[a]
                fourierTransform0 = abs(fourierTransform)[a]
                
                a = frequencies0 <= 20
                frequencies0 = frequencies0[a]
                fourierTransform0 = abs(fourierTransform0)[a]
                
                top = (fourierTransform0**2 / frequencies0).sum()
                bot = (fourierTransform0**2).sum()
                Tm = round(top/bot,2)

                
                if False:    # Fourier Transform            
                    ax4.plot(frequencies, abs(fourierTransform))
                    ax4.set_xlim(0,30)
                    ax4.grid(axis='y')
                    ax4.set_ylabel('FAS [g/Hz]', fontsize = 16)
                    ax4.yaxis.set_tick_params(labelsize=14)
                    
                    ax4.set_xlabel('Frequency [Hz]' , fontsize = 16)
                    ax4.xaxis.set_tick_params(labelsize=14)
        
                    ax1.set_title(f'Ground motion' ,fontweight='bold', fontsize = 16)
                    # ax2.set_title(f'Spectogram')
                    ax4.set_title(f'Acceleration spectrum',  fontweight='bold', fontsize = 16)
                    
                if True: # Spectra
                
                    line = ax4.axvline(x = Tm, color = 'black', alpha = 1, linewidth=1.2, linestyle = '--')
                    ax4.text(Tm+0.02, min(df.loc[i,'Spectra acc']), f'T_m = {round(Tm,2)} s', rotation = 90,
                             ha = 'left', va = 'bottom', fontweight='bold', fontsize = 14)
                
                
                
                    ax4.plot(T, df.loc[i,'Spectra acc'])
                    ax4.grid(axis='y')
                    ax4.set_ylabel('Spectral acceleration [g]', fontsize = 16)
                    ax4.yaxis.set_tick_params(labelsize=14)
                    
                    ax4.set_xlabel('Period [s]' , fontsize = 16)
                    ax4.xaxis.set_tick_params(labelsize=14)
        
                    ax1.set_title(f'Ground motion' ,fontweight='bold', fontsize = 16)
                    # ax2.set_title(f'Spectogram')
                    ax4.set_title(f'Acceleration spectrum',  fontweight='bold', fontsize = 16)
                    
                
                plt.close()
                
                fig.savefig(os.path.join(folder_save_plots, f'{ID}_{file[:-4]}_FFT.png'))
            
            
            if plot_spectra:
                
                i = n
                
                ID = int_to_str3([i])[0]
            
                fig = plt.figure(figsize = (10,5))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']) + f' - ID:{ID}',y = 0.95, fontweight='bold', fontsize = 16)
    
                # ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
                ax2 = fig.add_axes([0.08, 0.12, 0.7, 0.75])
                ax3 = fig.add_axes([0.85, 0.12, 0.03, 0.75])
                # ax4 = fig.add_axes([0.1, 0.08, 0.7, 0.25])
    
                #make time vector
                t = df.loc[i,'Input time']
    
    
                #plot waveform (top subfigure)    
                # ax1.plot(t, df.loc[i,'Input acc'])
                # ax1.grid()
                # ax1.set_ylabel('Acc. [m/s\u00b2]')
                # ax1.set_xlabel('Time [s]')
    
    
    
                #plot spectrogram (bottom subfigure)
                #spl2 = x
                Pxx, freqs, bins, im = ax2.specgram(df.loc[i,'Input acc'], Fs=1/df.loc[i,'dT'][0], cmap='jet_r') # remove cmap for different result
    
                # line colour is white
                # for periods in struc_periods:
                #     ax2.axhline(y = 1/periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--')
                #     ax2.text(t[0]-1.5, 1/periods, f'f$_{struc_periods.index(periods)+1}$', fontsize='small')
    
                mappable = ax2.images[0]
                # plt.colorbar(
                plt.colorbar(mappable=mappable, cax=ax3).set_label(label='Amplitude [dB]', size=16)
                ax2.set_xlabel('Time [s]' , fontsize = 16)
                ax2.xaxis.set_tick_params(labelsize=14)
                ax2.set_ylabel('Freq. [Hz]', fontsize = 16)
                ax2.yaxis.set_tick_params(labelsize=14)
                ax3.yaxis.set_tick_params(labelsize=14)
                
                # line colour is white
                for periods in struc_periods:
                    
                    frequency = round(1 / periods,1)
                    
                    line = ax2.axhline(y = 1/periods, color = 'black', alpha = 1, linewidth=2, linestyle = '--')
                    ax2.text(0.9*t[-1], 1/periods*1.1, f'f({struc_periods.index(periods)+1})={frequency} Hz', fontsize=12, fontweight='bold'
                             , horizontalalignment = 'right', backgroundcolor = 'w')
                    # line2 = ax4.axvline(x = periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--')
                    # ax4.text(periods-0.01, df.loc[n, 'Peak acc']+0.1 ,f'{struc_periods.index(periods)+1}', fontsize='small')
                
                ax2.legend([line],['Struct. freq.'], fontsize = 16, loc= 'upper left')
                # plt.legend([line],['Struct. freq.'], fontsize = 16)
                # ax4.legend([line2],['Struct. periods'])
    
                # ax4.plot(T, df.loc[i,'Spectra acc'])
                # ax4.grid(axis='y')
                # ax4.set_ylabel('Acc. [m/s\u00b2]')
                # ax4.set_xlabel('Period [s]')
    
                # ax1.set_title(f'Ground motion')
                # ax2.set_title(f'Spectogram')
                # ax4.set_title(f'Acceleration spectrum')
                
                plt.close()
                
                fig.savefig(os.path.join(folder_save_plots, f'{ID}_{file[:-4]}_Spec.png'))
                # print('saved')
                
            if plot_GM:
                
                i = n
                
                ID = int_to_str3([i])[0]
            
                fig = plt.figure(figsize = (14,10))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']) + f' - ID:{ID}' , fontweight='bold', fontsize = 16)
    
                ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.38]) #[left bottom width height]
                # ax2 = fig.add_axes([0.1, 0.425, 0.7, 0.25], sharex=ax1)
                # ax3 = fig.add_axes([0.83, 0.425, 0.03, 0.25])
                ax4 = fig.add_axes([0.1, 0.08, 0.8, 0.38])
    
                #make time vector
                t = df.loc[i,'Input time']
    
    
                #plot waveform (top subfigure)    
                ax1.plot(t, df.loc[i,'Input acc'])
                ax1.grid()
                ax1.set_ylabel('Acceleration [g]', fontsize = 16)
                ax1.yaxis.set_tick_params(labelsize=14)
                
                ax1.set_xlabel('Time [s]', fontsize = 16)
                ax1.xaxis.set_tick_params(labelsize=14)
    
    
    
                #plot spectrogram (bottom subfigure)
                #spl2 = x
                # Pxx, freqs, bins, im = ax2.specgram(df.loc[i,'Input acc'], Fs=1/df.loc[i,'dT'][0], cmap='jet_r') # remove cmap for different result
    
                # line colour is white
                # for periods in struc_periods:
                #     ax2.axhline(y = 1/periods, color = 'black', alpha = 0.7, linewidth=0.8, linestyle = '--')
                #     ax2.text(t[0]-1.5, 1/periods, f'f$_{struc_periods.index(periods)+1}$', fontsize='small')
    
                # mappable = ax2.images[0]
                # plt.colorbar(mappable=mappable, cax=ax3, label='Amplitude [dB]')
                # ax2.set_xlabel('Time [s]')
                # ax2.set_ylabel('Freq. [Hz]')
                
                # line colour is white
                # for periods in [struc_periods[0]]:
                   
                if True:
                    indx = 0
                    for periods in struc_periods:
                        
                        period = round(periods,2)
                        line2 = ax4.axvline(x = periods, color = 'black', alpha = 1, linewidth=1.2, linestyle = '--')
                    
    
                            
                        if indx <= 1:
                            ax4.text(periods+0.02, min(df.loc[i,'Spectra acc']),f'{period} s', rotation = 90,
                                     ha = 'left', va = 'bottom', fontweight='bold', fontsize = 14)
                        else:
                            ax4.text(periods-0.01, max(df.loc[i,'Spectra acc']),f'{period} s', rotation = 90
                                     , ha='right', va = 'top', fontweight='bold', fontsize = 14)
                            
                        indx += 1
                    
                    
                    # ax2.legend([line],['Struct. freq.'])
                    ax4.legend([line2],['Struct. freq.'], fontsize = 16, loc= 'upper right')
    
                ax4.text(x=0, y=1, s=f'Tm = {Tm} s', transform=ax4.transAxes, va = 'bottom', ha='left', fontsize = 16)
                ax4.plot(T, df.loc[i,'Spectra acc'])
                ax4.grid(axis='y')
                ax4.set_ylabel('Spectral acceleration [g]', fontsize = 16)
                ax4.yaxis.set_tick_params(labelsize=14)
                
                ax4.set_xlabel('Period [s]' , fontsize = 16)
                ax4.xaxis.set_tick_params(labelsize=14)
    
                ax1.set_title(f'Ground motion' ,fontweight='bold', fontsize = 16)
                # ax2.set_title(f'Spectogram')
                ax4.set_title(f'Acceleration spectrum',  fontweight='bold', fontsize = 16)
                
                plt.close()
                
                fig.savefig(os.path.join(folder_save_plots, f'{ID}_{file[:-4]}.png'))
                
                
            #%%
            if plot_GM: # Only time
                
                i = n
                
                ID = int_to_str3([i])[0]
            
                fig = plt.figure(figsize = (14,10))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']) + f' - ID:{ID}' , fontweight='bold', fontsize = 16)
    
                ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.38]) #[left bottom width height]
                # ax2 = fig.add_axes([0.1, 0.425, 0.7, 0.25], sharex=ax1)
                # ax3 = fig.add_axes([0.83, 0.425, 0.03, 0.25])
                #ax4 = fig.add_axes([0.1, 0.08, 0.8, 0.38])
    
                #make time vector
                t = df.loc[i,'Input time']
    
    
                #plot waveform (top subfigure)    
                ax1.plot(t, df.loc[i,'Input acc'])
                ax1.grid()
                ax1.set_ylabel('Acceleration [g]', fontsize = 16)
                ax1.yaxis.set_tick_params(labelsize=14)
                
                ax1.set_xlabel('Time [s]', fontsize = 16)
                ax1.xaxis.set_tick_params(labelsize=14)
    
                ax1.set_title(f'Ground motion' ,fontweight='bold', fontsize = 16)
    
                
                plt.close()
                
                fig.savefig(os.path.join(folder_save_plots, f'{ID}_{file[:-4]}_Time.png'))
                
            if plot_GM: # Only Spec
                
                i = n
                
                ID = int_to_str3([i])[0]
            
                fig = plt.figure(figsize = (14,10))
                plt.suptitle('GM: ' + str(df.loc[i,'Ground motion']) + f' - ID:{ID}' , fontweight='bold', fontsize = 16)
    
                #ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.38]) #[left bottom width height]
                # ax2 = fig.add_axes([0.1, 0.425, 0.7, 0.25], sharex=ax1)
                # ax3 = fig.add_axes([0.83, 0.425, 0.03, 0.25])
                ax4 = fig.add_axes([0.1, 0.55, 0.8, 0.38]) #[left bottom width height]
    
                #make time vector
                t = df.loc[i,'Input time']
    
    
                
                if True:
                    indx = 0
                    for periods in struc_periods:
                        
                        period = round(periods,2)
                        line2 = ax4.axvline(x = periods, color = 'black', alpha = 1, linewidth=1.2, linestyle = '--')
                    
    
                            
                        if indx <= 1:
                            ax4.text(periods+0.02, min(df.loc[i,'Spectra acc']),f'{period} s', rotation = 90,
                                     ha = 'left', va = 'bottom', fontweight='bold', fontsize = 14)
                        else:
                            ax4.text(periods-0.01, max(df.loc[i,'Spectra acc']),f'{period} s', rotation = 90
                                     , ha='right', va = 'top', fontweight='bold', fontsize = 14)
                            
                        indx += 1
                    
                    
                    # ax2.legend([line],['Struct. freq.'])
                    ax4.legend([line2],['Struct. freq.'], fontsize = 16, loc= 'upper right')
    
                ax4.text(x=0, y=1, s=f'Tm = {Tm} s', transform=ax4.transAxes, va = 'bottom', ha='left', fontsize = 16)
                ax4.plot(T, df.loc[i,'Spectra acc'])
                ax4.grid(axis='y')
                ax4.set_ylabel('Spectral acceleration [g]', fontsize = 16)
                ax4.yaxis.set_tick_params(labelsize=14)
                
                ax4.set_xlabel('Period [s]' , fontsize = 16)
                ax4.xaxis.set_tick_params(labelsize=14)
    
                ax1.set_title(f'Ground motion' ,fontweight='bold', fontsize = 16)
                # ax2.set_title(f'Spectogram')
                ax4.set_title(f'Acceleration spectrum',  fontweight='bold', fontsize = 16)
                
                plt.close()
                
                fig.savefig(os.path.join(folder_save_plots, f'{ID}_{file[:-4]}_Spec.png'))
            
#%%                
            
                
#%%            
            
            n += 1
            
            

# Export dataframe

# df.to_pickle(output_directory + '/GM_spectra.pkl')
            
            
          