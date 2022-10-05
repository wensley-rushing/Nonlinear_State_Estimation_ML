# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:53:31 2022

@author: larsk
"""

import openseespy.opensees as ops
import opsvis as opsv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import signal
from scipy.fft import fftshift


#from Model_definition_2D_frame import createModel
from Model_definition_3x3_frame import createModel
from gravityAnalysis import runGravityAnalysis
from ReadRecord import ReadRecord

import sys
import os


#%% Folder structure

folder_accs = r'C:\Users\larsk\Danmarks Tekniske Universitet\Thesis_Nonlinear-Damage-Detection\OpenSeesPy_Model_2_Steel_Frame_2D_Python\output_files\ACCS'

folder_structure = r'C:\Users\larsk\Danmarks Tekniske Universitet\Thesis_Nonlinear-Damage-Detection\OpenSeesPy_Model_2_Steel_Frame_2D_Python\output_files'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = list(Structure.Nodes)

#%%
load_IDs = ['003']
load_Nodes = [23, 33, 43]

load_Nodes_id = []
for i in range(len(load_Nodes)):
    load_Nodes_id.append( struc_nodes.index(load_Nodes[i]) )

# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_accs):
    for file in files:
        if rdirs == folder_accs and file.endswith(".out") and file[3:6] in load_IDs:
            #print(os.path.join(rdirs, file))
            #print(idx)
            print(file)
            
            time_Accs = np.loadtxt( os.path.join(folder_accs, file) )
            
            if file[3:6][0] != str(0):
                idx = int(file[3:6])
            elif file[3:6][1] != str(0):
                idx = int(file[4:6])
            else:
                idx = int(file[5:6])
                    
            GM = Index_Results['Ground motion'][idx]
            LF = Index_Results['Load factor'][idx]
            
            
            
            
            for i in range(len(load_Nodes)):
                time = time_Accs[:,0]
                signal_x = time_Accs[:,load_Nodes_id[i]+1]
                
                fs = 1/(time[1] - time[0])
                x = signal_x
                
                if True:
                
                    plt.figure()
                    plt.plot(time,signal_x, color = 'tab:blue')
                    plt.title(f'Acceleration response  Node {load_Nodes[i]} \n {GM} , Lf = {LF}')
                    plt.xlabel('time [s]')
                    plt.ylabel('Acceleration [m/s\u00b2]')
                    plt.grid()
                    plt.show()
                    
                    
                    #----------------------
                    
                    
                    f, t, Sxx = signal.spectrogram(x, fs,  mode='magnitude')
                    plt.figure()
                    im = plt.pcolormesh(t, f, Sxx, shading='gouraud')
    
    
                    plt.colorbar(im, label='Magnitude')
                    plt.title(f'Spectogram of response @ Node {load_Nodes[i]} \n {GM} , Lf = {LF}')
    
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [s]')
                    plt.show()
                
                
                if True:
                    fig = plt.figure()
                    ax1 = fig.add_axes([0.1, 0.77, 0.7, 0.2]) #[left bottom width height]
                    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
                    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

                    #make time vector
                    t = time

                    #plot waveform (top subfigure)    
                    ax1.plot(t, x)
                    ax1.grid()
                    ax1.set_ylabel('Acc. [m/s\u00b2]')

                    ax2.set_ylabel('Freq. [Hz]')
                    ax2.set_xlabel('Time [s]')

                    #plot spectrogram (bottom subfigure)
                    #spl2 = x
                    Pxx, freqs, bins, im = ax2.specgram(x, Fs=fs, cmap='jet_r') # remove cmap for different result
                    mappable = ax2.images[0]
                    plt.colorbar(mappable=mappable, cax=ax3, label='Amplitude [dB]')
                    
                    ax1.set_title(f'Response and spectogram Node {load_Nodes[i]} \n {GM} , Lf = {LF}')
                
                
           
sys.exit()            
