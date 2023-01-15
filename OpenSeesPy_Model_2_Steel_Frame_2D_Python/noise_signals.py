# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:53:17 2022

@author: lucag
"""

import numpy as np


np.random.seed(2)   #if you specify the seed, every run produces the same random numbers and you can replicate the same results



def add_noise_percentage(y, noise_level):
    # y is a numpy array
	# noise_level is the noise in relation to the signal. e.g. noise_level=0.01 corresponds to 1% of noise
    noise_std = noise_level*np.std(y)   
    print('STD_Per: ', round(noise_std,2))
    
    y_noisy = y + np.random.normal(scale=noise_std,size=y.shape)
    return y_noisy
	
	
def add_noise_abs(y, noise_level_ms2):
	# y is a numpy array
	# noise_level_ms2 is the std of the noise
    print('STD_ABS: ', round(noise_level_ms2,2))
    
    y_noisy = y + np.random.normal(scale=noise_level_ms2,size=y.shape)
    return y_noisy


def add_noise_SNR_db(y, SNR_db):
	# y is a numpy array [Volts]
    
    # The power of signal
    y_power = y**2 # Watts
    # Mean of the power E[S^2]
    y_power_mean = y_power.mean() # Watts
    
    # Convert mean of power to dB
    y_power_mean_db = 10*np.log10(y_power_mean) # dB
    
    # The mean noise in dB
    noise_mean_db = y_power_mean_db - SNR_db # dB
    # Convert mean noise to Power 
    noise_mean = 10**(noise_mean_db/10) # Watts
 
    # np.sqrt(noise_mean) is the std of the noise
    noise_sd = np.sqrt(noise_mean)
    # print(round(noise_sd,4))
	
    if SNR_db >= 1000:
        print('STD_SNR: ', 'N/A')
        y_noisy = y
    else:
        print('STD_SNR: ', round(noise_sd,2))
        y_noisy = y + np.random.normal(loc=0, scale=noise_sd, size=y.shape)
        
    return y_noisy

#%%

dt = 0.01
t = np.arange(1000)*dt
y = np.sin(t)
std_y = np.std(y)

print('STD(y): ', round(std_y, 2))

noise_per = 0.05
noise_abs = 0.05
noise_snr = 25


y_noisy1 = add_noise_percentage(y, noise_per) # 5%
y_noisy2 = add_noise_abs(y, noise_abs) # with a standard deviation of 0.01
y_noisy3 = add_noise_SNR_db(y, noise_snr) # with SRN = 20 dB

import matplotlib.pyplot as plt


fig, axs = plt.subplots(2,1, figsize=(15,10))

# Overall
for ax in axs:
    ax.plot(t,y, #color='k', linestyle='solid', # -----
            label="True signal")
    
    ax.plot(t,y_noisy1, #color='k', linestyle=(0, (1, 3)), # . . . . 
            label=f"Percentag noise - std_noise = {noise_per*100}% std(y)")
    
    ax.plot(t,y_noisy2, #color='k', linestyle=(0, (5, 5)), #  - - - - 
            label=f"Absolute noise - std_noise = {noise_abs*100}%")
    
    ax.plot(t,y_noisy3, #color='k', linestyle=(0, (3, 10, 1, 5)), #  -.-.-.-.-.
            label=f"SNR noise - SNR_noise = {noise_snr} dB")
    ax.legend(fontsize=12)
    ax.grid()
    
    ax.set_xlabel('Time', weight='bold', fontsize=16)
    ax.set_ylabel('Amplitude', weight='bold', fontsize=16)
    
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)



# ONLY 2nd sub-plot
axs[1].set_ylim(0.65,1.25)
axs[1].set_xlim(1,2)

plt.show()

#%%
print('----------------')
for i in [1000, 40, 30, 20]:
    y_noisy9 = add_noise_SNR_db(y, i) # with SRN = 20 dB