# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:19:08 2022

@author: gabri

Resources:
    
    Theory:
    https://en.wikipedia.org/wiki/Convolution
    https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
    
    Normalization:
    https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import math
from scipy import signal


# x1 = np.arange(0, 2*math.pi)

n_periods = 5
step = 10
x1 = np.arange(0, n_periods*360 + step, step)*math.pi/180

# x1 = np.linspace(0,4,9)*math.pi
x2 = x1




series_1 = []
series_2 = []
series_3 = []

for i in range(0, len(x1)):
    series_1.append(math.sin(x1[i]))
    # series_2.append(math.sin(x1[i]-1)) 
    series_3.append(math.sin(x1[i]+math.pi/2)) 
    

plt.figure()
plt.plot(x1, series_1, label='series 1')
# plt.plot(x1, series_2, label='series 2')
plt.plot(x2, series_3, label='series 3')
plt.xticks(np.linspace(0,2*n_periods,4*n_periods+1)*math.pi)
plt.xlim(0,4*math.pi)
plt.legend()
plt.grid()
plt.show()


correlation = np.correlate(series_1, series_3, mode = 'full')

lags = signal.correlation_lags(len(series_1), len(series_3), mode="full")
lag = lags[np.argmax((correlation))]

new_signal_3 = np.roll(series_3, lag)


if lag > 0:    
    new_signal_3 = new_signal_3[lag:]
    x2 = x2[lag:]
    
else:
    new_signal_3 = new_signal_3[:lag] 
    x2 = x2[:lag]

plt.figure()
plt.plot(x1, series_1, label='series 1')
# plt.plot(x1, series_2, label='series 2')
plt.plot(x2, new_signal_3, label='series 3', linestyle='-')
plt.xticks(np.linspace(0,2*n_periods,4*n_periods+1)*math.pi)
plt.xlim(0,4*math.pi)
plt.legend()
plt.grid()


sys.exit()
#%%

x = [4,4,4,4,6,8,10,8,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
y = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,8,10,8,6,4,4]

correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
# correlation = signal.correlate(series_1-np.mean(series_1), series_3 - np.mean(series_3), mode="full")

# correlation = signal.correlate(series_1, series_2)
# lags = signal.correlation_lags(len(x), len(y), mode="full")
# lag = lags[np.argmax(abs(correlation))]

#%%

vec_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
vec_2 = [4, 5, 6, 7, 8, 9, 10, 1, 2, 3]

correlation = np.correlate(vec_1,vec_2,  mode = 'full')

lags = signal.correlation_lags(len(vec_1), len(vec_2), mode="full")
lag = lags[np.argmax(abs(correlation))]

new_vec_2 = np.roll(vec_2, lag)


