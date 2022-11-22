# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:57:00 2022

@author: gabri
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys

acc = [4.627087937,
       2.496847705,
       0.379488996,
       5.141099759,
       5.106387944,
       0.642664943,
       0,
       1.363855448,
       0.432722818,
       1.867198023
       ]

x_acc = np.arange(0,len(acc))*0.02

# if L=3, s=2

y_true_red_2 = [acc[2], acc[4], acc[6], acc[8]]
x_red = [x_acc[2], x_acc[4], x_acc[6], x_acc[8]]


rmse = []
err = []
plot_curve = []
x_plot = []
j = 0
flag = 0




for i in range(0,len(acc)):   
    
    if x_acc[i] == x_red[0]: # if we are at the first value of the reduced vector we can start making estimations and measure
                                # the error
        flag += 1
        
    if flag == 1:  # measure the error
        
        if x_acc[i] == x_red[j]:
            
            err.append((acc[i] - y_true_red_2[j])**2)
            plot_curve.append(y_true_red_2[j])
            
            j += 1
            if  x_acc[i] == x_red[-1]:  # if we are at the last value of the reduced vector we cannot make any other estimation
                flag = 0
            
        else:
            
            inter = np.interp(x_acc[i] ,[x_red[j-1], x_red[j]], [y_true_red_2[j-1], y_true_red_2[j]]) 
            err.append((acc[i] - inter)**2)
            
            plot_curve.append(inter)
        
        x_plot.append(x_acc[i])
        
        
        
rmse.append(math.sqrt(sum(err)/len(err)))


plt.figure()
plt.plot(x_acc, acc, label = 'Entire')
plt.plot(x_red, y_true_red_2, label = 'Reduced - python')
plt.grid()
plt.legend()

plt.figure()
plt.plot(x_acc, acc, label = 'Entire')
plt.plot(x_plot, plot_curve, label = 'Reduced - calculation')
plt.grid()
plt.legend()

        