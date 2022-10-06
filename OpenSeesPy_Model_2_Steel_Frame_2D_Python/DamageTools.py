# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:36:33 2022

@author: gabri
"""


import numpy as np
import os

#%%

   
def Yielding_point(x, y):
    
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    point_x = [x[0],x[1]]
    point_y = [y[0],y[1]]
    K_i = abs(y[0]-y[1]) / abs(x[0]-x[1])

    loc_min = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1
    
    # plt.figure()
    # plt.plot(x,y)
    # plt.plot(linear_x, linear_y)
    # plt.scatter(x[loc_min],y[loc_min])
    # plt.grid()
    # plt.show()


    if len(loc_min)==0:
        loc_min = len(y)-1 # select the last min
    else:
        loc_min = loc_min[0]

    linear_y = np.arange(0, y[loc_min], 1)
    linear_x = linear_y/K_i

    dim = len(linear_y)

    # plt.figure()
    # plt.plot(x,y)
    # plt.grid()
    # plt.show()


    diff = []
    min_diff = 10

    for i in range(1, dim):  
        
        bilin_x = [0, linear_x[dim-i],x[loc_min]]
        bilin_y = [0, linear_y[dim-i], y[loc_min]]
        
        # plt.figure()
        # plt.plot(x,y)
        # plt.plot(bilin_x, bilin_y)
        # plt.title('Case %.0f: ' %(i))
        # plt.grid()
        # plt.show()
        
        
        A_tot_bilin = np.trapz(bilin_y[:loc_min+1], x=bilin_x[:loc_min+1])
        
        # bilinear curve area
        
        A_tot_real = np.trapz(y[:loc_min+1], x=x[:loc_min+1])
       # A_3 = np.trapz(y[:loc_min+1], x=x[:loc_min+1])
        
        # diff = A_tot_bilin - A_tot_real
        
        # print('Case %.0f: %.1f-%.1f = %.1f' %(i,A_tot_bilin, A_tot_real , diff))
        diff.append(abs(A_tot_bilin - A_tot_real))
        
      
        if diff[i-1] < min_diff:
            min_diff = diff[i-1]
            D_y = linear_x[dim-i]
            F_y = linear_y[dim-i]
            
            # plt.figure()
            # plt.plot(x,y)
            # plt.plot(bilin_x, bilin_y)
            # plt.title('Case %.0f: ' %(i))
            # plt.grid()
            # plt.show()


        
       # print('Yielding point: %.3f - %.2f' %(D_y, M_y))
    
    return D_y, F_y, x[loc_min], y[loc_min] # yielding point ; ultimate resistance point [deformation, force] 
                                                                        

#%%
"""
@author: Daniel Hutabarat - UC Berkeley, 2017
"""

def processNGAfile(filepath, scalefactor=None):
    '''
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and 
    time iterval of the recording.
    Parameters:
    ------------
    filepath : string (location and name of the file)
    scalefactor : float (Optional) - multiplier factor that is applied to each
                  component in acceleration array.
    
    Output:
    ------------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.
    
    Example: (plot time vs acceleration)
    filepath = os.path.join(os.getcwd(),'motion_1')
    desc, npts, dt, time, inp_acc = processNGAfile (filepath)
    plt.plot(time,inp_acc)
        
    '''    
    
    if not scalefactor:
        scalefactor = 1.0
    with open(filepath,'r') as f:
        content = f.readlines()
    counter = 0
    desc, row4Val, acc_data = "","",[]
    for x in content:
        if counter == 1:
            desc = x
        elif counter == 3:
            row4Val = x
            if row4Val[0][0] == 'N':
                val = row4Val.split()
                npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                dt = float(val[(val.index('DT='))+1])
            else:
                val = row4Val.split()
                npts = float(val[0])
                dt = float(val[1])
        elif counter > 3:
            data = str(x).split()
            for value in data:
                a = float(value) * scalefactor
                acc_data.append(a)
            inp_acc = np.asarray(acc_data)
            time = []
            for i in range (0,len(acc_data)):
                t = i * dt
                time.append(t)
        counter = counter + 1
    return desc, npts, dt, time, inp_acc
