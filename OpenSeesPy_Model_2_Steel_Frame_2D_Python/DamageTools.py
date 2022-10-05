# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:36:33 2022

@author: gabri
"""

@ -82,32 +82,41 @@
        
Outputs:
    R

    
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



    loc_min = loc_min[0] # select the last min

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
        
        
        # bilinear curve area
        
        A_tot_real = np.trapz(y[:loc_min+1], x=x[:loc_min+1])
       # A_3 = np.trapz(y[:loc_min+1], x=x[:loc_min+1])
        
        diff = A_tot_bilin - A_tot_real
        
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
                                                                        
