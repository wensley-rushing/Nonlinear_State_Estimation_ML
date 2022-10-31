# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:59:37 2022

@author: s163761
"""

import GPy
import matplotlib.pyplot as plt
GPy.plotting.change_plotting_library('matplotlib')

import numpy as np
import pylab as pb

# Create distance matrix faster
from scipy.spatial import distance_matrix

import sys

X0 = np.random.uniform(-3.,3.,(20,1))
Y0 = np.sin(X0) + np.random.randn(20,1)*0.05


dim = 25
N = 100
X = np.random.rand(N,3*dim)


Y = np.random.rand(N,1)
#Y = np.random.rand(100,1)


kernel  = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1., active_dims=list(range(0*dim,1*dim,1)))
kernel += GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1., active_dims=list(range(1*dim,2*dim,1)))
kernel += GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1., active_dims=list(range(2*dim,3*dim,1)))



m = GPy.models.GPRegression(X,Y,kernel)
#m['.*Gaussian_noise.variance'] = 0.
print(kernel)
plt.matshow(kernel.K(X))
plt.colorbar()
print('GPy:', kernel.K(X))
print()

print('Start optimizing')
#m.optimize(messages=True)
#m.optimize_restarts(num_restarts = 10)

#print(kernel)

#sys.exit()
kernel2  = 1.*np.exp(-1/(2*1.)* (distance_matrix(X[:,0*dim:1*dim],X[:,0*dim:1*dim],p=2)**2))
kernel2 += 1.*np.exp(-1/(2*1.)* (distance_matrix(X[:,1*dim:2*dim],X[:,1*dim:2*dim],p=2)**2))
kernel2 += 1.*np.exp(-1/(2*1.)* (distance_matrix(X[:,2*dim:3*dim],X[:,2*dim:3*dim],p=2)**2))
plt.matshow(kernel2)
plt.colorbar()

print('Own', kernel2)
print()


diff = kernel.K(X)-kernel2
print('Diff', diff)
print()

perc = kernel.K(X)/kernel2
print('Perc', perc)


sys.exit()
#%%

X1 = [1,2,3,4]
X2 = [5,6,7,8]

X12 = np.column_stack((X1, X2))

X3 = [[1,2,3,4],
      [5,6,7,8],
      [1,2,3,4],
      [5,6,7,8]]

X4 = [[21,2,3,4],
      [25,6,7,8],
      [21,2,3,4],
      [25,6,7,8]]

X34 = np.array(np.append(X3, X4, axis=1))

X1 = np.random.rand(4,1)
ker = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)
m1 = GPy.models.GPRegression(X34,X1,ker)

sys.exit()

#%%
kb = GPy.kern.Brownian(input_dim=1) 

kb.plot()
plt.grid()
plt.title('Kernel 2')


#%%
ker = kernel + kb

ker.plot()
plt.grid()
plt.title('Kernel 1 + Kernel 2')

print(ker)

#%%
ker1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[0], ARD = True)

ker1.plot()
plt.grid()
plt.title('Kernel tau=1')

print(ker1)

T = 1

i = 0
for tau in [3, 4]:
    i += 1
    ker0 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=tau, active_dims=[i], ARD=False)
    
    # ker0.plot()
    # plt.grid()
    # plt.title(f'Kernel tau={tau}')
    
    print(ker0)
    
    T += tau
    ker1 += ker0
    
    # ker1.plot()
    # plt.grid()
    # plt.title(f'Sum Kernel tau={T}')
    
print(ker1)

#%% 
ker9 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[0], ARD = False)
print(ker9)

#%% ----------------------------------


#%pylab inline
import pylab as pb
pb.ion()
import GPy
GPy.plotting.change_plotting_library('matplotlib')

import matplotlib.pyplot as plt

import numpy as np
#%%

#This functions generate data corresponding to two outputs
f_output1 = lambda x: 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2.
f_output2 = lambda x: 6. * np.cos(x/5.) + .2*x + 35. + np.random.rand(x.size)[:,None] * 8.


#{X,Y} training set for each output
X1 = np.random.rand(100)[:,None]; X1=X1*75
X2 = np.random.rand(100)[:,None]; X2=X2*70 + 30
Y1 = f_output1(X1)
Y2 = f_output2(X2)
#{X,Y} test set for each output
Xt1 = np.random.rand(100)[:,None]*100
Xt2 = np.random.rand(100)[:,None]*100
Yt1 = f_output1(Xt1)
Yt2 = f_output2(Xt2)


xlim = (0,100); ylim = (0,50)
fig = pb.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax1.set_xlim(xlim)
ax1.set_title('Output 1')
ax1.plot(X1[:,:1],Y1,'kx',mew=1.5,label='Train set')
ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5,label='Test set')
ax1.legend()
ax2 = fig.add_subplot(212)
ax2.set_xlim(xlim)
ax2.set_title('Output 2')
ax2.plot(X2[:,:1],Y2,'kx',mew=1.5,label='Train set')
ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5,label='Test set')
ax2.legend()


def plot_2outputs(m,xlim,ylim):
    fig = pb.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,100),ax=ax1)
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(100,200),ax=ax2)
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)
    
#%%

K=GPy.kern.RBF(1)
B = GPy.kern.Coregionalize(input_dim=1,output_dim=2) 
multkernel = K.prod(B,name='B.K')
print(multkernel)

#Components of B
print('W matrix\n',B.W)
print('\nkappa vector\n',B.kappa)
print('\nB matrix\n',B.B)

icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=GPy.kern.RBF(1))
print(icm)

#%% 

K = GPy.kern.Matern32(1)
icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=K)

m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=icm)
m['.*Mat32.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
print('m un-optimized:', m)
plot_2outputs(m,xlim=(0,100),ylim=(-20,60))


m.optimize()
print('m optimized:', m)
plot_2outputs(m,xlim=(0,100),ylim=(-20,60))

plt.matshow(K.K(X1))
plt.colorbar()



