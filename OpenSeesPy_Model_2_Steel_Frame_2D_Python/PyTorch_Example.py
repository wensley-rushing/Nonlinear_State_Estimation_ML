# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:05:25 2022

@author: s163761
"""

# Don't ask, this is just how it is...
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(30))
plt.close()

#%% Imports
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#from numpy import array
import torch
import gc
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader

import sys
import matplotlib.pyplot as plt

#%% Imputs

Batch_Size_Train = 5 # Number of samples before an estimation
Batch_Size_Test = 1

Subvec_lengh = 3 # Length of subvector (L=25)

# Number of loops for optimization of loss
Epochs = 5

#%% Import
csv_path = r'import_loads'

solar_power = pd.read_csv(os.path.join(csv_path, 'PV_Elec_Gas2.csv')).rename(columns={'Unnamed: 0':'timestamp'}).set_index('timestamp')

#%% Train/Test Data
train_set = solar_power[:'2018-10-31']
valid_set = solar_power['2018-11-01':'2019-11-18']
print('Proportion of train_set : {:.2f}'.format(len(train_set)/len(solar_power)))
print('Proportion of valid_set : {:.2f}'.format(len(valid_set)/len(solar_power)))

#%% Define Function: Split
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = Subvec_lengh
train_x,train_y = split_sequence(train_set.Elec_kW.values,n_steps)
valid_x,valid_y = split_sequence(valid_set.Elec_kW.values,n_steps)

#%% Build CNN model
class ElecDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label
    
# -----------------------------------------------------------------------------
class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(Subvec_lengh,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        #print(f'Input: {x.size()}')
        x = self.conv1d(x)
        #print(f'Output conv: {x.size()}')
        x = self.relu(x)
        #print(f'Output relu 1: {x.size()}')
        # x = x.view(-1)
        x = x[:,:,-1]
        #print(f'Output relu 1 - reshape : {x.size()}')
        x = self.fc1(x)
        #print(f'Output linear 1: {x.size()}')
        x = self.relu(x)
        #print(f'Output relu 2: {x.size()}')
        x = self.fc2(x)
        #print(f'Output linear 2 = prediction: {x.size()}')
        
        return x
    
    
#%% Choose Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device( "cpu")
model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

#%% Train/Test Data to correct form
train = ElecDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
valid = ElecDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)

train_loader = torch.utils.data.DataLoader(train,batch_size=Batch_Size_Train,shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid,batch_size=Batch_Size_Test,shuffle=False)

x, y = next(iter(train_loader))

#%% Define Train/Test Functions
train_losses = []
valid_losses = []
def Train():
    
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device).float() # X: float32
        labels = labels.to(device).float() # Y: float32
        
        optimizer.zero_grad()
        preds = model(inputs) # X
        loss = criterion(preds,labels)  # Pred, Y
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.cpu().detach().numpy())
    
    print(f'train_loss {train_loss}')
    
    # print('Input, Output type: ', inputs.dtype, labels.dtype)
    # print('Input, Output type .float (to model): ', inputs.float().dtype, labels.float().dtype)
    # print('Input loss: ', preds.dtype , labels.dtype)
#------------------------------------------------------------------------------    
def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.cpu().detach())#.numpy())
        print(f'valid_loss {valid_loss}')
       
#%% Run EPOCHS loop
epochs = Epochs
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    gc.collect()
 
#%% PLOT - See results after training

plt.figure()
plt.plot(train_losses,label='train_loss')
plt.plot(valid_losses,label='valid_loss')
plt.title('MSE Loss')
#plt.ylim(0, 100)
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#%% Split data for prediction
#target_x , target_y = split_sequence(train_set.Elec_kW.values,n_steps) # Training set
target_x , target_y = split_sequence(valid_set.Elec_kW.values,n_steps) # Validation set
inputs = target_x.reshape(target_x.shape[0],target_x.shape[1],1)

#%% Make predictions
model.eval()
prediction = []
batch_size = Batch_Size_Test
iterations =  int(inputs.shape[0]/batch_size)

model.cpu()


#idx = 0
for i in range(iterations):
    preds = model(torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float())
    #print(batch_size*i,batch_size*(i+1))
    #idx += 1
    prediction.append(preds.detach().numpy())
    
#print(idx)
    
#%% See predictions
fig, ax = plt.subplots(1, 2,figsize=(11,4))
ax[0].set_title('predicted one')
ax[0].plot(prediction)
ax[1].set_title('real one')
ax[1].plot(target_y)

for a in range(len(ax)):
    ax[a].grid()
    
    Min = min(min(prediction), min(target_y))
    Max = max(max(prediction), max(target_y))
    
#     ax[a].set_ylim(1.2*Min, 1.2*Max)
    
# plt.show()