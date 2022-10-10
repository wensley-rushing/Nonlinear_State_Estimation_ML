# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:14:28 2022

@author: gabri
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv


output_capacity_directory = ('elements_capacity')
output_directory = ('output_files')


# Demand

col_curv_d = np.loadtxt(output_directory+'/21_groundmotion_section_def.out')
col_curv_d = col_curv_d[:,1] 

col_M_d = np.loadtxt(output_directory+'/21_groundmotion_section_force.out')
col_M_d = col_M_d[:,1]/1000 #kNm

Park_Ang_example = np.linspace(0, 1, 100)
curvature_example = np.linspace(0, 0.06, 100)

#%% Load Capacity Files

# AnySection

AnySection = pd.read_csv((output_capacity_directory+'/anysection_curves.csv'), usecols = [1,2,3,4])
AnySection.iloc[:,1] = - AnySection.iloc[:,1]
AnySection.iloc[:,3] = - AnySection.iloc[:,3]

col_yielding_idx = AnySection[AnySection.iloc[:,3]==56.06].index.values.astype(int)[0]
beam_yielding_idx = AnySection[AnySection.iloc[:,1]==123.26].index.values.astype(int)[0]

col_ult_idx = AnySection[AnySection.iloc[:,3]==67.04].index.values.astype(int)[0]
beam_ult_idx = AnySection[AnySection.iloc[:,1]==127.04].index.values.astype(int)[0]

# OpenSees Pushover

column_capacity = pd.read_csv((output_capacity_directory + "/column_pushover.csv"), usecols = [1,2,3,4,5,6])
beam_capacity = pd.read_csv((output_capacity_directory + "/beam_pushover.csv"), usecols = [1,2,3,4,5,6])



# Plot 1: Park and Ang

plt.figure()
plt.plot(curvature_example, Park_Ang_example)
plt.xlabel('Curvature')
plt.ylabel('Park and Ang')
plt.title('Demand')
plt.grid()
plt.show()

# Plot 2: Capacity - column

plt.figure()
plt.plot(AnySection.iloc[:col_ult_idx,2], AnySection.iloc[:col_ult_idx,3], label = 'AnySection')
plt.plot(column_capacity.loc[:,'Curv'], column_capacity.loc[:,'M'], label = 'OpenSees')
plt.xlabel('Curvature')
plt.ylabel('Moment (kNm)')
plt.title('Capacity - Column')
plt.grid()
plt.legend()
plt.show()


# Plot 3: Capacity - beam

plt.figure()
plt.plot(AnySection.iloc[:beam_ult_idx,0], AnySection.iloc[:beam_ult_idx,1], label = 'AnySection')
plt.plot(beam_capacity.loc[:,'Curv'], beam_capacity.loc[:,'M'], label = 'OpenSees')
plt.xlabel('Curvature')
plt.ylabel('Moment (kNm)')
plt.title('Capacity - Beam')
plt.grid()
plt.legend()
plt.show()

import sys

sys.exit()

#%% Merge plots

fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
# twin2 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
# twin2.spines.right.set_position(("axes", 1.2))

p1, = ax.plot(AnySection.iloc[:col_yielding_idx,2], AnySection.iloc[:col_yielding_idx,3], "b-") #,label="Moment [kNm]")
p2, = twin1.plot(curvature_example, Park_Ang_example, "r-") #, label="Park and Ang")

twin1.axhline(y=1, color = 'black', linestyle = '--')
twin1.text(0.06, 1.02, 'Collapse')

twin1.axhline(y=0.5, color = 'black', linestyle = '--')
twin1.text(0.06, 0.52, 'Severe')

twin1.axhline(y=0.2, color = 'black', linestyle = '--')
twin1.text(0.06, 0.22, 'Moderate')

twin1.axhline(y=0.1, color = 'black', linestyle = '--')
twin1.text(0.06, 0.12, 'Minor')
# p3, = twin2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

ax.set_xlim(0, 0.06)
ax.set_ylim(0, 70)
twin1.set_ylim(0, 1.2)
# twin2.set_ylim(1, 65)

ax.set_xlabel("Curvature")
ax.set_ylabel("Moment [kNm]")
twin1.set_ylabel("Park and Ang")
# twin2.set_ylabel("Velocity")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
# twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
# twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

# ax.legend(handles=[p1, p2, p3])
ax.legend(handles=[p1, p2])

plt.show()