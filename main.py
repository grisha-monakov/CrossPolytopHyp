# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:11:06 2019

@author: Григорий
"""
import CrossPolytop
from time import clock, time, ctime
from math import factorial
import copy


#Dimension
DIM = 3
#Number of points
NUM = 6
#Number of steps
STEPS = 100
#Do you want to save results?
SAVE = False

#Calculating time of work.
t = clock()

#Creating a random configuration with parameters as above
config_f = CrossPolytop.Points(DIM, NUM)
config_s = copy.deepcopy(config_f)

#Plotting initial configuration
if DIM == 2:
    CrossPolytop.plot2D(config_f)
if DIM == 3:
    CrossPolytop.plot3D(config_f)
    
#Maximizing the volume of the convex hull for this configuration
for _ in range(STEPS):
    config_f = config_f.updatePositions()

#Plotting final configuration
if DIM == 2:
    CrossPolytop.plot2D(config_f)
if DIM == 3:
    CrossPolytop.plot3D(config_f)

#Printing results
V = config_f.getVolume()
print('Volume =', V)
print('Volume - Volume of crosspolytop =', V - 2 ** DIM / factorial(DIM))
print('Distanse to crosspolytop =', config_f.distToCrosspolytop())
print('time =', clock() - t)

#Saving results
if SAVE:
    CrossPolytop.saveExperiment(config_s, config_f, STEPS)
