# -*- coding: utf-8 -*-
"""
Created on Wed May 10 03:24:39 2017

@author: Family
"""

#*****************************************************************************/
# @file    mlp_gaussian.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
#from scipy import spatial
import Kmeans

for i in range(50): print(' ')    
plt.close("all")

#define Macros
CLASS_NUM=2
SAMPLE_PER_CLASS=1000

samples_class0=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class1=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)

X = np.concatenate((samples_class0, samples_class1)).T
t = np.concatenate((np.zeros([SAMPLE_PER_CLASS,1]), np.ones([SAMPLE_PER_CLASS,1]))).T

maxEpoch=20  # minimum 2 epoch
K = 2
var_threshold = 0.01
dist_threshold = 0.01

cluster, centers, iterration = Kmeans.kmeans(X, K, var_threshold, dist_threshold, maxEpoch)

print('centers=\n', centers[:,:,iterration])
print(iterration)

center_displacement = np.diff(centers[:,:,:], axis = 2)


plt.figure(1)
for i in range(centers.shape[0]):
    for j in range(centers.shape[1]):
        plt.plot(np.arange(0,iterration), center_displacement[i,j,:iterration])
plt.xlabel('Epoch')
plt.ylabel('Displacement Of centers')
plt.show()
plt.grid()
plt.savefig('..\images\iris_kmeans.jpg')





