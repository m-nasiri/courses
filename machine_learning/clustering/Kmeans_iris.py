#*****************************************************************************/
# @file    Kmeans_iris.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    10 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
import scipy
import Kmeans

for i in range(50): print(' ')    
plt.close("all")

iris = scipy.io.loadmat('..\dataset\iris.mat')
X = iris["irisInputs"]
t = iris["irisTargets"]
                             
                            
maxEpoch=100  # minimum 2 epoch
K = 3
var_threshold = 0.001
dist_threshold = 0.001

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




