#*****************************************************************************/
# @file    mlp.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
import scipy
import Kmeans   

# import some data to play with
satimage = scipy.io.loadmat('..\dataset\satimage.mat')
data = satimage["satimage"]
X = data[:, :-1].T
target = data[:,  -1]

#t = np.zeros((7, NS))
#for s in range(NS):
#    t[target[s]-1,s]=1
#t = np.delete(t, 5, axis=0)
             
maxEpoch=100  # minimum 2 epoch
K = 6
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
plt.savefig('..\images\satimage_kmeans.jpg')


