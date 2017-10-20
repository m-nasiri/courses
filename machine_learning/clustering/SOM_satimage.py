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
import SOM

# import some data to play with
satimage = scipy.io.loadmat('..\dataset\satimage.mat')
data = satimage["satimage"]
X = data[:, :-1].T
target = data[:,  -1]

#t = np.zeros((7, NS))
#for s in range(NS):
#    t[target[s]-1,s]=1
#t = np.delete(t, 5, axis=0)
             
maxEpoch=500  # minimum 2 epoch
K = 6
mu = 0.001

W, epoch_error = SOM.som(X, K, mu, maxEpoch)

#print('centers=\n', W)
center_displacement = np.abs(np.diff(epoch_error, axis = 1))

plt.figure(1)
plt.plot(center_displacement.T)
plt.xlabel('Epoch')
plt.ylabel('Displacement Of centers')
plt.grid()
plt.show()
plt.savefig('..\images\satimage_som1.jpg')

print('centers=\n', np.round(W[:,0]))

