#*****************************************************************************/
# @file    som_iris.py 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    10 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
#from scipy import spatial
import scipy
import SOM

for i in range(50): print(' ')    
#plt.close("all")

iris = scipy.io.loadmat('..\dataset\iris.mat')
X = iris["irisInputs"]
t = iris["irisTargets"]
                             
                            
maxEpoch=80  # minimum 2 epoch
K = 3
mu = 2

W, epoch_error = SOM.som(X, K, mu, maxEpoch)

print('centers=\n', W)
center_displacement = np.abs(np.diff(epoch_error, axis = 1))

plt.figure(1)
plt.plot(center_displacement.T)
plt.xlabel('Epoch')
plt.ylabel('Displacement Of centers')
plt.grid()
plt.show()
plt.savefig('..\images\iris_som3.jpg')




