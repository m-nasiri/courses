#*****************************************************************************/
# @file    som_gaussian.py 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    10 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
#from scipy import spatial
import SOM

for i in range(50): print(' ')    
#plt.close("all")

#define Macros
CLASS_NUM=2
SAMPLE_PER_CLASS=100

samples_class0=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class1=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)

X = np.concatenate((samples_class0, samples_class1)).T
t = np.concatenate((np.zeros([SAMPLE_PER_CLASS,1]), np.ones([SAMPLE_PER_CLASS,1]))).T

maxEpoch=120  # minimum 2 epoch
K = 2
mu = 0.4


W, epoch_error = SOM.som(X, K, mu, maxEpoch)

print('centers=\n', W)
center_displacement = np.abs(np.diff(epoch_error, axis = 1))

plt.figure(1)
plt.plot(center_displacement.T)
plt.xlabel('Epoch')
plt.ylabel('Displacement Of centers')
plt.grid()
plt.show()
plt.savefig('..\images\gaussian_som1.jpg')





