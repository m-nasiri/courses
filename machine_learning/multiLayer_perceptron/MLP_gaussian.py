#*****************************************************************************/
# @file    mlp_gaussian.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
import NN

for i in range(50): print(' ')    

#define Macros
CLASS_NUM=2
SAMPLE_PER_CLASS=1000

samples_class0=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class1=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)

#plt.figure(1)
#plt.plot(samples_class0[:, 0], samples_class0[:, 1], 'ro')
#plt.plot(samples_class1[:, 0], samples_class1[:, 1], 'bo')

X = np.concatenate((samples_class0, samples_class1)).T
t = np.concatenate((np.zeros([SAMPLE_PER_CLASS,1]), np.ones([SAMPLE_PER_CLASS,1]))).T

N = np.array([2,2,1])
NS = SAMPLE_PER_CLASS * CLASS_NUM             
mu=0.1
sample=0
maxEpoch=30

error_array = NN.mlp(X, t, N, NS, mu, maxEpoch)

plt.figure(num = 1, figsize=(8,6))
plt.plot(error_array.T)
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.grid()
plt.show()
plt.savefig('..\images\gaussian_2-2-1.jpg')






