#*****************************************************************************/
# @file    mlp_iris.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import NN

for i in range(50): print(' ')
  

# import some data to play with
iris = scipy.io.loadmat('..\dataset\iris.mat')
X = iris["irisInputs"]
t = iris["irisTargets"]

NS=150;
N = np.array([4,6,3])
mu=0.1
maxEpoch=100


error_array = NN.mlp(X, t, N, NS, mu, maxEpoch)


plt.figure(num = 1, figsize=(8,6))
plt.plot(error_array.T)
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.grid()
plt.show()
plt.savefig('..\images\iris_4-6-3.jpg')





