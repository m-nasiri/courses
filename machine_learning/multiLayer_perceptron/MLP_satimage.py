#*****************************************************************************/
# @file    mlp.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import NN    

# import some data to play with
satimage = scipy.io.loadmat('..\dataset\satimage.mat')
data = satimage["satimage"]
X = data[:, :-1].T
target = data[:,  -1]

N = np.array([36,30,30,30,6])
NS=6435;

t = np.zeros((7, NS))
for s in  range(NS):
    t[target[s]-1,s]=1
t = np.delete(t, 5, axis=0)
             
mu=0.01
sample=0
maxEpoch=10000


error_array = NN.mlp(X, t, N, NS, mu, maxEpoch)

plt.figure(num = 1, figsize=(8,6))
plt.plot(error_array.T)
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.grid()
plt.show()
plt.savefig('..\images\satimage_36-30-30-30-6.jpg')




