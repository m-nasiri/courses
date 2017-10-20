#*****************************************************************************/
# @file    rbf_iris.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    25 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
#import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rbf
import scipy


# import some data to play with
iris = scipy.io.loadmat('..\dataset\iris.mat')
X = iris["irisInputs"]
t = iris["irisTargets"]
             
K = 3

# Clustering
kmeans = KMeans(n_clusters=K, random_state=0).fit(X.T)
centers = kmeans.cluster_centers_
lables = kmeans.labels_

variances = np.zeros((1,K))
for cent in range(K):
    variances[0, cent] = np.var(X[:,(lables == cent)])
spreads = np.matrix(np.sqrt(variances))



# RBF  
WW, average_error_array =rbf.RBF(X, t, K, centers, spreads, mu = 0.1, maxEpoch = 200)


# Prediction
class_output = np.zeros((t.shape[0],X.shape[1]))
class_result = np.zeros((1,X.shape[1]))
for idx in range(X.shape[1]):
    c = rbf.RBFpredict(np.matrix(X[:,idx]).T, WW, K, centers, spreads)
    class_output[:, idx] = c[:,0]
    class_result[0, idx] = np.argmax(class_output[:, idx])


plt.figure(num = 1, figsize=(8,6))
plt.plot(average_error_array.T)
plt.xlabel('Epoch')
plt.ylabel('Average Error Rate')
plt.grid()
plt.show()
plt.savefig('..\images\iris_RBF_K=12.jpg')






