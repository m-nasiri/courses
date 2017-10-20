#*****************************************************************************/
# @file    rbf_gaussian.py
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

for i in range(50): print(' ')    
#plt.close("all")

#define Macros
CLASS_NUM=2
SAMPLE_PER_CLASS=500

samples_class0=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class1=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)

X = np.concatenate((samples_class0, samples_class1)).T
t = np.zeros((CLASS_NUM, CLASS_NUM*SAMPLE_PER_CLASS))
t[0,                0:SAMPLE_PER_CLASS          ] = 1
t[1, SAMPLE_PER_CLASS:CLASS_NUM*SAMPLE_PER_CLASS] = 1


K = 2

# Clustering
kmeans = KMeans(n_clusters=K, random_state=0).fit(X.T)
centers = kmeans.cluster_centers_
lables = kmeans.labels_

variances = np.zeros((1,K))
for cent in range(K):
    variances[0, cent] = np.var(X[:,(lables == cent)])
spreads = np.matrix(np.sqrt(variances))



# RBF    
WW, average_error_array =rbf.RBF(X, t, K, centers, spreads, mu = 0.1, maxEpoch = 30)


# Prediction
class_output = np.zeros((t.shape[0],X.shape[1]))
class_result = np.zeros((1,X.shape[1]))
for idx in range(X.shape[1]):
    c = rbf.RBFpredict(np.matrix(X[:,idx]).T, WW, K, centers, spreads)
    class_output[:, idx] = c[:,0]
    class_result[0, idx] = np.argmax(class_output[:, idx])


plt.figure(num = 1, figsize=(8,6))
#red_patch = mpatches.Patch(label='Number Of Clusters ='+str(K))
#plt.legend(handles=[red_patch])
plt.plot(average_error_array.T)
plt.xlabel('Epoch')
plt.ylabel('Average Error Rate')
plt.grid()
plt.show()
plt.savefig('..\images\gaussian_RBF.jpg')    
    

colormap=['r','b']
plt.figure(num = 2, figsize=(8,6))
for i in range(X.shape[1]):
    plt.scatter(X[0,i],X[1,i], s=20, c=colormap[class_result[0,i].astype(int)])    
plt.savefig('..\images\gaussian_predicted_RBF.jpg') 

colormap=['b','r']
plt.figure(num = 3, figsize=(8,6))
for i in range(X.shape[1]):
    plt.scatter(X[0,i],X[1,i], s=20, c=colormap[t[0,i].astype(int)])
plt.savefig('..\images\gaussian_correct_class.jpg') 
    
    








