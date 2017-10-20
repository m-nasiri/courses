#*****************************************************************************/
# @file    Kmeans.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    10 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
from scipy import spatial

def kmeans(X, K, var_threshold, dist_threshold, maxEpoch):
    
    centers = np.zeros((K, X.shape[0], maxEpoch))
    variances = np.zeros((K, maxEpoch))
    var_diff = np.zeros((1, K))     # variances of samples in a cluster
    dist_diff = np.zeros((1, K))    # displacement of centers
    dist = np.zeros((K,X.shape[1]))
    rand = np.random.choice(X.shape[1], K, replace=False)
    centers[:,:,0] = X[:,rand].T
    
    for itr in range(1,maxEpoch):      
    
        for cent in range(K):
            dist[cent,:] = spatial.distance.cdist(X.T, np.matrix(centers[cent,:,itr-1])).T
    
        cluster = np.argmin(dist, axis = 0)
        for cent in range(K):
            centers[cent,:, itr] = np.mean(X[:,(cluster == cent)], axis = 1)
            variances[cent, itr] = np.var(X[:,(cluster == cent)])
        
        for cent in range(K):
                var_diff[0, cent] = ((variances[cent, itr] - variances[cent, itr-1])/variances[cent, itr])
                dist_diff[0, cent] = spatial.distance.cdist(np.matrix(centers[cent,:, itr]), np.matrix(centers[cent,:, itr-1]))
        
        var_diff_flag = var_diff>var_threshold
        dist_diff_flag = dist_diff>dist_threshold          
        
        if ((var_diff_flag.any() == False) and (dist_diff_flag.any() == False)):
            break
    
    return cluster, centers ,itr


