#*****************************************************************************/
# @file    SOM.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    10 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
from scipy import spatial

def som(X, K, mu, maxEpoch):
    mu = np.linspace(mu, 0, maxEpoch)
    W = np.random.uniform(0,30,(K, X.shape[0]))
    #W = np.random.randn(K, X.shape[0])
    W_array = np.zeros((K, maxEpoch*X.shape[1]))
    sample_error = np.zeros((1,X.shape[1]))
    epoch_error = np.zeros((1,maxEpoch))
    itr = 0
    for epoch in range(maxEpoch):
        
        for sample in range(X.shape[1]):
            
            dist = spatial.distance.cdist(W, np.matrix(X[:,sample]))
            min_dist_arg = np.argmin(dist)
            #print(min_dist_arg)
            error = X[:,sample]-W[min_dist_arg,:]
            diff = mu[epoch] * error
            W[min_dist_arg,:] = W[min_dist_arg,:] + diff
            
            W_array[:, itr] = np.linalg.norm(W, axis = 1)
            sample_error[0, sample] = np.linalg.norm(W)
            itr = itr + 1
            
        epoch_error[0, epoch] = np.sum(sample_error)/X.shape[1]
    
    return W, epoch_error


