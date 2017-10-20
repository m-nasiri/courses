#*****************************************************************************/
# @file    rbf.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    10 May 2017
# @brief   
#*****************************************************************************/

import numpy as np


def  sigmoid(x): return 1/(1 + np.exp(-x))      # activation function
def dsigmoid(x): return x * (1 - x)             # derivative of sigmoid

def phi(x, center, spread):
    rr = np.sum((x-center)**2)
    ss = spread**2
    return np.exp(-rr/(2*ss)) 

# X         input data   (#features * #sample)
# t         target       (#classes * #sample)
# HL        #hidden Layers
# centers   centers of clusters
# spreads   standard deviation of clusters
# mu        learning rate
# maxEpoch  #epoch
def RBF(X, t, HL, centers, spreads, mu, maxEpoch, verbose=False):
    
    y = np.zeros((HL,1))
    yy = np.zeros((HL+1,1)) 
    WW = np.random.randn(t.shape[0], HL+1)


    epoch_error_array = np.zeros(t.shape)
    average_error_array = np.zeros((1,maxEpoch))

    for epoch in range(maxEpoch):
        for sample in range(X.shape[0]):
    
            for k, cen in enumerate(centers):    
                y[k,0] = phi(X[sample,:], cen, spreads[0,k])
    
            #feed-forward
            yy[1:,0]=y[:,0]
            O = sigmoid(np.dot(WW, yy))

            #back-propagation
            e = t[:,sample]-O[:,0]
            dphi = dsigmoid(O)
            delta = dphi[:,0] * e
            dW =  mu * np.dot(np.array([delta]).T, yy.T)
            WW = WW + dW
        
            # store error
            epoch_error_array[:, sample] = e
  
        total_error = np.sum(epoch_error_array**2, axis = 0)/2
        average_error = np.mean(total_error)
        average_error_array[0,epoch] = average_error
        if(verbose == True):
            print('RBF epoch ={}   error={}'.format(epoch, average_error))

    return WW, average_error_array

def RBFpredict(x, WW, HL, centers, spreads):

    y = np.zeros((HL,1))
    yy = np.zeros((HL+1,1)) 
    
    for k, cen in enumerate(centers):
        y[k,0] = phi(x, cen, spreads[0,k])
        
        #feed-forward
        yy[1:,0]=y[:,0]
        O = sigmoid(np.dot(WW, yy))    
    return O





