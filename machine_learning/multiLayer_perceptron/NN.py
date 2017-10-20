#*****************************************************************************/
# @file    mlp.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   
#*****************************************************************************/

import numpy as np


def  sigmoid(x): return 1/(1 + np.exp(-x))      # activation function
def dsigmoid(x): return x * (1 - x)             # derivative of sigmoid

def ff_layer(WW, OO):
    return sigmoid(np.dot(WW, OO))

def bp_outputLayer(O, OO, lr, target, smpl):
    e = np.ones((O.shape[0],1))
    e[:,0] = (target[:, smpl]-O[:,0])
    dphi = O * (1-O)
    delta = dphi * e
    dW = lr * delta * OO.T
    return dW, delta, e

def bp_hiddenLayer(O, OO, lr, delt, WW):
    e = np.dot(WW[:, 1:].T, delt)
    dphi = O * (1-O)
    delta = dphi * e
    dW = lr * delta * OO.T
    return dW, delta, e
    

def mlp(X,t,N,NS,mu,maxEpoch):
    # Process             
    L = len(N)-1
    sample=0
    WW = []
    dW = []
    OO = []
    O = []
    e =[]
    delta = []
    for l in range(L):
        WW.append((np.random.randn(N[l+1], N[l]+1)))
        dW.append((np.random.randn(N[l+1], N[l]+1)))
        OO.append((np.ones((N[l]+1,1))))
        O.append((np.zeros((N[l+1],1))))
        e.append((np.zeros((N[l+1],1))))
        delta.append((np.zeros((N[l+1],1))))
    
    
    Eepoch=np.zeros((1, NS))
    Earray=np.zeros((1, maxEpoch))
    
    for epoch in range(maxEpoch):
        
        for sample in range(NS):
            
            ## FeedForward        
            OO[0][1:,0] = X[:, sample]
            O[0]=ff_layer(WW[0], OO[0])
            for l in range(L-1):
                OO[l+1][1:,0]=O[l][:,0]
                O[l+1]=ff_layer(WW[l+1], OO[l+1])
            
            ## Back propagation        
            dW[L-1], delta[L-1], e[L-1] = bp_outputLayer(O[L-1], OO[L-1], mu, t, sample)
            for l in range(L-1,0,-1):
                dW[l-1], delta[l-1], e[l-1] = bp_hiddenLayer(O[l-1], OO[l-1], mu, delta[l], WW[l])
            
            # Update Weights
            for l in range(L):
                WW[l] = WW[l] + dW[l]
            
            Eepoch[0, sample] = np.sum(np.power(e[L-1],2))/2
            
        Earray[0, epoch] = np.mean(Eepoch)
        print('epoch ={}   error={}'.format(epoch, Earray[0, epoch]))
    
    #print(WW1)
    #print(WW2)
    #print('Earray =',Earray)
    print('Eepoch', np.mean(Eepoch))
    return Earray
    





