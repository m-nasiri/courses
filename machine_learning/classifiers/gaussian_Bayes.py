# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:40:18 2017

@author: Family
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def multivariate(log_tr, x, mu, cov_inv):
    x_mu = np.matrix(x - mu)
    result = log_tr + (x_mu * cov_inv * x_mu.T)
    return  result
    
    
for i in range(20): print(' ')


SAMPLE_PER_CLASS=1000
samples_class0=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class1=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)

plt.figure(1)
plt.plot(samples_class0[:, 0], samples_class0[:, 1], 'ro')
plt.plot(samples_class1[:, 0], samples_class1[:, 1], 'bo')

#define Macros
TEST_PERCENT=0.1  #  0<TEST_PERCENT<1
TRAIN_PERCENT=1-TEST_PERCENT
CLASS_NUM=2
ITERR_NUM=10        # cross validation
CLASS_NUM=2

samples_all = np.concatenate((samples_class0, samples_class1))
targets_all = np.concatenate((np.zeros([SAMPLE_PER_CLASS,1]), np.ones([SAMPLE_PER_CLASS,1])))
sample_number=SAMPLE_PER_CLASS*CLASS_NUM

K=int(sample_number*TRAIN_PERCENT)
performance=np.zeros(K)        

iteration_error=np.zeros(ITERR_NUM, dtype=int)
for iteration in range(ITERR_NUM):
    samples, test_samples, target_samples, target_test_samples = train_test_split(samples_all, targets_all, test_size=TEST_PERCENT, random_state=iteration)
    
    target_samples=target_samples.astype(int)
    
    
    samples_class0 = np.empty((0,2))
    samples_class1 = np.empty((0,2))
    for i,s in enumerate(samples): 
        if (target_samples[i]==0):
            samples_class0 = np.append(samples_class0, np.matrix(s), axis=0)
        elif (target_samples[i]==1):
            samples_class1 = np.append(samples_class1, np.matrix(s), axis=0)
            
    
    cov_class0=np.cov(samples_class0.T)     #(2*2)
    cov_class1=np.cov(samples_class1.T)     #(2*2)
    
    cov_class0_inv=np.linalg.inv(cov_class0)    #(2*2)
    cov_class1_inv=np.linalg.inv(cov_class1)    #(2*2)
    
    mu_class0=np.mean(samples_class0, axis=0)   #(1*2)
    mu_class1=np.mean(samples_class1, axis=0)   #(1*2)
    
    trace_class0=np.trace(cov_class0)
    trace_class1=np.trace(cov_class1)
    
    log_trace_class0=np.log2(trace_class0)
    log_trace_class1=np.log2(trace_class1)
    
    
    g_class=np.zeros((len(test_samples),CLASS_NUM))
    for i in range(len(test_samples)):    
        g_class[i,0]=multivariate(trace_class0, test_samples[i,:],mu_class0,cov_class0_inv)
        g_class[i,1]=multivariate(trace_class1, test_samples[i,:],mu_class1,cov_class1_inv)
    
    bayes_class_assigned=np.argmin(g_class, axis=1)
    iteration_error[iteration]=np.sum(np.absolute(bayes_class_assigned-target_test_samples.T))
    #print(iteration_error[iteration])
        
        
print(iteration_error)
error=np.sum(iteration_error)
number_of_tests=ITERR_NUM*len(test_samples);
performance=(1-(error/(number_of_tests)))*100
print('error ={}   number_of_tests={}  performancce={}'.format(error, number_of_tests, performance))


