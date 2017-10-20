# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:40:18 2017

@author: Family
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def multivariate(log_tr, x, mu, cov_inv):
    x_mu = np.matrix(x - mu)
    result = log_tr + (x_mu * cov_inv * x_mu.T)
    return  result
    
    
for i in range(20): print(' ')

iris = datasets.load_iris()

#define Macros
TEST_PERCENT=0.1  #  0<TEST_PERCENT<1
TRAIN_PERCENT=1-TEST_PERCENT
CLASS_NUM=3
ITERR_NUM=10        # cross validation
FEATURE_NUM=4

samples_all = iris.data[:,:]
targets_all = iris.target
sample_number=iris.data.shape[0]
       

iteration_error=np.zeros(ITERR_NUM, dtype=int)

for iteration in range(ITERR_NUM):
    
    samples, test_samples, target_samples, target_test_samples = train_test_split(samples_all, targets_all, test_size=TEST_PERCENT, random_state=iteration)
    
    target_samples=target_samples.astype(int)
    
    samples_class0 = np.empty((0,FEATURE_NUM))
    samples_class1 = np.empty((0,FEATURE_NUM))
    samples_class2 = np.empty((0,FEATURE_NUM))
    
    for i,s in enumerate(samples):
        if   (target_samples[i]==0):
            samples_class0 = np.append(samples_class0, np.matrix(s), axis=0)
        elif (target_samples[i]==1):
            samples_class1 = np.append(samples_class1, np.matrix(s), axis=0)
        elif (target_samples[i]==2):
            samples_class2 = np.append(samples_class2, np.matrix(s), axis=0)
               
    cov_class0=np.cov(samples_class0.T)     
    cov_class1=np.cov(samples_class1.T)     
    cov_class2=np.cov(samples_class2.T)     
    
    cov_class0_inv=np.linalg.inv(cov_class0)    
    cov_class1_inv=np.linalg.inv(cov_class1) 
    cov_class2_inv=np.linalg.inv(cov_class2)  
    
    mu_class0=np.mean(samples_class0, axis=0)   
    mu_class1=np.mean(samples_class1, axis=0)   
    mu_class2=np.mean(samples_class2, axis=0)   
    
    trace_class0=np.trace(cov_class0)
    trace_class1=np.trace(cov_class1)
    trace_class2=np.trace(cov_class2)
    
    log_trace_class0=np.log2(trace_class0)
    log_trace_class1=np.log2(trace_class1)
    log_trace_class2=np.log2(trace_class2)
    
    g_class=np.zeros((len(test_samples),CLASS_NUM))
    for i in range(len(test_samples)):    
        g_class[i,0]=multivariate(trace_class0, test_samples[i,:],mu_class0,cov_class0_inv)
        g_class[i,1]=multivariate(trace_class1, test_samples[i,:],mu_class1,cov_class1_inv)
        g_class[i,2]=multivariate(trace_class2, test_samples[i,:],mu_class1,cov_class2_inv)
    
    bayes_class_assigned=np.argmin(g_class, axis=1)
    iteration_error=0
    for sample in range(len(test_samples)):
        if (target_test_samples[sample]!=bayes_class_assigned[sample]): 
            iteration_error=iteration_error+1
    #print(iteration_error[iteration])
                
print(iteration_error)
error=np.sum(iteration_error)
number_of_tests=ITERR_NUM*len(test_samples);
performance=(1-(error/(number_of_tests)))*100
print('error ={}   number_of_tests={}  performancce={}'.format(error, number_of_tests, performance))

