# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:40:18 2017

@author: Family
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io


def multivariate(log_tr, x, mu, cov_inv):
    x_mu = np.matrix(x - mu)
    result = log_tr + (x_mu * cov_inv * x_mu.T)
    return  result
    
    
for i in range(20): print(' ')

# import some data to play with
satimage = scipy.io.loadmat('satimage.mat')

#define Macros
TEST_PERCENT=0.1  #  0<TEST_PERCENT<1
TRAIN_PERCENT=1-TEST_PERCENT
CLASS_NUM=7
ITERR_NUM=1        # cross validation
FEATURE_NUM=36

data = satimage["satimage"]
samples=data[0:4435,:-1]
test_samples=data[4435: ,:-1]
target_samples=data[0:4435,-1]
target_test_samples=data[4435: ,-1]

       

#iteration_error=np.zeros(ITERR_NUM, dtype=int)
    
    
target_samples=target_samples.astype(int)

samples_class1 = np.empty((0,FEATURE_NUM))
samples_class2 = np.empty((0,FEATURE_NUM))
samples_class3 = np.empty((0,FEATURE_NUM))
samples_class4 = np.empty((0,FEATURE_NUM))
samples_class5 = np.empty((0,FEATURE_NUM))
samples_class7 = np.empty((0,FEATURE_NUM))

for i,s in enumerate(samples):
    if   (target_samples[i]==1):
        samples_class1 = np.append(samples_class1, np.matrix(s), axis=0)
    elif (target_samples[i]==2):
        samples_class2 = np.append(samples_class2, np.matrix(s), axis=0)
    elif (target_samples[i]==3):
        samples_class3 = np.append(samples_class3, np.matrix(s), axis=0)
    elif (target_samples[i]==4):
        samples_class4 = np.append(samples_class4, np.matrix(s), axis=0)
    elif (target_samples[i]==5):
        samples_class5 = np.append(samples_class5, np.matrix(s), axis=0)
    elif (target_samples[i]==7):
        samples_class7 = np.append(samples_class7, np.matrix(s), axis=0)


cov_class1=np.cov(samples_class1.T)     
cov_class2=np.cov(samples_class2.T)     
cov_class3=np.cov(samples_class3.T)     
cov_class4=np.cov(samples_class4.T)     
cov_class5=np.cov(samples_class5.T)     
cov_class7=np.cov(samples_class7.T)     
   
cov_class1_inv=np.linalg.inv(cov_class1) 
cov_class2_inv=np.linalg.inv(cov_class2)  
cov_class3_inv=np.linalg.inv(cov_class3) 
cov_class4_inv=np.linalg.inv(cov_class4) 
cov_class5_inv=np.linalg.inv(cov_class5) 
cov_class7_inv=np.linalg.inv(cov_class7) 

mu_class1=np.mean(samples_class1, axis=0)   
mu_class2=np.mean(samples_class2, axis=0)   
mu_class3=np.mean(samples_class3, axis=0)   
mu_class4=np.mean(samples_class4, axis=0)   
mu_class5=np.mean(samples_class5, axis=0) 
mu_class7=np.mean(samples_class7, axis=0)   
  
trace_class1=np.trace(cov_class1)
trace_class2=np.trace(cov_class2)
trace_class3=np.trace(cov_class3)
trace_class4=np.trace(cov_class4)
trace_class5=np.trace(cov_class5)
trace_class7=np.trace(cov_class7)

log_trace_class1=np.log2(trace_class1)
log_trace_class2=np.log2(trace_class2)
log_trace_class3=np.log2(trace_class3)
log_trace_class4=np.log2(trace_class4)
log_trace_class5=np.log2(trace_class5)
log_trace_class7=np.log2(trace_class7)


g_class=np.zeros((len(test_samples),CLASS_NUM+1))
for i in range(len(test_samples)):    
    g_class[i,1]=multivariate(trace_class1, test_samples[i,:],mu_class1,cov_class1_inv)
    g_class[i,2]=multivariate(trace_class2, test_samples[i,:],mu_class2,cov_class2_inv)
    g_class[i,3]=multivariate(trace_class3, test_samples[i,:],mu_class3,cov_class3_inv)
    g_class[i,4]=multivariate(trace_class4, test_samples[i,:],mu_class4,cov_class4_inv)
    g_class[i,5]=multivariate(trace_class5, test_samples[i,:],mu_class5,cov_class5_inv)
    g_class[i,7]=multivariate(trace_class7, test_samples[i,:],mu_class7,cov_class7_inv)


g_class[:,0]=np.max(g_class)
g_class[:,6]=np.max(g_class)
bayes_class_assigned=np.argmin(g_class, axis=1)

iteration_error=0
for sample in range(len(test_samples)):
    print(sample)
    if (target_test_samples[sample]!=bayes_class_assigned[sample]): 
        iteration_error=iteration_error+1

print(iteration_error)
error=iteration_error
number_of_tests=ITERR_NUM*len(test_samples);
performance=(1-(error/(number_of_tests)))*100
print('error ={}   number_of_tests={}  performancce={}'.format(error, number_of_tests, performance))

