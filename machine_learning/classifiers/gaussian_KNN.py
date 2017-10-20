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

for i in range(20): print(' ')


SAMPLE_PER_CLASS=100
samples_class_pos=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class_neg=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)

plt.figure(1)
plt.plot(samples_class_pos[:, 0], samples_class_pos[:, 1], 'ro')
plt.plot(samples_class_neg[:, 0], samples_class_neg[:, 1], 'bo')

#define Macros
TEST_PERCENT=0.1  #  0<TEST_PERCENT<1
TRAIN_PERCENT=1-TEST_PERCENT
CLASS_NUM=2
ITERR_NUM=10        # cross validation

samples_all = np.concatenate((samples_class_pos, samples_class_neg))
targets_all = np.concatenate((np.zeros([SAMPLE_PER_CLASS,1]), np.ones([SAMPLE_PER_CLASS,1])))
sample_number=SAMPLE_PER_CLASS*CLASS_NUM

K=int(sample_number*TRAIN_PERCENT)
performance=np.zeros(K)

for k in range(1,K):
    
    iteration_error=np.zeros(ITERR_NUM, dtype=int)
    
    for iteration in range(ITERR_NUM):
        
        samples, test_samples, target_samples, target_test_samples = train_test_split(samples_all, targets_all, test_size=TEST_PERCENT, random_state=iteration)
        
        feature_dist=np.zeros(samples.shape)    
        
    
        for sample in range((test_samples.shape)[0]):
            
            #distance from all features (axis)
            feature_dist[:,:]=samples-test_samples[sample,:]   
            
            feature_dist_pow2=np.power(feature_dist, 2)
            sample_dist=np.sum(feature_dist_pow2, axis=1)
            
            sample_dist_sorted=np.sort(sample_dist)[0:k]
            sample_dist_argsort=np.argsort(sample_dist)[0:k]
            
            # K Nearest Neighbour
            nearest_neighbours=target_samples[sample_dist_argsort]
            hist_class=np.histogram(nearest_neighbours, bins=np.arange(CLASS_NUM+1))[0]
            KNN_class_assign=np.argmax(hist_class)
            #print('assigned class={} real class={}'.format(KNN_class_assign,target_test_samples[sample]))
            if (target_test_samples[sample]!=KNN_class_assign): iteration_error[iteration]=iteration_error[iteration]+1
        
        #print('iteration= {} iteration_error = {}'.format(iteration, iteration_error[iteration]))
    
    #print(iteration_error)
    error=np.sum(iteration_error)
    number_of_tests=ITERR_NUM*len(test_samples);
    performance[k]=(1-(error/(number_of_tests)))*100
    print('k={}   error ={}   number_of_tests={}  performancce={}'.format(k, error, number_of_tests, performance[k]))


plt.figure(2)
plt.plot(performance[1:],c='b')
plt.xlabel('K')
plt.ylabel('Performance')
plt.title('Performane of KNN classifier')
plt.grid()
plt.show()

