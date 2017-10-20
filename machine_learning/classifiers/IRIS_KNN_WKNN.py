# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:40:18 2017

@author: Family
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

for i in range(100): print(' ')

# import some data to play with
iris = datasets.load_iris()

#define Macros
TEST_PERCENT=0.1  #  0<TEST_PERCENT<1
TRAIN_PERCENT=1-TEST_PERCENT
CLASS_NUM=3
ITERR_NUM=10        # cross validation

samples_all = iris.data[:,:]
targets_all = iris.target
sample_number=iris.data.shape[0]

K=int(sample_number*TRAIN_PERCENT)
performance=np.zeros(K)
performance_w=np.zeros(K)

for k in range(1,K):    
    iteration_error=np.zeros(ITERR_NUM, dtype=int)
    iteration_error_w=np.zeros(ITERR_NUM, dtype=int)
    
    for iteration in range(ITERR_NUM):
        samples, test_samples, target_samples, target_test_samples = train_test_split(iris.data, iris.target, test_size=TEST_PERCENT, random_state=iteration)
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
            WKNN_class_assign=np.argmax(hist_class)
            
            argmaxs=np.where(hist_class==hist_class.max())
            KNN_class_assign=np.random.choice(argmaxs[0])
            
            #print('assigned class={} real class={}'.format(KNN_class_assign,target_test_samples[sample]))
            if (target_test_samples[sample]!=KNN_class_assign): iteration_error[iteration]=iteration_error[iteration]+1
            if (target_test_samples[sample]!=WKNN_class_assign): iteration_error_w[iteration]=iteration_error_w[iteration]+1
        #print('iteration= {} iteration_error = {}'.format(iteration, iteration_error[iteration]))

    #print(iteration_error)
    error=np.sum(iteration_error)
    number_of_tests=ITERR_NUM*len(test_samples);
    performance[k]=(1-(error/(number_of_tests)))*100
    print('k={}   error ={}   number_of_tests={}  performancce={}'.format(k, error, number_of_tests, performance[k]))
    error_w=np.sum(iteration_error_w)
    number_of_tests=ITERR_NUM*len(test_samples);
    performance_w[k]=(1-(error_w/(number_of_tests)))*100
    print('k={}   error ={}   number_of_tests={}  performancce={}'.format(k, error_w, number_of_tests, performance_w[k]))


plt.plot(performance[1:],c='b')
plt.plot(performance_w[1:],c='r')
plt.xlabel('K')
plt.ylabel('Performance')
plt.title('Performane of W-KNN and KNN classifier')
plt.grid()
plt.show()


