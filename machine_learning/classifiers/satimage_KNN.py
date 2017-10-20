# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:40:18 2017

@author: Family
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import datetime

for i in range(10): print(' ')

# import some data to play with
satimage = scipy.io.loadmat('satimage.mat')


#define Macros
TEST_PERCENT=0.1  #  0<TEST_PERCENT<1
TRAIN_PERCENT=1-TEST_PERCENT
CLASS_NUM=7
ITERR_NUM=1        # cross validation

data = satimage["satimage"]
samples=data[0:4435,:-1]
test_samples=data[4435: ,:-1]
target_samples=data[0:4435,-1]
target_test_samples=data[4435: ,-1]


K=np.linspace(1,30,10,dtype=int)
performance=np.zeros(len(K))

start=datetime.datetime.now()
number=0
for k in K:    
    iteration_error=0
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
        hist_class=np.histogram(nearest_neighbours, bins=np.arange(CLASS_NUM+2))[0]
        #print(hist_class)
        KNN_class_assign=np.argmax(hist_class)
        #print('assigned class={} real class={}'.format(KNN_class_assign,target_test_samples[sample]))
        if (target_test_samples[sample]!=KNN_class_assign): iteration_error=iteration_error+1
    
        
    #print('iteration= {} iteration_error = {}'.format(iteration, iteration_error[iteration]))
    
    #print(iteration_error)
    error=iteration_error
    number_of_tests=ITERR_NUM*len(test_samples);
    performance[number]=(1-(error/(number_of_tests)))*100
    print('k={}   error ={}   number_of_tests={}  performancce={}'.format(k, error, number_of_tests, performance[number]))
    number=number+1

end=datetime.datetime.now()
print (end - start)
plt.plot(K,performance[:],c='b')
plt.xlabel('K')
plt.ylabel('Performance')
plt.title('Performane of KNN classifier')
plt.grid()
plt.show()


