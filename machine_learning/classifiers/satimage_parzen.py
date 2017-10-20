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

for i in range(100): print(' ')

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


h0=np.linspace(850,1400,2,dtype=int)
performance=np.zeros(len(h0))
H=np.power(h0,2)

start=datetime.datetime.now()
number=0
for h in H:
    
    feature_dist=np.zeros(samples.shape)
    iteration_error=0

    for sample in range((test_samples.shape)[0]): 
        #distance from all features (axis)
        feature_dist[:,:]=samples-test_samples[sample,:]   
        
        feature_dist_pow2=np.power(feature_dist, 2)
        sample_dist=np.sum(feature_dist_pow2, axis=1)
        
        sample_dist_sorted=np.sort(sample_dist)
        sample_dist_argsort=np.argsort(sample_dist)
        
        dist_of_samples_in_ball  = sample_dist_sorted[sample_dist_sorted<h]
        
        if (len(dist_of_samples_in_ball)==0):
            print('There is no sample in this ball near sample {}.'.format(sample))
            
        samples_in_ball          = sample_dist_argsort[0:len(dist_of_samples_in_ball)]
        target_of_samples_in_ball= target_samples[samples_in_ball]
        hist_class=np.histogram(target_of_samples_in_ball, bins=np.arange(CLASS_NUM+2))[0]
        parzen_window_class_assign=np.argmax(hist_class)
        
        #print(dist_of_samples_in_ball)
        #print(samples_in_ball)
        #print(target_of_samples_in_ball)
        #print(hist_class)
        #print(parzen_window_class_assign)
        #print('parzen assigned class={} real class={}'.format(parzen_window_class_assign,target_test_samples[sample]))
        if (target_test_samples[sample]!=parzen_window_class_assign): iteration_error=iteration_error+1
    
    error=iteration_error
    number_of_tests=ITERR_NUM*len(test_samples);
    performance[number]=(1-(error/(number_of_tests)))*100
    print('h={}   error ={}   number_of_tests={}  performancce={}'.format(h0[number], error, number_of_tests, performance[number]))
    number=number+1

end=datetime.datetime.now()
print (end - start)
plt.plot(h0, performance, c='b')
plt.xlabel('K')
plt.ylabel('Performance')
plt.title('Performane of Parzen window classifier')
plt.grid()
plt.show()


