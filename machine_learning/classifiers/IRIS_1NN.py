# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:40:18 2017

@author: Family
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np

# import some data to play with
iris = datasets.load_iris()

#define Macros
SAMPLE_LEN=40
TEST_LEN=50-SAMPLE_LEN
FEATURE_NUM=4
CLASS_NUM=3
error=0

samples_all = iris.data[:, :FEATURE_NUM]
targets_all = iris.target

samples_class_setosa    =samples_all[(0*(SAMPLE_LEN+TEST_LEN)):(0*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN),:]
samples_class_versicolor=samples_all[(1*(SAMPLE_LEN+TEST_LEN)):(1*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN),:]
samples_class_virginica =samples_all[(2*(SAMPLE_LEN+TEST_LEN)):(2*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN),:]

target_samples_class_setosa    =targets_all[(0*(SAMPLE_LEN+TEST_LEN)):(0*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN)]
target_samples_class_versicolor=targets_all[(1*(SAMPLE_LEN+TEST_LEN)):(1*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN)]
target_samples_class_virginica =targets_all[(2*(SAMPLE_LEN+TEST_LEN)):(2*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN)]

test_samples_class_setosa    =samples_all[(0*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN):(1*(SAMPLE_LEN+TEST_LEN)),:] 
test_samples_class_versicolor=samples_all[(1*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN):(2*(SAMPLE_LEN+TEST_LEN)),:]
test_samples_class_virginica =samples_all[(2*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN):(3*(SAMPLE_LEN+TEST_LEN)),:]

target_test_samples_class_setosa    =targets_all[(0*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN):(1*(SAMPLE_LEN+TEST_LEN))] 
target_test_samples_class_versicolor=targets_all[(1*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN):(2*(SAMPLE_LEN+TEST_LEN))]
target_test_samples_class_virginica =targets_all[(2*(SAMPLE_LEN+TEST_LEN)+SAMPLE_LEN):(3*(SAMPLE_LEN+TEST_LEN))]

samples     =np.concatenate((samples_class_setosa,
                             samples_class_versicolor,
                             samples_class_virginica))
target_samples=np.concatenate((target_samples_class_setosa,
                               target_samples_class_versicolor,
                               target_samples_class_virginica))
test_samples=np.concatenate((test_samples_class_setosa,
                             test_samples_class_versicolor,
                             test_samples_class_virginica))
target_test_samples=np.concatenate((target_test_samples_class_setosa,
                                    target_test_samples_class_versicolor,
                                    target_test_samples_class_virginica))


feature_dist=np.zeros(samples.shape)

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold =  ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

sample=0
for sample in range((test_samples.shape)[0]):
    
    feature_dist[:,:]=samples-test_samples[sample,:]   #distance from all features (axis)
    
    feature_dist_pow2=np.power(feature_dist, 2)
    sample_dist=np.sum(feature_dist_pow2, axis=1)
    
    sample_dist_sorted=np.sort(sample_dist)
    sample_dist_argsort=np.argsort(sample_dist)
    
    # One Nearest Neighbour
    nearest_neighbours=sample_dist_argsort[0]
    
    ONN_class_assign=target_samples[sample_dist_argsort[0]]
    #print('assigned class={} real class={}'.format(ONN_class_assign,target_test_samples[sample]))
    if (target_test_samples[sample]!=ONN_class_assign): error=error+1
print('error=',error)






