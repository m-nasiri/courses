#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:27:10 2017

@author: family
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

dataset = './dataset_noisy_0.025/'
n_shapes_list =  [1,2,3,4]

for n_shapes in n_shapes_list:
    
    read_dataset_dir = "./dataset_noiseless/"+str(n_shapes+1)+"/"
    write_dir = dataset + str(n_shapes+1)+"/"
    
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
            
    n_samples = 100
    for i in range(n_samples): #n_samples
        file_name = read_dataset_dir + 'image_'+str(i)+'.png'
        print(file_name)
        im = plt.imread(file_name)[:,:,0:3]
        not_solid_image = (np.random.random((im.shape)) - 0.5)/20 
        im = im + not_solid_image
        im = im / np.max(im)
        plt.imsave(write_dir + 'image_'+str(i)+'.png', im)
    


