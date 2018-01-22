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

n_shapes_list =  [2,3,4]
for n_shapes in n_shapes_list:
    
    dataset_dir = "./dataset_noiseless/"+str(n_shapes+1)+"/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    center_list = np.array([[0.16, 0.6],
                            [0.32, 0.3],
                            [0.48, 0.3],
                            [0.64, 0.3],
                            [0.83, 0.3],
                            ])
    
    center_list = np.array([0.16, 0.32, 0.48, 0.64, 0.83])
        
    n_samples = 100
    for sample in range(n_samples):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        
        #shapes_centers = np.random.random((2, n_shapes))
        shapes_centers = np.vstack((np.random.choice(center_list, size=5, replace=False),
                                    np.random.choice(center_list, size=5, replace=False)))
        shapes_vertices_num = np.random.randint(3, 8, n_shapes)
        shapes_radius = (0.25 - 0.15) * np.random.random(n_shapes) + 0.15   # 0.15 ~ 0.25
        shapes_orientation = np.random.random(n_shapes)
        shapes_colors = np.random.random((n_shapes, 4))
        shapes_colors[:,3] = (1.0 - 0.5) * np.random.random((n_shapes)) + 0.5   # darker colors
        
        patch_list = []
        for shape in range(n_shapes):
            patch_list.append(patches.RegularPolygon(shapes_centers[:,shape], 
                                                     shapes_vertices_num[shape], 
                                                     shapes_radius[shape],
                                                     facecolor= shapes_colors[shape],
                                                     orientation= shapes_orientation[shape]
                                                     ))
        
        for p in patch_list:
            ax.add_patch(p)
        
        ax.set_axis_off()
        fig_name = dataset_dir+'image_'+str(sample)+'.png'
        fig.savefig(fig_name, dpi=90, bbox_inches='tight')
        plt.close("all")
        print('writing '+dataset_dir+'image_'+str(sample)+'.png ...')
    
#    for i in range(n_samples): #n_samples
#        file_name = dataset_dir + 'image_'+str(i)+'.png'
#        print(file_name)
#        im = plt.imread(file_name)[:,:,0:3]
#        not_solid_image = (np.random.random((im.shape)) - 0.5)/5
#        im = im + not_solid_image
#        im = im / np.max(im)
#        plt.imsave(dataset_dir + 'image_'+str(i)+'.png', im)
    


