###############################################################################
#coder: Majid Nasiri
#github: https://github.com/m-nasiri/courses/new/master/evolutionary_computing
#base code:
#date: 2018-Jan-19
"""
this code is for visualizing first layer filters and activation layers of 
convolutional neural network.
"""
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import os
import numpy as np
import matplotlib.pyplot as plt
import tuning


#dataset = 'dataset_noiseless'
dataset = 'dataset_noisy_0.025'
#dataset = 'dataset_noisy_0.1'

#n_centers_tune_list = [[2,3,4,5]]
#n_centers_tune_list = [[2],[3],[4],[5]]
n_centers_tune_list = [[2],[3],[4],[5]]
#n_centers_tune_list = [[2],]
compresion_ratio_list = [2,5,10,20]
#compresion_ratio_list = [5]
#max_generation_tune_list = [3,7,12,20]
max_generation_tune_list = [5]
#n_population_tune_list = [2, 4, 8, 12, 16, 24, 32]
n_population_tune_list = [4]


gen_best_fitness_array_tune = []


for n_centers_tune in n_centers_tune_list:
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    for compresion_ratio in compresion_ratio_list:
        for max_generation_tune in max_generation_tune_list:
            for n_population_tune in n_population_tune_list:
                gen_best_fitness_array, bfgn_array_array = tuning.tune_pv(n_centers_tune=n_centers_tune,
                                                                          compresion_ratio=compresion_ratio,
                                                                          max_generation_tune=max_generation_tune,
                                                                          n_population_tune=n_population_tune,
                                                                          dataset_name= dataset,
                                                                          )
                gen_best_fitness_array_tune.append(gen_best_fitness_array)
                ax1.plot(gen_best_fitness_array.T, 
                         label='compresion ratio = '+str(compresion_ratio))
                


    ax1.legend()
    ax1.set_xticks(range(max_generation_tune))
    ax1.set_xlabel('generation')
    ax1.set_ylabel('mean of fitness')
    ax1.grid('on')
    fig1.show()
    fig1.savefig('compresion_ratio_vs_mean_best_fitness_n_cluster='+str(n_centers_tune))



#plt.figure()
#hist, bins = np.histogram(bfgn_array_array, bins=max_generation_tune)
#width = 0.7 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.ylabel('number of winning')
#plt.xlabel('generation')
#plt.show()


