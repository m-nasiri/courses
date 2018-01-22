
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import matplotlib.pyplot as plt
import ga_base


color_list = [[0.4, 0.5, 0.6],
              [0.7, 0.1, 0.3],
              [0.3, 0.8, 0.8],
              [1.0, 0.3, 0.6],
              [0.1, 0.4, 0.1],
              [0.5, 1.0, 0.8]]

def tune_pv(n_centers_tune = None,
            compresion_ratio=None,
            max_generation_tune=None,
            n_population_tune=None,
            dataset_name = None,
            ):
    
    mgbfa_array_array = []
    bfgn_array_array = []
    n_centers = n_centers_tune
    for n_center in n_centers:
        input_image_dir = './'+dataset_name+'/'+str(n_center) + '/'
        result_image_dir = './'+dataset_name+'/'+str(n_center) + '_result_compresion_ratio= '+str(compresion_ratio)+'/'
        
        if not os.path.exists(result_image_dir):
            os.makedirs(result_image_dir)
                   
        mgbfa_array = []
        bfgn_array = []
        for iidx in range(100):
            #iidx = 25
            plt.close("all")
            file_name = input_image_dir + 'image_' + str(iidx)
            file_name1 = file_name +'.png'
            im0 = plt.imread(file_name1)[:,:,0:3]
            print('clustering '+ file_name1 + ' ...')
            im = im0[::compresion_ratio,::compresion_ratio,:]
            
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(im)
                
            # genetic parameters
            n_population = n_population_tune
            max_generation = max_generation_tune
            # make a model
            model = ga_base.evolutionary_computing(rep= 'float',
                                                   max_generation= max_generation,
                                                   population_size= n_population,
                                                   k= n_center,
                                                   data= im,
                                                   crossover= 'single-point',
                                                   crossover_prob= 0.9,
                                                   mutation= 'uniform',
                                                   mutation_prob= 0.9,
                                                   ps= ('FPS','roulette_wheel'),
                                                   ss= ('mu_plus_lambda', 1.0),
                                                   result_dir= './results',
                                                   verbose= False,
                                                   )
            
            model.run()
            mgbfa = model.gen_best_fitness_array
            mgbfa_array.append(mgbfa)
            
#            print('model.best_fitness = ', model.best_fitness)
            #print(model.gen_best_fitness_array)
            #    print(model.best_chromosome.reshape(-1,3))
            #    print('model.n_clusters', model.gen_n_cluster)
            #    print(np.asarray(model.gen_best_chromosome).reshape((2,-1,3)))
            bfgn = model.best_fitness_generation_num
            bfgn_array.append(bfgn)

            mbi = model.best_cluster_samples_indexes
            clustered_im = np.zeros((im.shape)).reshape((-1,3))
            for i in range(n_center):
                clustered_im[mbi[i]] = np.array(color_list[i])
            clustered_im = np.reshape(clustered_im, (im.shape))
            plt.subplot(1,2,2)
            plt.imshow(clustered_im)
            file_name = result_image_dir + 'image_' + str(iidx)
            plt.savefig(file_name, dpi=90)
                
        mgbfa_array_array.append(np.mean(np.asarray(mgbfa_array), axis=0))
        bfgn_array_array.append(np.asarray(bfgn_array))
    
    mgbfa_array_array = np.asarray(mgbfa_array_array)
    
    return mgbfa_array_array, bfgn_array_array

