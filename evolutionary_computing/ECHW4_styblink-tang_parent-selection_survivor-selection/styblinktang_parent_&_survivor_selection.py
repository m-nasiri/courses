#*****************************************************************************/
# @file    sty.py
# @author  Majid Nasiri 95340651
# @version V2.0.0
# @date    29 May 2017
# @brief   
#*****************************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import styblinktang



plt.close("all")
result_dir = './results'
xover_list = ['single']
mutation_list = ['uniform']

#xover_list = ['single',
#              'simple',
#              'whole',
#              'blend'
#              ]

#mutation_list = ['uniform', 
#                 'non_uniform', 
#                 'uncorr_one_sigma',
#                 'uncorr_n_sigma',
#                 'corr'
#                 ]


parent_selection_list = [('FPS',                    'roulette_wheel'),
                         ('FPS',                    'SUS'),
                         ('linear_ranking',         'roulette_wheel'),
                         ('linear_ranking',         'SUS'),
                         ('exponential_ranking',    'roulette_wheel'),
                         ('exponential_ranking',    'SUS'),
                         ('best_n_of_k',            2, 5),      # 2 best of 5 random
                         ('best_n_of_k',            1, 5),      # 1 best of 5 random
#                         ('uniform')    # not developed yet
                         ]

survivor_selection_list = [#('generational',     None),
#                           ('fifo',             None),  # not developed yet
#                           ('random',           None), # not developed yet
                           ('GENITOR',          0.5),   # worst 50% replaced by new offspring
#                           ('elitism',          None),  # not developed yet
                           ('round_robin',      None),
                           ('mu_plus_lambda',    0.4),   # best mu number of mu+lamba will be remain (lamba = 40% * mu)
                           ('mu_lambda',         1.4),   # best mu number of lamba will be remain (lamba = 1.4 * mu)
                           ]


for parent_selection in parent_selection_list:
    all_models_best_fitness_array_mean = []
    fig1 = plt.figure(figsize=(12,6))
    for survivor_selection in survivor_selection_list:
        # for more exploring about behaviour of models 
        # we have run 20 examples for each model and take 
        # mean over their best fitness arrays
        model_best_fitness_array = []
        for itr in range(3):
            # create a model with hyperparameters 
            stylinski_model = styblinktang.evolutionary_computing(rep ='float',
                                                     crossover = 'single',
                                                     mutation = 'uncorr_n_sigma',
                                                     ps = parent_selection,
                                                     ss = survivor_selection,
                                                     result_dir = result_dir
                                                     )
 
            # run the model 
            stylinski_model.run(display='off')            
            model_best_fitness_array.append(stylinski_model.best_fitness_array)
            print('        best chromosome :', stylinski_model.best_chromosome)
            print('           best fitness :', stylinski_model.best_fitness)
            print('------------------------:-------------------')
            
        model_best_fitness_array = np.asarray(model_best_fitness_array)
        model_best_fitness_array_mean = np.mean(model_best_fitness_array, axis=0)
        label = 'ss_type='+stylinski_model.survivor_selection_type[0]
        plt.plot(model_best_fitness_array_mean, label=label)
        
    plt.legend()
    plt.title('parent selection type = '+ \
              stylinski_model.parent_selection_type[0]+\
              ' & '+\
              str(stylinski_model.parent_selection_type[1]))
    
    fig1.savefig(result_dir+'/parent_selection_type='+ \
                stylinski_model.parent_selection_type[0]+\
                ' & '+\
                str(stylinski_model.parent_selection_type[1])+\
                '.png',
                dpi=fig1.dpi)





for survivor_selection in survivor_selection_list:
    all_models_best_fitness_array_mean = []
    fig1 = plt.figure(figsize=(12,6))
    for parent_selection in parent_selection_list:
        # for more exploring about behaviour of models 
        # we have run 20 examples for each model and take 
        # mean over their best fitness arrays
        model_best_fitness_array = []
        for itr in range(100):
            # create a model with hyperparameters 
            stylinski_model = styblinktang.evolutionary_computing(rep ='float',
                                                     crossover = 'single',
                                                     mutation = 'uncorr_n_sigma',
                                                     ps = parent_selection,
                                                     ss = survivor_selection,
                                                     result_dir = result_dir,
                                                     )
 
            # run the model 
            stylinski_model.run(display='off')            
            model_best_fitness_array.append(stylinski_model.best_fitness_array)
            print('        best chromosome :', stylinski_model.best_chromosome)
            print('           best fitness :', stylinski_model.best_fitness)
            print('------------------------:-------------------')
            
        model_best_fitness_array = np.asarray(model_best_fitness_array)
        model_best_fitness_array_mean = np.mean(model_best_fitness_array, axis=0)
        label ='ps_type='+stylinski_model.parent_selection_type[0]+\
        ' & '+ str(stylinski_model.parent_selection_type[1])
        plt.plot(model_best_fitness_array_mean, label=label)
        
    plt.legend()
    plt.title('survivor selection type = '+ stylinski_model.survivor_selection_type[0])
    
    fig1.savefig(result_dir+'/survivor_selection_type='+\
                 stylinski_model.survivor_selection_type[0]+'.png',
                dpi=fig1.dpi)


 



    


