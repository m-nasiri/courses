#*****************************************************************************/
# @file    Tuning.py
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
import meta_ga


A = None  # number of vectors tested
B = 10    # number of tests per vector
C = 100   # number of fitness evaluations per tests

# max

plt.close("all")
result_dir = './results/'


meta_representation = 'float'
meta_xover_list = ['single']
#meta_xover_list = ['simple']
meta_mutation_list = ['uniform']
meta_parent_selection_list = [('FPS','roulette_wheel')]  # FPS & roulette wheel
meta_survivor_selection_list = [('GENITOR', 0.5)]        # worst 50% replaced by new offspring
meta_parent_selection = meta_parent_selection_list[0]
meta_survivor_selection = meta_survivor_selection_list[0]


base_representation = 'float'
#base_xover_list = ['single']
base_xover_list = ['simple']
base_mutation_list = ['uniform']
base_parent_selection_list = [('best_n_of_k', 2, 5)]     # 2 best of 5 random
base_survivor_selection_list = [('GENITOR', 0.5)]        # worst 50% replaced by new offspring
base_parent_selection = base_parent_selection_list[0]
base_survivor_selection = base_survivor_selection_list[0]


meta_max_generation = 200
meta_population_size = 4*2
# create a meta model with hyperparameters 
meta_ga_model = meta_ga.parameter_tuning(meta_rep = meta_representation,
                                         meta_max_generation= meta_max_generation,    
                                         meta_population_size= meta_population_size,
                                         meta_crossover= meta_xover_list[0],
                                         meta_crossover_prob= 0.8,
                                         meta_mutation= meta_mutation_list[0],
                                         meta_mutation_prob= 0.6,
                                         meta_ps= meta_parent_selection,
                                         meta_ss= meta_survivor_selection,
                                         meta_utility_type= "MBF",
                                         B= B,
                                         base_rep= base_representation,
                                         base_max_generation= C,
                                         base_population_size= None,
                                         base_crossover= base_xover_list[0],
                                         base_crossover_prob= None,
                                         base_mutation= base_mutation_list[0],
                                         base_mutation_prob= None,
                                         base_ps= base_parent_selection,
                                         base_ss= base_survivor_selection,
                                         result_dir= result_dir,
                                         verbose = False
                                         )

meta_ga_model.run()
print(meta_ga_model.best_chromosome)
print(meta_ga_model.best_fitness)
#print(meta_ga_model.gen_fitness_array)

fig1 = plt.figure(1)
plt.plot(meta_ga_model.gen_fitness_array.max(axis=1))
plt.xlabel("generations")
plt.ylabel("best fitness")
plt.grid()
fig1.savefig(result_dir+ \
             'meta_xovr='+ meta_xover_list[0]+ \
             '__meta_mutation='+ meta_mutation_list[0]+ \
             '__base_xovr='+ base_xover_list[0]+ \
             '__base_mutation='+ base_mutation_list[0]+ \
             '_best.png',
             dpi=fig1.dpi)


generation_number = meta_ga_model.gen + 1
fig2 = plt.figure(2)
for gen in range(generation_number):
    plt.scatter(gen*np.ones((1, meta_population_size)), 
                meta_ga_model.gen_fitness_array[gen].T,
                s = 10,
                marker = '.')

plt.plot(meta_ga_model.gen_fitness_array.max(axis=1), label='best fitness')
plt.xlabel("generations")
plt.ylabel("population fitnesses")
plt.legend(loc='center right')
plt.title("best performance = " + str(meta_ga_model.best_fitness))
fig2.savefig(result_dir+ \
             'meta_xovr='+ meta_xover_list[0]+ \
             '__meta_mutation='+ meta_mutation_list[0]+ \
             '__base_xovr='+ base_xover_list[0]+ \
             '__base_mutation='+ base_mutation_list[0]+ \
             '.png',
             dpi=fig2.dpi)


