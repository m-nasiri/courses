#*****************************************************************************/
# @file    main.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    22 Dec 2017
# @brief   multiobjective problem
#*****************************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ga


plt.close("all")
result_dir = './results'
xover_list = ['single']
mutation_list = ['uniform']
parent_selection_list = [('FPS', 'roulette_wheel')]
#survivor_selection_list = [('mu_plus_lambda',    1.5),
#                           ('generational',    None),]


#parent_selection_list = [('FPS',                    'roulette_wheel'),
#                         ('FPS',                    'SUS'),
#                         ('linear_ranking',         'roulette_wheel'),
#                         ('linear_ranking',         'SUS'),
#                         ('exponential_ranking',    'roulette_wheel'),
#                         ('exponential_ranking',    'SUS'),
#                         ('best_n_of_k',            2, 5),      # 2 best of 5 random
#                         ('best_n_of_k',            1, 5),      # 1 best of 5 random
#                         ]
#
survivor_selection_list = [('generational',     None),
                           ('GENITOR',          0.5),   # worst 50% replaced by new offspring
                           ('round_robin',      None),
                           ('mu_plus_lambda',    1.4),   # best mu number of mu+lamba will be remain (lamba = 40% * mu)
                           ('mu_lambda',         1.4),   # best mu number of lamba will be remain (lamba = 1.4 * mu)
                           ]


for ps in parent_selection_list:
    for ss in survivor_selection_list:
        # create a model with hyperparameters 
        multobjective_model = ga.evolutionary_computing(rep ='float',
                                                        max_generation = 20,
                                                        population_size = 20*4,
                                                        crossover = xover_list[0],
                                                        crossover_prob = 0.9,
                                                        mutation = mutation_list[0],
                                                        mutation_prob = 0.9,
                                                        ps = ps,
                                                        ss = ss,
                                                        result_dir = result_dir,
                                                        verbose = True,
                                                        )

        # run the model 
        multobjective_model.run()            
        gar = multobjective_model.generation_array
        gp = multobjective_model.generation_pareto
        garr = np.asarray(gar[1:])
        generation_variance_array = [np.var(g) for g in garr]
        plt.plot(generation_variance_array, label='survival selection type= '+ss[0])
        plt.xlabel('Generation')
        plt.ylabel('Variance of generation chromosomes')
        plt.legend()
        plt.grid()
        

#        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,)
#        cmap = plt.get_cmap('tab20')
#        for j,g in enumerate(gar[1:]): #[gar[-1]]
#            ax1.scatter(g[:,0], g[:,1], s=40, cmap=cmap, edgecolors='none', label='generation= '+str(j))
#        ax1.legend()
#        ax1.set_title('Decision Space')
#        ax1.set_xlabel('Radius of cone')
#        ax1.set_ylabel('Height of cone')
#        ax1.set_xlim([0, 0.9])
#        ax1.set_ylim([0, 4])
#        ax1.grid(True)
#
#        for j,g in enumerate(gp[1:]): #[gp[-1]]
#            ax2.scatter(g[:,0], g[:,1], s=40, cmap=cmap, edgecolors='none', label='generation= '+str(j))        
#        ax2.legend()
#        ax2.set_title('Objective Space')
#        ax2.set_xlabel('Volume of cone')
#        ax2.set_ylabel('Surface of cone')        
#        ax2.set_xlim([0, 13])
# #       ax2.set_ylim([0, 12])
#        ax2.grid(True)
#        
#        fig.set_size_inches(14, 8)
#        fig.savefig(result_dir+'/#parent_selection='+ps[0]+\
#                    '__#survival_selection='+ss[0]+'.jpg', dpi=100)
#        plt.close('all')

