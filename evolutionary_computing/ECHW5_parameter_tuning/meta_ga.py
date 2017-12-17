#*****************************************************************************/
# @file    stylinktang.py
# @author  Majid Nasiri 95340651
# @version V3.0.0
# @date    15 December 2017
# @brief   
#*****************************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import ga


def gene_transform(gene):
    gene_list = [int(gene[0] * 40 + 50),
                 gene[1] / 2 + 0.5,
                 gene[2] / 2 + 0.5]
    return gene_list
        

class parameter_tuning():
    def __init__(self,
                 meta_rep,
                 meta_max_generation,       # meta ga parameters
                 meta_population_size,
                 meta_crossover,
                 meta_crossover_prob,
                 meta_mutation,
                 meta_mutation_prob,
                 meta_ps,
                 meta_ss,
                 meta_utility_type,
                 B,                     # number of tests per vector  
                 base_rep,              # base ga parameters
                 base_max_generation,   # C number of fitness evaluations per tests
                 base_population_size,
                 base_crossover,
                 base_crossover_prob,
                 base_mutation,
                 base_mutation_prob,
                 base_ps,
                 base_ss,                                         
                 result_dir= './results',
                 verbose = True,
                 ):
        
        
        self.representation_type = meta_rep
        self.max_generation = meta_max_generation
        self.population_size = meta_population_size

        # gene 1 = base_population_size     interval=(10,90)
        # gene 2 = base_crossover_prob      interval=(0,1)
        # gene 2 = base_mutation_prob       interval=(0,1)
        self.gene_num = 3
        self.parent_selection_type = meta_ps
        self.xover_type = meta_crossover
        self.crossover_prob = meta_crossover_prob
        self.mutation_type = meta_mutation
        self.mutation_prob = meta_mutation_prob
        self.mutation_prob = 1/self.gene_num        
        self.survivor_selection_type = meta_ss
        self.B = B
        self.termination = 0
        self.termination_count = 0
        self.termination_max_unchage = 25
        self.utility_type = meta_utility_type
        self.best_fitness = -float("inf")
        self.gen_fitness_array = []
        self.gen_best_fitness_array = []
        
        self.rdir = result_dir
        if not os.path.exists(self.rdir):
            os.makedirs(self.rdir)
            
        if (verbose == True):
            print('-----------------------------:------------------------')
            print('      meta presentation type : ', meta_rep)
            print('  meta parent selection type : ', meta_ps)
            print('         meta crossover type : ', meta_crossover)
            print('  meta crossover probability : ', meta_crossover_prob)
            print('          meta mutation type : ', meta_mutation)
            print('   meta mutation probability : ', meta_mutation_prob)
            print('meta survivor selection type : ', meta_ss[0])
            print('-----------------------------:------------------------')
        
        self.base_representation_type = base_rep
        self.base_max_generation = base_max_generation
        self.base_population_size = base_population_size
        self.base_parent_selection_type = base_ps
        self.base_xover_type = base_crossover
        self.base_crossover_prob = base_crossover_prob
        self.base_mutation_type = base_mutation
        self.base_mutation_prob = base_mutation_prob
        self.base_mutation_prob = 1/self.gene_num        
        self.base_survivor_selection_type = meta_ss
        
    def evaluate_function(self, xi, verbose):
        g_evaluation_array = []
        for g in xi:
            g_list = gene_transform(g)
            
            #print('g_list=', g_list)
            # gene 1 = gene1 * 40 + 50  to fit in interval (10,90)
            # gene 2 = gene2 / 2 + 0.5  to fit in interval (0,1)
            # gene 3 = gene3 / 2 + 0.5  to fit in interval (0,1)
            styblinktang_model = ga.evolutionary_computing(rep= self.base_representation_type,
                                                           max_generation = self.base_max_generation,
                                                           population_size = g_list[0],
                                                           crossover= self.xover_type,
                                                           crossover_prob = g_list[1],
                                                           mutation= self.base_mutation_type,
                                                           mutation_prob = g_list[2],
                                                           ps = self.base_parent_selection_type,
                                                           ss = self.base_survivor_selection_type,
                                                           result_dir = self.rdir,
                                                           verbose= verbose,
                                                           )
            styblinktang_model.run()
            g_evaluation_array.append(styblinktang_model.best_fitness)
                
       
        return np.asarray(g_evaluation_array)
    
    def utility(self):
        """
        EVALUATE each candidate
        """ 
        
        if (self.utility_type == "MBF"):
            gen_evaluation_array = []
            for i in range(self.B):
                #if i==0:    verbose = True
                #else:       verbose = False
                verbose = False
                gen_evaluation = self.evaluate_function(self.generation, verbose=verbose)
                gen_evaluation_array.append(gen_evaluation)
            gen_evaluation_array = np.asarray(gen_evaluation_array)
            self.gen_evaluation = np.mean(gen_evaluation_array, axis=0)
            
            #print(self.gen_evaluation)
            self.gen_fitness = self.gen_evaluation.copy()
            self.gen_fitness_array.append(self.gen_fitness)
            
            # save best fitness
            self.gen_best_fitness_arg = self.gen_fitness.argmax()
            self.gen_best_fitness = self.gen_fitness.max()
            self.gen_best_fitness_array.append(self.gen_best_fitness)
            if (self.gen_best_fitness > self.best_fitness):
                self.best_fitness = self.gen_best_fitness 
                self.best_chromosome = gene_transform(self.generation[self.gen_best_fitness_arg])
            
    def initialize(self):
        """
        INITIALISE population with random candidate solutions
        """
        # scaling gene 1  to fit in [-1,1]        
        if (self.representation_type == 'float'):
            self.generation = np.random.uniform(-1, 1, size=self.gene_num)
            for _ in range(self.population_size-1):
                self.generation = np.vstack([self.generation, np.random.uniform(-1, 1, size=self.gene_num)])

        #print(self.generation.shape)

    def parent_selection(self):
        
        self.offspring_num = int(self.population_size/2)
        
        if self.survivor_selection_type[0] in ['generational', 'round_robin']:
            self.offspring_num = self.population_size
        elif self.survivor_selection_type[0] in ['GENITOR', 'mu_plus_lambda', 'mu_lambda']:
            self.offspring_num = int(self.population_size * self.survivor_selection_type[1])
        
        self.pair_parents_num = int(self.offspring_num/2)
                
        if (len(self.parent_selection_type) == 2):  
            # Probability assignment to each chromosome 
            if (self.parent_selection_type[0] =='FPS'):
                # due to minus negative values in fitness we biased all 
                # values with minimum fitness
                gen_value = self.gen_fitness - np.min(self.gen_fitness) + 0.001
                # transfer generation values to normalized probability
                self.gen_probability = gen_value / np.sum(gen_value)
                
            elif (self.parent_selection_type[0] =='linear_ranking'):
                gen_worst_to_best_idxs = self.gen_fitness.argsort()            
                s = 1.5 #s = 2
                i = np.arange(0, self.population_size)
                prob = (2-s)/self.population_size + (2*i*(s-1))/(self.population_size*(self.population_size-1)) 
                self.gen_probability = np.zeros((self.population_size))
                self.gen_probability[gen_worst_to_best_idxs] = prob
                
                # make probability normal
                self.gen_probability = self.gen_probability / np.sum(self.gen_probability)
                #print('gen_probability', self.gen_probability)
                
            elif (self.parent_selection_type[0] =='exponential_ranking'):
                gen_worst_to_best_idxs = self.gen_fitness.argsort()
                i = np.arange(0, self.population_size)
                c = self.population_size
                prob = (1-np.exp(-i))/c
                self.gen_probability = np.zeros((self.population_size))
                self.gen_probability[gen_worst_to_best_idxs] = prob
                # make probability normal
                self.gen_probability = self.gen_probability / np.sum(self.gen_probability)
                  
                
            # Implementing Selection Probabilities
            if (self.parent_selection_type[1] == 'roulette_wheel'):
                # roulette wheel
                # calculate cumulative probability
                cumulative_probability = np.cumsum(self.gen_probability)                
                self.parents = np.zeros((self.offspring_num), dtype=np.int32)
                cm = 0
                while(cm < self.offspring_num):
                    r = np.random.uniform(0, 1, size=1)
                    i = 0
                    while(cumulative_probability[i] < r):
                        i = i + 1
                    self.parents[cm] = i
                    cm = cm + 1
                    
                self.pair_parents = np.reshape(self.parents, (-1, 2))
                self.offspring = self.generation[self.parents]
            
            elif (self.parent_selection_type[1] == 'SUS'):
                # stochastic universal sampling (SUS)
                # calculate cumulative probability
                cumulative_probability = np.cumsum(self.gen_probability)
                self.parents = np.zeros((self.offspring_num), dtype=np.int32)
                cm = 0
                i = 0
                while(cm < self.offspring_num):
                    r = np.random.uniform(0, 1/self.offspring_num, size=1)
                    while(r <= cumulative_probability[i] and cm < self.offspring_num):
                        self.parents[cm] = i
                        r = r + 1/self.offspring_num
                        cm = cm + 1
                    i = i + 1
                    
                self.pair_parents = np.reshape(self.parents, (-1, 2))
                self.offspring = self.generation[self.parents]
                    
                
                
        else:
            # Tournament Selection
            if (self.parent_selection_type[0] == 'best_n_of_k'):
                # pick k individuals randomly without replacement
                #print('parent_selection_type', self.parent_selection_type)
                n = self.parent_selection_type[1]
                k = self.parent_selection_type[2]
                random_k_parents = np.random.choice(np.arange(0, self.population_size), size=(self.offspring_num, k))
                # sort maximum to minimum
                parents_pack = np.argsort(-self.gen_fitness[random_k_parents], axis=1)[:,0:n]                
                self.parents = np.reshape(parents_pack, (1,-1))[0]
                self.pair_parents = np.reshape(self.parents, (-1, 2)) 
                self.offspring = self.generation[self.parents]
                
                

    def crossover(self):
        
            recombination_prob_array = np.random.choice(np.arange(0, 2) , size=self.pair_parents_num, p = [1-self.crossover_prob, self.crossover_prob])
            if (self.xover_type =='single'):
                xover_point = np.random.randint(self.gene_num, size=self.pair_parents_num)                
                # reshape offspring (or parents to pair parents)
                self.offspring = np.reshape(self.offspring, (-1, 2, self.gene_num))
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(self.pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        self.offspring[off, :, xover_point[off]] = ((  alpha) * self.offspring[off, 0, xover_point[off]]) + ((1-alpha) * self.offspring[off, 1, xover_point[off]])
    
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))
                
                
            elif (self.xover_type =='simple'):
                xover_point = np.random.randint(self.gene_num, size=self.pair_parents_num)
                self.offspring = np.reshape(self.offspring, (-1, 2, self.gene_num))
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(self.pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        for xpi in range(xover_point[off], self.gene_num):
                            self.offspring[off, :, xpi] = (alpha * self.offspring[off, 0, xpi])+((1-alpha) * self.offspring[off, 1, xpi])
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))

                
    def mutation(self):
        
            mutation_probs = np.random.choice(np.arange(0, 2) , size=self.offspring_num, p = [1-self.mutation_prob, self.mutation_prob])
            if (self.mutation_type =='uniform'):
                mutation_points = np.random.randint(self.gene_num, size=self.offspring_num)   
                new_genes = np.random.uniform(-1, 1, size=self.offspring_num)
                
                for off in range(self.offspring_num):
                    if (mutation_probs[off] == True):
                        self.offspring[off, mutation_points[off]] = new_genes[off]
                        
            
                    
    def survivor_selection(self):            
        if (self.survivor_selection_type[0] == 'generational'):                        
            # all generation have been replaced with new offsprings 
            self.generation = self.offspring.copy()
            
        elif (self.survivor_selection_type[0] == 'GENITOR'):
            # sort maximum to minimum
            gen_new_idxs = self.gen_fitness.argsort()[::-1]   
            # save indices of best chromosomes
            gen_new_idxs = gen_new_idxs[0:(self.population_size-self.offspring_num)]      
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = np.concatenate((self.generation[gen_new_idxs], self.offspring))

        elif (self.survivor_selection_type[0] == 'mu_plus_lambda'):
            # merge current generation with offsprings
            mu_plus_lamda = np.concatenate((self.generation, self.offspring))
            mu_plus_lamda_evaluation = self.evaluate_function(mu_plus_lamda)
            mu_plus_lamda_fitness = - mu_plus_lamda_evaluation
            # sort maximum to minimum            
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness.argsort()[::-1]
            # select best mu individuals for next generation
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness_arg[0:self.population_size]
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = mu_plus_lamda[mu_plus_lamda_fitness_arg, :]
    
        elif (self.survivor_selection_type[0] == 'mu_lambda'):
            # merge current generation with offsprings
            offspring_evaluation = self.evaluate_function(self.offspring)
            offspring_fitness = - offspring_evaluation
            # sort maximum to minimum            
            offspring_fitness_arg = offspring_fitness.argsort()[::-1]
            # select best mu individuals for next generation
            offspring_fitness_arg = offspring_fitness_arg[0:self.population_size]
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = self.offspring[offspring_fitness_arg, :]            

        elif (self.survivor_selection_type[0] == 'round_robin'):
            # merge current generation with offsprings
            round_robin = np.concatenate((self.generation, self.offspring))
            round_robin_evaluation = self.evaluate_function(round_robin)
            round_robin_fitness = - round_robin_evaluation
            q = 10
            random_individuals = np.random.choice(np.arange(0, len(round_robin_fitness)) , size=(len(round_robin_fitness), q))
            round_robin_random_fitness = round_robin_fitness[random_individuals]
            number_of_smaller_fitnesses_array = np.zeros((len(round_robin_fitness),), dtype=np.int32)
            for indv in range(len(round_robin_fitness)):
                idxs_of_smaller_fitnesses = np.where( round_robin_fitness[indv] > round_robin_random_fitness[indv,:])
                number_of_smaller_fitnesses_array[indv] = len(idxs_of_smaller_fitnesses[0])
            # sort maximum to minimum 
            number_of_smaller_fitnesses_array_arg = number_of_smaller_fitnesses_array.argsort()[::-1]
            # select best mu individuals for next generation
            best_indvidual_indxs = number_of_smaller_fitnesses_array_arg[0:self.population_size]
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = round_robin[best_indvidual_indxs, :]  

    def termination_check(self):
        
        if (self.gen == self.max_generation):
            self.termination = True
        else:
            # check variance of 5 latest fitness array 
            if (len(self.gen_best_fitness_array) > 5):
                gen_best_fitness_array_variance = np.var(self.gen_best_fitness_array[-5:])
                print("gen_best_fitness_array_variance", gen_best_fitness_array_variance)
                if (gen_best_fitness_array_variance < 0.2):
                    self.termination = True

        
    def run(self, display='off'):
        """
        largest quality smaller than self.cost_max is desire solution
        """
        
        #INITIALISE population with random candidate solutions
        self.initialize()
        
        #EVALUATE each candidate
        self.utility()
        
        #REPEAT UNTIL ( TERMINATION CONDITION is satisfied )
        self.gen = 0
        while(self.termination == False):
            
            self.gen +=1
            print('meta generation = ', self.gen)
            
            #SELECT parents
            self.parent_selection()
            
            #RECOMBINE pairs of parents
            self.crossover()
            
            #MUTATE the resulting offspring 
            self.mutation()
            
            #EVALUATE new candidates            
            self.utility()

            # SELECT individuals for the next generation
            self.survivor_selection()            

            # check termination cretaria
            self.termination_check()
                
            
        self.gen_fitness_array = np.asarray(self.gen_fitness_array)
    



 



    


