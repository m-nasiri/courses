#*****************************************************************************/
# @file    ga.py
# @author  Majid Nasiri 95340651
# @version V4.0.0
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



def gene_transform(chromosome):
    transformed_chromosome = [chromosome[0] * np.sqrt(np.pi/4),    # 0 =< r =< np.sqrt(np.pi/4)
                              chromosome[1] * 4,    # 0 =< h =< 4
                              ]
    return transformed_chromosome


def evaluate_function(xi):
    # cone surface = T = pi * r * (r + sqrt(r^2 + h^2))  --> miniimize
    # cone volume = V = pi/3 * r^2 * h                   --> maximize
    # base surface = B = pi * r^2  =< 4 m^2
    # h =< 4 m
    # 0 =< r =< np.sqrt(np.pi/4)
    # 0 =< h =< 4
    gen_objective_space_samples = []
    for g in xi:
        chromosome = gene_transform(g)
        r = chromosome[0]
        h = chromosome[1]
        T = np.pi * r * (r + np.sqrt(r**2 + h**2))
        V = np.pi * r**2 * h / 3
        gen_objective_space_samples.append(np.array([T, V]))
        
    gen_objective_space_samples = np.asarray(gen_objective_space_samples)     
    #print('gen_objective_space_samples', gen_objective_space_samples.shape)
    
    # minimize is better => larger arg for minimum values
#    feature_0_sorted = np.argsort(gen_objective_space_features[:,0])[::-1]
#    print(feature_0_sorted)
#    feature_1_sorted = np.argsort(gen_objective_space_features[:,1])
#    print(feature_1_sorted)
    
    gen_dominated_count = []
    for current_sample in gen_objective_space_samples:
        cnt = 0 # number of samples which dominate current sample
        #print(current_sample, 'current_sample')
        for sample in gen_objective_space_samples:
            if (current_sample[0]>=sample[0] and current_sample[1]<=sample[1]):
                #print(sample, 'dominated')
                cnt +=1
                
        gen_dominated_count.append(cnt)
    gen_dominated_count = np.asarray(gen_dominated_count)   
    #print('gen_dominated_count', gen_dominated_count)
    
    return gen_dominated_count, gen_objective_space_samples


class evolutionary_computing():
    def __init__(self, 
                 rep='float',
                 max_generation= None,
                 population_size= None,
                 crossover= None,
                 crossover_prob= None,
                 mutation= None,
                 mutation_prob= None,
                 ps= None,
                 ss= None,
                 result_dir= None,
                 verbose= True,
                 ):
        
        self.representation_type = rep
        self.max_generation = max_generation
        self.population_size = population_size
        self.gene_num = 2

        self.parent_selection_type = ps
        
        
        self.xover_type = crossover
        self.crossover_prob = crossover_prob      
        
        
        self.mutation_type = mutation
        self.mutation_prob = mutation_prob       
        
        self.survivor_selection_type = ss
                
        self.termination = 0
        self.termination_count = 0
        self.termination_max_unchage = 25
        
        self.best_fitness = -float("inf")
        self.gen_fitness_array = []
        self.gen_best_fitness_array = []
        self.generation_array = []
        self.generation_objective_space_array = []
        self.generation_pareto = []
        self.generation_pareto_num = []
        
        self.rdir = result_dir
        if not os.path.exists(self.rdir):
            os.makedirs(self.rdir)
    
        if (verbose == True):
            print('------------------------:---------------------')
            print('    representation type : ', rep)
            print('  parent selection type : ', ps)
            print('         crossover type : ', crossover)
            print('  crossover probability : ', crossover_prob)
            print('          mutation type : ', mutation)
            print('   mutation probability : ', mutation_prob)
            print('survivor selection type : ', ss[0])
            print('------------------------:---------------------')
           
    def fitness(self):
        """
        EVALUATE each candidate
        """        

        self.gen_evaluation, gosa = evaluate_function(self.generation)
        #print('gen_evaluation', self.gen_evaluation)
        # inverse of number of samples which dominate each specific sample as fitness
        self.gen_fitness = 1/self.gen_evaluation
        self.gen_fitness_array.append(self.gen_fitness)
        
        gen = []
        for g in self.generation:
            gen.append(gene_transform(g))
        self.generation_array.append(np.asarray(gen))
        self.generation_objective_space_array.append(gosa)
        index_of_non_dominated_points = np.where(self.gen_evaluation==1)[0]
        pareto_set = [z for z in gosa[index_of_non_dominated_points,:]]
        pareto_set = np.asarray(pareto_set)        
        self.generation_pareto.append(pareto_set)
        self.generation_pareto_num.append(len(pareto_set))
        
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
        
        # scaling gene 1  to fit in [0,1]        
        if (self.representation_type == 'float'):
            self.generation = np.random.uniform(0, 1, size=(self.population_size, self.gene_num))
            
        #print(self.generation.shape)


    def parent_selection(self):
        
        self.offspring_num = int(self.population_size/2)
        
        if self.survivor_selection_type[0] in ['generational', 'round_robin']:
            self.offspring_num = self.population_size
        elif self.survivor_selection_type[0] in ['GENITOR', 'mu_plus_lambda', 'mu_lambda']:
            self.offspring_num = int(int(self.population_size * self.survivor_selection_type[1]/2) * 2)
        
        self.pair_parents_num = int(self.offspring_num/2)
                
        if (len(self.parent_selection_type) == 2):  
            # Probability assignment to each chromosome 
            if (self.parent_selection_type[0] =='FPS'):
                # due to minus negative values in fitness we biased all 
                # values with minimum fitness
                gen_value = self.gen_fitness # - np.min(self.gen_fitness) + 0.001
                # transfer generation values to normalize probability
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
            
            recombination_prob_array = np.random.choice(np.arange(0, 2), size=self.pair_parents_num, p = [1-self.crossover_prob, self.crossover_prob])
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
                new_genes = np.random.uniform(0, 1, size=self.offspring_num)
                
                for off in range(self.offspring_num):
                    if (mutation_probs[off] == True):
                        self.offspring[off, mutation_points[off]] = new_genes[off]
                        
            
            elif (self.mutation_type =='non_uniform'):
                for off in range(self.offspring_num):
                    N = np.random.normal()
                    self.offspring[off, :] = self.offspring[off, :] + N
            
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
            mu_plus_lamda_evaluation,_ = evaluate_function(mu_plus_lamda)
            mu_plus_lamda_fitness = 1/mu_plus_lamda_evaluation
            # sort maximum to minimum            
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness.argsort()[::-1]
            # select best mu individuals for next generation
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness_arg[0:self.population_size]
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = mu_plus_lamda[mu_plus_lamda_fitness_arg, :]
    
        elif (self.survivor_selection_type[0] == 'mu_lambda'):
            # merge current generation with offsprings
            offspring_evaluation,_ = evaluate_function(self.offspring)
            offspring_fitness = 1/offspring_evaluation
            # sort maximum to minimum            
            offspring_fitness_arg = offspring_fitness.argsort()[::-1]
            # select best mu individuals for next generation
            offspring_fitness_arg = offspring_fitness_arg[0:self.population_size]
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = self.offspring[offspring_fitness_arg, :]            

        elif (self.survivor_selection_type[0] == 'round_robin'):
            # merge current generation with offsprings
            round_robin = np.concatenate((self.generation, self.offspring))
            round_robin_evaluation,_ = evaluate_function(round_robin)
            round_robin_fitness = 1/round_robin_evaluation
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


    def run(self, display='off'):
        """
        largest quality smaller than self.cost_max is desire solution
        """
        
        #INITIALISE population with random candidate solutions
        self.initialize()
        
        #EVALUATE each candidate
        self.fitness()
        
        #REPEAT UNTIL ( TERMINATION CONDITION is satisfied )
        self.gen = 0
        while(self.termination == False):
            
            self.gen +=1
            print('generation = ', self.gen)

            #SELECT parents
            self.parent_selection()
            
            #RECOMBINE pairs of parents
            self.crossover()
            
            #MUTATE the resulting offspring 
            self.mutation()
            
            #EVALUATE new candidates            
            self.fitness()

            # SELECT individuals for the next generation
            self.survivor_selection()            
               
            # check termination cretaria
            self.termination_check()
    



 



    


