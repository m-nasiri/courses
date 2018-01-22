#*****************************************************************************/
# @file    ga_base.py
# @author  Majid Nasiri 95340651
# @version V3.0.0
# @date    15 December 2017
# @brief   
#*****************************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os



class evolutionary_computing():
    def __init__(self,
                 rep='float',
                 max_generation = None,
                 population_size = None,
                 k = None,         # number of cluster
                 data = None,   # input image for clustering
                 crossover='single-point',
                 crossover_prob = None,
                 mutation='uniform',
                 mutation_prob = None,
                 ps = ('FPS','roulette_wheel'),
                 ss = ('generational', None),
                 result_dir = './results',
                 verbose = True,
                 ):
        
        self.representation_type = rep
        self.max_generation = max_generation
        self.population_size = population_size
        self.gene_num = 3 * k   # row and column
        self.k = k
        self.data = data
        
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
        self.gen_best_chromosome = []
        
        self.verbose = verbose
        self.rdir = result_dir
        if not os.path.exists(self.rdir):
            os.makedirs(self.rdir)
    
        if (self.verbose == True):
            print('------------------------:---------------------')
            print('    representation type : ', rep)
            print('  parent selection type : ', ps)
            print('         crossover type : ', crossover)
            print('  crossover probability : ', crossover_prob)
            print('          mutation type : ', mutation)
            print('   mutation probability : ', mutation_prob)
            print('survivor selection type : ', ss[0])
            print('------------------------:---------------------')
           
    def initialize(self):
        """
        INITIALISE population with random candidate solutions
        """
        row, column, _ = self.data.shape
        if (self.representation_type == 'float'):
            rs = np.random.choice(self.data.shape[0]*self.data.shape[1],
                                  size=(self.population_size, self.k), 
                                  replace=False)
            self.rs = rs
            rsr = np.reshape(rs, (-1,))
            pixels = np.reshape(self.data, (-1,3))
            centers = pixels[rsr]            
            self.generation = np.reshape(centers, (-1,self.gene_num))            
#            print(self.generation) # 4*9
    
    def evaluate_function(self, generation):
        generation_out = generation.copy()
        population_size = generation.shape[0]
        k = generation.shape[1]//3
        # xi = pixels   # 380 * 3
        # zj = centers  # 16 * 3
        xi = np.reshape(self.data, (-1, 3))
        zj = np.reshape(generation, (-1, 3))

        #print(xi.shape)
        #print(zj.shape)
        norm_array = []     # 4 * 5 * 380
        for _,cntr in enumerate(zj):
            distance = xi-cntr   # distance of k centers from all pixels
            #print(distance.shape)
            norm_array.append(np.linalg.norm(distance, axis=1))
            
        norm_array = np.asarray(norm_array)
        norm_array = norm_array.reshape((population_size, k, -1))
#        print('norm_shape', norm_array.shape)  # 4 * 5 * 380
        norm_array_closest_cluster = np.argmin(norm_array, axis=1) # 4 * 380
        #print(norm_array_closest_cluster.shape)
        
#        gen_evaluation = np.zeros(population_size,)
        gen_objective_space_samples = []
        b_w_cluster_var = []
        gen_n_cluster = np.zeros((population_size, self.k))
        gen_cluster_samples_indexes = []
        gen_within_cluster_var = []
        gen_between_cluster_var = []
        for indv in range(population_size):
#            print("-------------------")
            within_cluster_var = []
            cluster_new_center = []
            indv_cluster_samples_indexes = []
#            self.n_clusters = []
            for ki in range(k):
                cluster_samples_indexs = np.where(norm_array_closest_cluster[indv,:] == ki)[0]
                indv_cluster_samples_indexes.append(cluster_samples_indexs)
#                print(cluster_samples_indexs.shape)
                n_current_cluster = cluster_samples_indexs.shape[0]
                gen_n_cluster[indv, ki] = n_current_cluster
#                self.n_clusters.append(n_current_cluster)
                
                current_center = generation[indv, (3*ki):3*(ki+1)]

                current_cluster_samples = xi[cluster_samples_indexs]

                if (n_current_cluster == 0):
                    current_cluster_new_center = current_center
                    #current_within_cluster_var = 1 # a large number
                    current_within_cluster_var_n = 1 # a large number
                    
                else:
                    current_cluster_new_center = np.mean(current_cluster_samples, axis=0)
#                    current_within_cluster_var = np.var(current_cluster_samples)                 
                    current_within_cluster_var_n = (current_cluster_samples-current_cluster_new_center)**2 / n_current_cluster
                    current_within_cluster_var_n = np.sum(current_within_cluster_var_n)
                    #print('current_within_cluster_var', current_within_cluster_var)
                    #print('current_within_cluster_nvar',current_within_cluster_nvar)
                    
                within_cluster_var.append(current_within_cluster_var_n)
                #print(current_cluster_samples.shape)
                
#                print('current_center', current_center)
#                print('current_cluster_new_center', current_cluster_new_center)
                cluster_new_center.append(current_cluster_new_center)
                #print(current_cluster_new_center)

                
                # replace current centers with new centers
                generation_out[indv, (3*ki):3*(ki+1)] = current_cluster_new_center
                    
            
            gen_cluster_samples_indexes.append(indv_cluster_samples_indexes)
#            print('within_cluster_var', within_cluster_var)
            within_cluster_var = np.sum(within_cluster_var)
            gen_within_cluster_var.append(within_cluster_var)
            cluster_new_center = np.asarray(cluster_new_center)
            between_cluster_var = np.var(cluster_new_center)
            gen_between_cluster_var.append(between_cluster_var)
#            print('cluster_new_center', cluster_new_center)
            current_b_w_cluster_var = between_cluster_var - within_cluster_var
            b_w_cluster_var.append(current_b_w_cluster_var)
            
            gen_objective_space_samples.append(np.array([within_cluster_var, between_cluster_var]))
            
        
        if (self.verbose == True):
            for indv in range(population_size):
                print(gen_within_cluster_var[indv],'    ',                      
                      gen_between_cluster_var[indv],'    ',
                      b_w_cluster_var[indv],'    ',
                      gen_n_cluster[indv])
        
#        gen_objective_space_samples = np.asarray(gen_objective_space_samples)         
#        gen_dominated_count = []
#        for current_sample in gen_objective_space_samples:
#            cnt = 0 # number of samples which dominate current sample
#            #print(current_sample, 'current_sample')
#            for sample in gen_objective_space_samples:
#                if (current_sample[0]>=sample[0] and current_sample[1]<=sample[1]):
#                    #print(sample, 'dominated')
#                    cnt +=1
#                    
#            gen_dominated_count.append(cnt)
#        gen_dominated_count = np.asarray(gen_dominated_count)   
        
#        gen_evaluation = gen_dominated_count        
        gen_evaluation = np.asarray(b_w_cluster_var)
        
#        print('gen_evaluation', gen_evaluation)
        gen_within_cluster_var = np.asarray(gen_within_cluster_var)
#        print('gen_within_cluster_var',1/gen_within_cluster_var)
        gen_evaluation = 1/gen_within_cluster_var
        
        return gen_evaluation, generation_out, gen_cluster_samples_indexes

    def record(self):
        """
        RECORD status of genetic algorithm after changing generation population 
        or generation fitness
        """

        self.gen_fitness_array.append(self.gen_fitness)
        
        # save best fitness
        self.gen_best_fitness_arg = self.gen_fitness.argmax()
        self.gen_best_fitness = self.gen_fitness.max()
        self.gen_best_fitness_array.append(self.gen_best_fitness)
        self.gen_best_chromosome.append(self.generation[self.gen_best_fitness_arg])
        if (self.gen_best_fitness > self.best_fitness):
            self.best_fitness = self.gen_best_fitness
            self.best_chromosome = self.generation[self.gen_best_fitness_arg]
            self.best_cluster_samples_indexes = self.gen_cluster_samples_indexes[self.gen_best_fitness_arg]
            self.best_fitness_generation_num = self.gen-1
#        print('self.gen_best_fitness_arg', self.gen_best_fitness_arg)
#        print('self.gen_cluster_samples_indexes', self.best_cluster_samples_indexes)
#        print('self.gen_fitness', self.gen_fitness)        
#        print('self.gen_best_fitness', self.gen_best_fitness)
#        print('self.best_fitness', self.best_fitness)


    def fitness(self):
        """
        EVALUATE each candidate
        """        
#        print('------------------------- fitness calculation ---- start')
        self.gen_evaluation, self.generation, self.gen_cluster_samples_indexes = self.evaluate_function(self.generation)
#        print('self.gen_evaluation', self.gen_evaluation)
        self.gen_fitness = self.gen_evaluation.copy()
#        print('self.gen_fitness', self.gen_fitness) 
          
#        print('------------------------- fitness calculation ---- end')
    
    def parent_selection(self):
        self.offspring_num = int(self.population_size/2)
        
        if self.survivor_selection_type[0] in ['generational', 'round_robin']:
            self.offspring_num = self.population_size
        elif self.survivor_selection_type[0] in ['GENITOR', 'mu_plus_lambda', 'mu_lambda']:
            self.offspring_num = int((self.population_size * self.survivor_selection_type[1]//2)*2)
        
        self.pair_parents_num = int(self.offspring_num/2)
                
        if (len(self.parent_selection_type) == 2):
            # Probability assignment to each chromosome 
            if (self.parent_selection_type[0] =='FPS'):
                # due to minus negative values in fitness we biased all 
                # values with minimum fitness
                gen_value = self.gen_fitness - np.min(self.gen_fitness) + 0.001
                # transfer generation values to normalized probability
                self.gen_probability = gen_value / np.sum(gen_value)
                                
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
                

    def crossover(self):
        recombination_prob_array = np.random.choice(np.arange(0, 2), size=self.pair_parents_num, p = [1-self.crossover_prob, self.crossover_prob])
        if (self.xover_type == 'single-point'):
            xover_point = np.random.randint(self.gene_num//self.k, size=(self.pair_parents_num, self.k))
            # reshape offspring (or parents to pair parents)
            #print(self.offspring.shape)
            self.offspring = np.reshape(self.offspring, (-1, 2, self.k, 3)) # (6, 2, 4, 3) 
            #print(self.offspring.shape)
            # relpace parent's gene with new gene
            for off in range(self.pair_parents_num):
                if (recombination_prob_array[off] == True):
                    for k in range(self.k):
                        temp0 = self.offspring[off, 0, k, xover_point[off,k]:].copy()
                        temp1 = self.offspring[off, 1, k, xover_point[off,k]:].copy()
                        self.offspring[off, 0, k, xover_point[off,k]:] = temp1
                        self.offspring[off, 1, k, xover_point[off,k]:] = temp0                            
            self.offspring = np.reshape(self.offspring, (-1,self.gene_num))
            #print(self.offspring.shape)

                
    def mutation(self):
        mutation_probs = np.random.choice(np.arange(0, 2) , size=self.offspring_num, p = [1-self.mutation_prob, self.mutation_prob])
        if (self.mutation_type =='uniform'):
#            random_sigma = np.random.uniform(0, 1, size=(self.offspring_num, self.gene_num))
            random_sigma = np.random.uniform(-1, 1, size=(self.offspring_num, self.gene_num))/2
            
            for off in range(self.offspring_num):
                if (mutation_probs[off] == True):
                    for gene in range(self.gene_num):
                        #self.offspring[off, gene] = random_sigma[off, gene]
                        value = self.offspring[off, gene] + random_sigma[off, gene]
                        if (value > 1): value = 1
                        if (value < 0): value = 0
                        self.offspring[off, gene] = value

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
#            print('self.gen_fitness', self.gen_fitness)            
            offspring_evaluation, off_out, offspring_cluster_samples_indexes = self.evaluate_function(self.offspring)
            #off_out = self.offspring
            offspring_fitness = offspring_evaluation.copy()
            
            
            mu_plus_lamda_generation = np.concatenate((self.generation, off_out))
            mu_plus_lamda_fitness = np.concatenate((self.gen_fitness, offspring_fitness))
            mu_plus_lamda_cluster_samples_indexes = np.concatenate((self.gen_cluster_samples_indexes,
                                                                    offspring_cluster_samples_indexes))
                        
#            print(mu_plus_lamda_generation.shape)
#            print(mu_plus_lamda_fitness.shape)
#            print(len(mu_plus_lamda_cluster_samples_indexes))
            
            # sort maximum to minimum            
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness.argsort()[::-1]
            # select best mu individuals for next generation
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness_arg[0:self.population_size]


            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = mu_plus_lamda_generation[mu_plus_lamda_fitness_arg, :]
            self.gen_fitness = mu_plus_lamda_fitness[mu_plus_lamda_fitness_arg]
            self.gen_cluster_samples_indexes = mu_plus_lamda_cluster_samples_indexes[mu_plus_lamda_fitness_arg]
#            print('self.gen_fitness', self.gen_fitness)
            
            self.record()
    
        elif (self.survivor_selection_type[0] == 'mu_lambda'):
            # merge current generation with offsprings
            offspring_evaluation = self.evaluate_function(self.offspring)
#            offspring_fitness = - offspring_evaluation
#            # sort maximum to minimum            
#            offspring_fitness_arg = offspring_fitness.argsort()[::-1]
#            # select best mu individuals for next generation
#            offspring_fitness_arg = offspring_fitness_arg[0:self.population_size]
#            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
#            self.generation = self.offspring[offspring_fitness_arg, :]            

        elif (self.survivor_selection_type[0] == 'round_robin'):
            # merge current generation with offsprings
            round_robin = np.concatenate((self.generation, self.offspring))
            offspring_evaluation,_ = self.evaluate_function(self.offspring)
            offspring_fitness = offspring_evaluation.copy()
            round_robin_fitness = np.concatenate((self.gen_fitness, offspring_fitness))
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
#        else:
#            # check variance of 5 latest fitness array 
#            if (len(self.gen_best_fitness_array) > 5):
#                gen_best_fitness_array_variance = np.var(self.gen_best_fitness_array[-5:])
#                print("gen_best_fitness_array_variance", gen_best_fitness_array_variance)
#                if (gen_best_fitness_array_variance < 0.2):
#                    self.termination = True


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
#            print('-------------------------------------------- generation = ', self.gen)
            
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
        
        self.gen_fitness_array = np.asarray(self.gen_fitness_array)
    



 



    


