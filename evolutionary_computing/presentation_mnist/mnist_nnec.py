#*****************************************************************************/
# @file    mnist_nnec.py
# @author  Majid Nasiri
# @version V2.0.0
# @date    29 May 2017
# @brief    MNIST neural network evolutionary computing
#           Whe have optimized number of neurons in three hidden layer's of a 
#           Multi-Layer Percepton Network Using Genetic Algorithm.
#           In this task we implemented MLP network using Tensorflow farmework.
#*****************************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import mnist_perceptron_net
import time


"""
Representation --------------- integer
Recombination ---------------- 1-point crossover
Recombination Probability ---- 90%
Mutation --------------------- Adding a small (positive or negative) value to each gene
Mutation Probability --------- 50%
Parent Selection ------------- 5 best and 3 random worst
Survivor Selection ----------- (mu + lambda)
Population Size -------------- 20
Number of Offspring ---------- 20
Initialization --------------- Random
Termination Condition -------- Affer 20 generation

"""

     

class evolutionary_computing():
    def __init__(self, 
                 rep='integer', 
                 crossover='uniform', 
                 mutation='add_positive_negative', 
                 ps = ('best_n_of_k',     5, 20),
                 ss = ('mu_plus_lambda',    0.4),
                 result_dir = './results'
                 ):
        
        self.representation_type = rep
        self.max_generation = 12
        self.gen = 0
        self.population_size = 4*5
        self.gene_num = 3

        self.parent_selection_type = ps
        self.offspring_num = 12
        self.pair_parents_num = 6

        
        self.xover_type = crossover
        self.crossover_prob = 0.9        
        
        
        self.mutation_type = mutation
        self.mutation_prob = 0.5
        self.mutation_prob = 1/self.gene_num        
        
        
        self.survivor_selection_type = ss
                
        self.termination = 0
        self.termination_count = 0
        self.termination_max_unchage = 25
        
        self.best_fitness_array = []
        self.best_chromosome_array = []
        
        self.rdir = result_dir
        if not os.path.exists(self.rdir):
            os.makedirs(self.rdir)
    
        print('    representation type : ', rep)
        print('  parent selection type : ', ps)
        print('         crossover type : ', crossover)
        print('          mutation type : ', mutation)
        print('survivor selection type : ', ss[0])
    
    def evaluate_function(self, chromosomes):
        gen_evaluation = np.zeros((chromosomes.shape[0],))        
        for chro in range(chromosomes.shape[0]):
            print('chromosomes {}/{} of generation {} is evaluationg ...'.format(chro, chromosomes.shape[0], self.gen))
            model = mnist_perceptron_net.perceptron(self.generation[chro], n_classes=10)
            model.train_network()
            gen_evaluation[chro] = model.accuracy
        print(gen_evaluation)
        return gen_evaluation

    def fitness(self, chromosomes):
        """
        EVALUATE each candidate
        """        
        
        gen_evaluation = self.evaluate_function(chromosomes)
        chromosomes_fitness = gen_evaluation.copy()
        
        return chromosomes_fitness
    
        
    def initialize(self):
        """
        INITIALISE population with random candidate solutions
        """
        
        self.generation = np.random.randint(30, 80, size=self.gene_num)
        for _ in range(self.population_size-1):
            self.generation = np.vstack([self.generation, np.random.randint(30, 80, size=self.gene_num)])
                
        #print(self.generation)
        
    def parent_selection(self):
        """
        we kept the top 25% (5 nets), randomly kept 3 more loser networks, and 
        mutated a few of them.
        """
        # sort best to worst
        fitness_sorted_b2w = self.gen_fitness.argsort()[::-1]
        best_five_pop = fitness_sorted_b2w[0:5]
        worset_three_random_pop = np.random.choice(fitness_sorted_b2w[5:], size=(3,))
        
        self.mating_pool = np.concatenate((best_five_pop, worset_three_random_pop))
        
        # pick k individuals randomly without replacement        
        self.parents = np.random.choice(self.mating_pool, size=(self.offspring_num,))
        #print(self.parents)
        self.pair_parents = np.reshape(self.parents, (-1, 2))
        self.offspring = self.generation[self.parents]
        #print(self.pair_parents)
        
    def crossover(self):
            """
            Breeding children of our fittest networks is where a lot of the magic
            happens. Breeding will be different for every application of genetic 
            algorithms, and in our case, we’ve decided to randomly choose parameters
            for the child from the mother and father.
            """
            recombination_prob_array = np.random.choice(np.arange(0, 2) , size=self.pair_parents_num, p = [1-self.crossover_prob, self.crossover_prob])
            #print(recombination_prob_array)
            xover_point = np.random.randint(1, self.gene_num, size=self.pair_parents_num)
            #print(xover_point)               
            # reshape offspring (or parents to pair parents)
            self.offspring = np.reshape(self.offspring, (-1, 2, self.gene_num))
            # relpace parent's gene with new gene
            for off in range(self.pair_parents_num):
                if (recombination_prob_array[off] == True):
                    pl_genes = self.offspring[off, 0, 0:xover_point[off]].copy()
                    self.offspring[off, 0, 0:xover_point[off]] = self.offspring[off, 1, 0:xover_point[off]]
                    self.offspring[off, 1, 0:xover_point[off]] = pl_genes
                    
            self.offspring = np.reshape(self.offspring, (-1,self.gene_num))

    def mutation(self):
            """
            Mutation is also really important as it helps us find chance greatness. 
            In our case, we randomly choose a parameter and then randomly choose a 
            new parameter for it. It can actually end up mutating to the same thing,
            but that’s all luck of the draw.
            """
            mutation_probs = np.random.choice(np.arange(0, 2) , size=self.offspring_num, p = [1-self.mutation_prob, self.mutation_prob])
            mutation_points = np.random.randint(self.gene_num, size=self.offspring_num)
            new_genes = np.random.randint(30, 80, size=self.offspring_num)
                
            for off in range(self.offspring_num):
                if (mutation_probs[off] == True):
                    self.offspring[off, mutation_points[off]] = new_genes[off]
               
                    
    def survivor_selection(self):
        
        self.generation = np.concatenate((self.generation[self.mating_pool], self.offspring))
        self.gen_fitness = np.concatenate((self.gen_fitness[self.mating_pool], self.offspring_fitness))

        # save best fitness
        self.best_fitness_arg = self.gen_fitness.argmax()
        self.best_fitness = self.gen_fitness.max()
        self.best_chromosome = self.generation[self.best_fitness_arg]
        self.best_chromosome_array.append(self.best_chromosome)
        self.best_fitness_array.append(self.best_fitness)

    def run(self, display='off'):
        """
        largest quality smaller than self.cost_max is desire solution
        """
        
        #INITIALISE population with random candidate solutions
        self.initialize()
        
        #EVALUATE each candidate
        self.gen_fitness = self.fitness(self.generation)

        #REPEAT UNTIL ( TERMINATION CONDITION is satisfied 
        while(self.termination == 0):
            print('----------------------- : generation = ', self.gen)
            
            #SELECT parents
            self.parent_selection()
            
            #RECOMBINE pairs of parents
            self.crossover()
            
            #MUTATE the resulting offspring 
            self.mutation()
            
            #EVALUATE new candidates
            self.offspring_fitness = self.fitness(self.offspring)
            
            # SELECT individuals for the next generation
            self.survivor_selection()          
               
            self.gen +=1
            if (self.gen == self.max_generation):
                self.termination = 1
            




st = time.time()
model = evolutionary_computing()
model.run()
print(model.best_fitness_array)
print(model.best_chromosome_array)    
et = time.time()
print ('\n%2.2f sec' %  (et-st))
