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


def evaluate_function(xi):
    X = np.power(xi, 4) - 16 * np.power(xi, 2) + 5 * xi
    return np.sum(X, axis=1)/2

def map_index(n,i,j):
    if (n == 2):
        mat = np.array([[0,0],[0,0]])
        return mat[i,j]
    elif (n == 3):
        mat = np.array([[0,0,1],[0,0,2],[1,2,0]])    
        return mat[i,j]
    elif (n == 4):
        mat = np.array([[0,0,1,2],[0,0,3,4],[1,3,0,5],[2,4,5,0]])    
        return mat[i,j]
        


class evolutionary_computing():
    def __init__(self, 
                 rep='float',
                 max_generation = 20,
                 population_size = 20,
                 crossover='single',
                 crossover_prob = 0.9,
                 mutation='uniform',
                 mutation_prob = 0.5,
                 ps = ('FPS','roulette_wheel'),
                 ss = ('generational', None),
                 result_dir = './results',
                 verbose = True,
                 ):
        
        self.representation_type = rep
        self.max_generation = max_generation
        self.population_size = population_size
        self.gene_num = 5

        self.parent_selection_type = ps
        
        
        self.xover_type = crossover
        self.crossover_prob = crossover_prob      
        
        
        self.mutation_type = mutation
        self.mutation_prob = mutation_prob
        self.mutation_prob = 1/self.gene_num        
        
        
        self.survivor_selection_type = ss
                
        self.termination = 0
        self.termination_count = 0
        self.termination_max_unchage = 25
        
        self.best_fitness_array = []
        
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

        self.gen_evaluation = evaluate_function(self.generation)
        self.gen_fitness = -self.gen_evaluation

        # save best fitness
        self.best_fitness_arg = self.gen_fitness.argmax()
        self.best_fitness = self.gen_fitness.max()
        self.best_chromosome = self.generation[self.best_fitness_arg]
        self.best_fitness_array.append(self.best_fitness)
    
    def initialize(self):
        """
        INITIALISE population with random candidate solutions
        """
        
        if (self.representation_type == 'float'):
            self.generation = np.random.uniform(-5, 5, size=self.gene_num)
            for _ in range(self.population_size-1):
                self.generation = np.vstack([self.generation, np.random.uniform(-5, 5, size=self.gene_num)])
    
        if (self.mutation_type =='uncorr_one_sigma'):
            self.tao = 1 / np.sqrt(self.gene_num)
            self.eps = 0.1
            self.generation_sigma = np.random.uniform(-5, 5, size=1)
            for _ in range(self.population_size-1):
                self.generation_sigma = np.vstack([self.generation_sigma, np.random.uniform(-5, 5, size=1)])
    
        if (self.mutation_type =='uncorr_n_sigma'):
            self.tao = 1 / np.sqrt(2 * np.sqrt(self.gene_num))
            self.taoh = 1 / np.sqrt(2 * self.gene_num)
            self.eps = 0.1
            self.generation_sigma = np.random.uniform(-5, 5, size=self.gene_num)
            for _ in range(self.population_size-1):
                self.generation_sigma = np.vstack([self.generation_sigma, np.random.uniform(-5, 5, size=self.gene_num)])

        if (self.mutation_type =='corr'):
            self.tao = 1 / np.sqrt(2 * np.sqrt(self.gene_num))
            self.taoh = 1 / np.sqrt(2 * self.gene_num)
            self.eps = 0.1
            self.generation_sigma = np.random.uniform(-5, 5, size=self.gene_num)
            for _ in range(self.population_size-1):
                self.generation_sigma = np.vstack([self.generation_sigma, np.random.uniform(-5, 5, size=self.gene_num)])
                
            self.alpha_num = int((self.gene_num * (self.gene_num-1))/2)
            self.generation_alpha = np.random.uniform(-5, 5, size=self.alpha_num)
            for _ in range(self.population_size-1):
                self.generation_alpha = np.vstack([self.generation_alpha, np.random.uniform(-5, 5, size=self.alpha_num)])


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


            elif (self.xover_type =='whole'):
                self.offspring = np.reshape(self.offspring, (-1, 2, self.gene_num))
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(self.pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        for xpi in range(self.gene_num):
                            self.offspring[off, :, xpi] = (alpha * self.offspring[off, 0, xpi])+((1-alpha) * self.offspring[off, 1, xpi])
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))
            
            
            elif (self.xover_type =='blend'):
                self.offspring = np.reshape(self.offspring, (-1, 2, self.gene_num))
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(self.pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        di = np.abs(self.offspring[off,0,:] -  self.offspring[off,1,:])
                        zxi1 = self.offspring[off,0,:] - alpha * di
                        zxi2 = self.offspring[off,0,:] + alpha * di                        
                        zyi1 = self.offspring[off,1,:] - alpha * di
                        zyi2 = self.offspring[off,1,:] + alpha * di
                        for g in range(self.gene_num):
                            self.offspring[off,0,g] = np.random.uniform(zxi1[g], zxi2[g], size=1)
                            self.offspring[off,1,g] = np.random.uniform(zyi1[g], zyi2[g], size=1)
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))
                
    def mutation(self):
        
            mutation_probs = np.random.choice(np.arange(0, 2) , size=self.offspring_num, p = [1-self.mutation_prob, self.mutation_prob])
            if (self.mutation_type =='uniform'):
                mutation_points = np.random.randint(self.gene_num, size=self.offspring_num)   
                new_genes = np.random.uniform(-5, 5, size=self.offspring_num)
                
                for off in range(self.offspring_num):
                    if (mutation_probs[off] == True):
                        self.offspring[off, mutation_points[off]] = new_genes[off]
                        
            
            elif (self.mutation_type =='non_uniform'):
                for off in range(self.offspring_num):
                    N = np.random.normal()
                    self.offspring[off, :] = self.offspring[off, :] + N
            
            #elif (self.mutation_type =='self_adaptive'):            
                
            elif (self.mutation_type =='uncorr_one_sigma'):
                self.offspring_sigma = self.generation_sigma[self.pair_parents]
                self.offspring_sigma = np.reshape(self.offspring_sigma, (-1,1))
                N = np.random.normal(size=self.offspring_num)
                for off in range(self.offspring_num):
                    self.offspring_sigma[off] = self.offspring_sigma[off] * np.exp(self.tao * N[off])
                    if (self.offspring_sigma[off] < self.eps): self.offspring_sigma[off] = self.eps
                    
                for off in range(self.offspring_num):
                    for g in range(self.gene_num):
                        Ni = np.random.normal()
                        self.offspring[off, g] = self.offspring[off, g] + self.offspring_sigma[off] * Ni
            
            elif (self.mutation_type =='uncorr_n_sigma'):
                self.offspring_sigma = self.generation_sigma[self.pair_parents]
                self.offspring_sigma = np.reshape(self.offspring_sigma, (-1,self.gene_num))
                #print(self.offspring_sigma.shape)
                
                N = np.random.normal(size=self.offspring_num)
                Ni = np.random.normal(size=(self.offspring_num,self.gene_num))
                
                for off in range(self.offspring_num):
                    for g in range(self.gene_num):
                        self.offspring_sigma[off, g] = self.offspring_sigma[off, g] * np.exp(self.taoh * N[off] + self.tao * Ni[off, g])
                        if (self.offspring_sigma[off, g] < self.eps): self.offspring_sigma[off, g] = self.eps
                
                for off in range(self.offspring_num):
                    for g in range(self.gene_num):
                        Ni = np.random.normal()
                        self.offspring[off, g] = self.offspring[off, g] + self.offspring_sigma[off, g] * Ni


            elif (self.mutation_type =='corr'):
                self.offspring_sigma = self.generation_sigma[self.pair_parents]
                self.offspring_sigma = np.reshape(self.offspring_sigma, (-1,self.gene_num))
                self.offspring_alpha = self.generation_alpha[self.pair_parents]
                self.offspring_alpha = np.reshape(self.offspring_alpha, (-1,self.alpha_num))
                #print(self.offspring_sigma.shape)
                #print(self.offspring_alpha.shape)
                
                b = (np.pi/180)*5
                N = np.random.normal(size=self.offspring_num)
                Ni = np.random.normal(size=(self.offspring_num,self.gene_num))
                for off in range(self.offspring_num):
                    for g in range(self.gene_num):
                        self.offspring_sigma[off, g] = self.offspring_sigma[off, g] * np.exp(self.taoh * N[off] + self.tao * Ni[off, g])
                        if (self.offspring_sigma[off, g] < self.eps): self.offspring_sigma[off, g] = self.eps
                    
                N = np.random.normal(size=self.offspring_num)
                for off in range(self.offspring_num):
                    for a in range(self.alpha_num):
                        self.offspring_alpha[off, a] = self.offspring_alpha[off, a] + b * N[off]
                        if (np.abs(self.offspring_sigma[off, g]) > np.pi): self.offspring_sigma[off, g] -= 2*np.pi* np.sign(self.offspring_sigma[off, g])
                                
                for off in range(self.offspring_num):
                    cov_matrix = np.zeros((self.gene_num, self.gene_num))
                    #print(cov_matrix.shape)
                    for i in range(self.gene_num):
                        cov_matrix[i,i] = self.offspring_sigma[off, i]**2
                        
                    for i in range(self.gene_num):
                        for j in range(self.gene_num):
                            if (i != j):
                                cov_matrix[i,j] = (self.offspring_sigma[off, i]**2 - self.offspring_sigma[off, j]**2) * np.tan(2*self.offspring_alpha[off,map_index(self.gene_num,i,j)]) / 2

                    #print(cov_matrix)
                    N = np.random.multivariate_normal([0,0], cov_matrix, 1)
                    self.offspring[off, :] = self.offspring[off, :] +  N 
                    
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
            mu_plus_lamda_evaluation = evaluate_function(mu_plus_lamda)
            mu_plus_lamda_fitness = - mu_plus_lamda_evaluation
            # sort maximum to minimum            
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness.argsort()[::-1]
            # select best mu individuals for next generation
            mu_plus_lamda_fitness_arg = mu_plus_lamda_fitness_arg[0:self.population_size]
            # survivor_selection_type[1]% of old generation have been replaced with new offsprings 
            self.generation = mu_plus_lamda[mu_plus_lamda_fitness_arg, :]
    
        elif (self.survivor_selection_type[0] == 'mu_lambda'):
            # merge current generation with offsprings
            offspring_evaluation = evaluate_function(self.offspring)
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
            round_robin_evaluation = evaluate_function(round_robin)
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


                    
    def display(self, num):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(-5, 5, 0.05)
        Y = np.arange(-5, 5, 0.05)
        X, Y = np.meshgrid(X, Y)        
        R1 = np.power(X, 4) - 16 * np.power(X, 2) + 5 * X
        R2 = np.power(Y, 4) - 16 * np.power(Y, 2) + 5 * Y
        Z = 0.5 * (R1+R2)
        # Plot a basic wireframe.
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        #ax.set_zlim(-100, 100)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('best fitness = '+ str(self.best_fitness))
                
        for i,cor in enumerate(self.generation):
            ax.scatter(cor[0], cor[1], self.gen_evaluation[i], c='r')
        fig.savefig(self.rdir+ \
                    self.parent_selection_type[0]+\
                    '_#_'+\
                    self.survivor_selection_type[0]+\
                    '_#_'+str(num)+'.png',
                    dpi=fig.dpi)
        plt.close(fig)



    def run(self, display='off'):
        """
        largest quality smaller than self.cost_max is desire solution
        """
        
        #INITIALISE population with random candidate solutions
        self.initialize()
        
        #EVALUATE each candidate
        self.fitness()
        
        # show function and solutions 
        if (self.gene_num == 2 and display == 'on'):
            self.display(num=0)

        #REPEAT UNTIL ( TERMINATION CONDITION is satisfied )
        gen = 0
        while(self.termination == 0):
            
            #print('base_generation=', gen)

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
            
            # show function and solutions 
            if (self.gene_num == 2 and display == 'on'):
                self.display(gen+1)
    
            gen +=1
            if (gen == self.max_generation):
                self.termination = 1
    



 



    


