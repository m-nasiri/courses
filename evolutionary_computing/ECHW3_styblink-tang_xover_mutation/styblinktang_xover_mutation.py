#*****************************************************************************/
# @file    sty.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    29 May 2017
# @brief   
#*****************************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os


if (not os.path.exists('./results')):
    os.makedirs('./results')

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
    def __init__(self, rep='float', crossover='single', mutation='uniform'):
        
        
        self.representation_type = rep
        self.max_generation = 4
        self.population_size = 4*5
        self.gene_num = 2

        self.xover_type = crossover
        self.crossover_prob = 0.9        
        
        self.mutation_type = mutation
        self.mutation_prob = 0.5
        self.mutation_prob = 1/self.gene_num        
        
        
        self.termination = 0
        self.termination_count = 0
        self.termination_max_unchage = 25
        
        self.best_fittness_array = []
        
        print('representation type : ', rep)
        print('     crossover type : ', crossover)
        print('      mutation type : ', mutation)
                   
    def fittness(self):
        """
        EVALUATE each candidate
        """
        
        self.gen_evaluation = evaluate_function(self.generation)
        self.gen_fittness = -self.gen_evaluation
        gen_value = self.gen_fittness - np.min(self.gen_fittness)
        self.gen_probability = gen_value / np.sum(gen_value)
       
        # save best fittness
        self.best_fittness_arg = self.gen_fittness.argmax()
        self.best_fittness = self.gen_fittness.max()
        self.best_chromosome = self.generation[self.best_fittness_arg]
        self.best_fittness_array.append(self.best_fittness)
    
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
        ax.set_title('best fittness = '+ str(self.best_fittness))
        

        
        for i,cor in enumerate(self.generation):
            ax.scatter(cor[0], cor[1], self.gen_evaluation[i], c='r')
        
        fig.savefig('./results/'+self.xover_type+'_#_'+self.mutation_type+'_#_'+str(num)+'.png', dpi=fig.dpi)

    def run(self, display='off'):
        """
        largest quality smaller than self.cost_max is desire solution
        """
        
        #INITIALISE population with random candidate solutions
        self.initialize()
        
        
        #EVALUATE each candidate
        self.fittness()
        
        # show function and solutions 
        if (self.gene_num == 2 and display == 'on'):
            self.display(num=0)

        #REPEAT UNTIL ( TERMINATION CONDITION is satisfied )
        gen = 0
        while(self.termination == 0):
            
            #SELECT parents
            self.offspring_num = int(self.population_size/2);
            pair_parents_num = int(self.offspring_num/2)
            random_parents = np.random.choice(np.arange(0, self.population_size), size=(pair_parents_num,5), p = self.gen_probability)
            arg_sort = np.argsort(self.gen_fittness[random_parents])
            best_parent = np.column_stack((arg_sort[:,-1], arg_sort[:,-2]))
            self.offspring = self.generation[best_parent]
            
            
            ###################################################################    
            ###################################################################
            ###################################################################
            #RECOMBINE pairs of parents
            recombination_prob_array = np.random.choice(np.arange(0, 2) , size=pair_parents_num, p = [1-self.crossover_prob, self.crossover_prob])
            if (self.xover_type =='single'):
                xover_point = np.random.randint(self.gene_num, size=pair_parents_num)
                            
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        self.offspring[off, :, xover_point[off]] = (alpha * self.offspring[off, 0, xover_point[off]])+((1-alpha) * self.offspring[off, 1, xover_point[off]])
    
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))
                
                
            elif (self.xover_type =='simple'):
                xover_point = np.random.randint(self.gene_num, size=pair_parents_num)
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        for xpi in range(xover_point[off], self.gene_num):
                            self.offspring[off, :, xpi] = (alpha * self.offspring[off, 0, xpi])+((1-alpha) * self.offspring[off, 1, xpi])
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))


            elif (self.xover_type =='whole'):
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(pair_parents_num):
                    if (recombination_prob_array[off] == True):
                        for xpi in range(self.gene_num):
                            self.offspring[off, :, xpi] = (alpha * self.offspring[off, 0, xpi])+((1-alpha) * self.offspring[off, 1, xpi])
                self.offspring = np.reshape(self.offspring, (-1,self.gene_num))
            
            
            elif (self.xover_type =='blend'):
                #print(self.offspring)
                alpha = 0.5
                # relpace parent's gene with new gene
                for off in range(pair_parents_num):
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
            
            
            
            ###################################################################
            ###################################################################
            ###################################################################
            #MUTATE the resulting offspring 
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
                self.offspring_sigma = self.generation_sigma[best_parent]
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
                self.offspring_sigma = self.generation_sigma[best_parent]
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
                self.offspring_sigma = self.generation_sigma[best_parent]
                self.offspring_sigma = np.reshape(self.offspring_sigma, (-1,self.gene_num))
                self.offspring_alpha = self.generation_alpha[best_parent]
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


            
            
            # SELECT individuals for the next generation
            # best to worst
            gen_new_idxs = self.gen_probability.argsort()[::-1] 
            # save indices of best chromosomes
            gen_new_idxs = gen_new_idxs[0:self.offspring_num]   
            # 50% of old generation have been replaced with new offsprings 
            self.generation = np.concatenate((self.generation[gen_new_idxs], self.offspring))
            
            
            #EVALUATE new candidates            
            self.fittness()
            
            # show function and solutions 
            if (self.gene_num == 2 and display == 'on'):
                self.display(gen+1)
    

            gen +=1
            if (gen == self.max_generation):
                self.termination = 1
    






plt.close("all")
#xover_list = ['single']
#mutation_list = ['uniform']

xover_list = ['single', 'simple', 'whole', 'blend']
mutation_list = ['uniform', 'non_uniform', 'uncorr_one_sigma', 'uncorr_n_sigma', 'corr']


for xover_type in xover_list:
    for mutation_type in mutation_list:
        stylinski_model = evolutionary_computing(rep='float', crossover=xover_type, mutation=mutation_type)
        stylinski_model.run(display='on')

        fig = plt.figure()
        plt.plot(stylinski_model.best_fittness_array)
        plt.xlabel('Generation')
        plt.ylabel('Best fittness')
        print('    best chromosome :', stylinski_model.best_chromosome)
        print('      best fittness : ', stylinski_model.best_fittness)
        print('--------------------:-------------------')
        fig.savefig('./results/'+stylinski_model.xover_type+'_#_'+stylinski_model.mutation_type+'_#_result.png', dpi=fig.dpi)


 
    
#INITIALISE population with random candidate solutions
#EVALUATE each candidate
#REPEAT UNTIL ( TERMINATION CONDITION is satisfied )
    #SELECT parents
    #RECOMBINE pairs of parents
    #MUTATE the resulting offspring 
    #EVALUATE new candidates
    #SELECT individuals for the next generation
    

