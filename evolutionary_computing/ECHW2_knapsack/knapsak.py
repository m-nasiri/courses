#*****************************************************************************/
# @file    knapsack.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    29 May 2017
# @brief   0-1 knapsack using genetic algorithm
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt


class Knapsack:
    def __init__(self):
        
        #self.max_generation = 1        #1000
        self.chromosome_num = 500        #500
        self.gene_num = 16
        self.termination = 0
        self.termination_count = 0
        self.termination_max_unchage = 25
        self.random_ratio = 0
        self.mutation_ratio = 0
        self.crossover_ratio = 0
        self.cost_max = 100
        self.value = np.random.choice(np.arange(0, 1) , size=self.gene_num)
        self.cost = np.random.choice(np.arange(1, 12) , size=self.gene_num)
        self.solution_chromosome = np.zeros((self.gene_num,))
        self.solution_cost = 0
        self.recombination_prob = 0.7
        self.mutation_prob = 1/self.gene_num
        self.solution_cost_array = []
        
                   
    
    def evaluation(self):
        """
        EVALUATE each candidate
        """
        
        self.gen_quality = np.sum(np.multiply(self.generation, self.value), axis=1)
        self.gen_cost = np.sum(np.multiply(self.generation, self.cost), axis=1)
        self.gen_valid_cost = self.gen_cost.copy()
        self.gen_valid_cost[self.gen_valid_cost > self.cost_max] = 0
        self.gen_probability = 1 / (self.gen_cost + 0.0001)
        self.gen_probability = self.gen_probability / np.sum(self.gen_probability)
        
        # replace new better genotype with old best genotypes 
        current_solution_cost = np.max(self.gen_valid_cost)
        if (current_solution_cost > self.solution_cost):
            self.solution_cost = current_solution_cost    
            self.solution_chromosome = self.generation[np.argmax(self.gen_valid_cost)]
            self.termination_count = 0
        else:
            self.termination_count += 1
        
        # save best solution
        self.solution_cost_array.append(self.solution_cost/self.cost_max)
        
        # termination because of no more improvement in last 25 iteration
        if (self.termination_count > self.termination_max_unchage):
            self.termination = 1
            
        # termination condition knapsack is full with best choices
        if (self.solution_cost == self.cost_max):
            self.termination = 1
        
        
    
    def initialize(self):
        """
        INITIALISE population with random candidate solutions
        """
        self.generation = np.random.choice([0,1], size=self.gene_num)
        for _ in range(self.chromosome_num-1):
            self.generation = np.vstack([self.generation, np.random.choice([0,1], size=self.gene_num)])
    
        
    def run(self):
        """
        largest quality smaller than self.cost_max is desire solution
        """
        
        #INITIALISE population with random candidate solutions
        self.initialize()
        
        #EVALUATE each candidate
        self.evaluation()
        
        best_parent = np.zeros((2,), dtype=np.int)
        offspring = np.zeros((2,self.gene_num), dtype=np.int)
        self.candidates = np.zeros((self.chromosome_num, self.gene_num), dtype=np.int)
        gen = 0
        #REPEAT UNTIL ( TERMINATION CONDITION is satisfied )
        while(self.termination == 0):
            gen +=1
            #SELECT parents
            for candid in range(int(self.chromosome_num/2)):
                # Best out of random 2
                random_parents = np.random.choice(np.arange(0, self.chromosome_num) , size=2, p = self.gen_probability)
                best_parent[0] = random_parents[np.argmax([self.gen_cost[random_parents[0]], self.gen_cost[random_parents[1]]])]
                
                random_parents = np.random.choice(np.arange(0, self.chromosome_num) , size=2, p = self.gen_probability)
                best_parent[1] = random_parents[np.argmax([self.gen_cost[random_parents[0]], self.gen_cost[random_parents[1]]])]
                                
                recombination_flag = np.random.choice(np.arange(0, 2) , size=1, p = [1-self.recombination_prob, self.recombination_prob])
                
                if (recombination_flag == True):
                    #RECOMBINE pairs of parents
                    crossover_point = int(np.random.randint(self.gene_num, size=1))

                    offspring[0,:] = np.concatenate((self.generation[best_parent[0]][0:crossover_point], 
                                                 self.generation[best_parent[1]][crossover_point: ]))
                    
                    offspring[1,:] = np.concatenate((self.generation[best_parent[1]][0:crossover_point], 
                                                 self.generation[best_parent[0]][crossover_point: ]))
                else:
                    offspring[0,:] = self.generation[best_parent[0]]
                    offspring[1,:] = self.generation[best_parent[1]]
    
                #MUTATE the resulting offspring  
                mutation_flag = np.random.choice(np.arange(0, 2) , size=(2, self.gene_num), p = [1-self.mutation_prob, self.mutation_prob])

                self.candidates[2*candid     ,:] = np.bitwise_xor(mutation_flag[0], offspring[0])
                self.candidates[2*candid + 1 ,:] = np.bitwise_xor(mutation_flag[1], offspring[1])
                
            
            #SELECT individuals for the next generation
            # generational scheme for survivor selection
            # all of the population in each iteration are discarded and replaced by their offspring.
            self.generation = self.candidates.copy()
            
            #EVALUATE new candidates            
            self.evaluation()
            
            print('Generation', gen,
                  'maximum cost', self.solution_cost,
                  'best genotype', self.solution_chromosome)
        
    





# create a knapsak genetic algorithm model
knapsack_model = Knapsack()

knapsack_model.run()

print(knapsack_model.cost_max)
print(knapsack_model.solution_chromosome)
print(knapsack_model.solution_cost)


# plot fullness percent of knapsack
plt.figure()
plt.plot(np.transpose(knapsack_model.solution_cost_array))
plt.xlabel('Generation')
plt.ylabel('Fullness percent of knapsack')
plt.grid()
plt.show()
 
    


