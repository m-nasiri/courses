#*****************************************************************************/
# @file    genetic_algorithm.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    29 May 2017
# @brief   
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt


def  sigmoid(x): return 1/(1 + np.exp(-x))      # activation function

def ff_layer(WW, OO):
    return sigmoid(np.dot(WW, OO))

for i in range(50): print(' ') 

X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]]).T

#AND
#t=np.array([[0,0,0,1]])
#OR
#t=np.array([[0,1,1,1]])
#NAND
#t=np.array([[1,1,1,0]])
#NOR
#t=np.array([[1,0,0,0]])
#XNOR
#t=np.array([[1,0,0,1]])
#XOR
t=np.array([[0,1,1,0]])



N = np.array([2,2,1])
mu=0.1
max_generation=1000
chromosome_number = 40
desire_error = 0.0
criteria_satisfied = 0
random_ratio = 0.2
mutation_ratio = 0.1
crossover_ratio = 0.9

random_samples =  0.2 * chromosome_number 
random_samples = int(random_samples)

WW = []
WW.append((np.random.randn(N[0+1], N[0]+1)))
WW.append((np.random.randn(N[1+1], N[1]+1)))

OO = []
OO.append((np.ones((N[0]+1,1))))
OO.append((np.ones((N[1]+1,1))))

O = []
O.append((np.zeros((N[0+1],1))))
O.append((np.zeros((N[1+1],1))))

generation = np.random.randn(chromosome_number, 9)
generation_error_array = np.zeros(t.shape)
generation_error = np.zeros((1,chromosome_number))
generation_accuracy_array = []

generation_number = 0
while(criteria_satisfied==0 and  generation_number<max_generation):
    
    for pop in range(chromosome_number):
        WW[0] = np.reshape(generation[pop,0                                :((0)                              +(WW[0].shape[0] * WW[0].shape[1]))],WW[0].shape)
        WW[1] = np.reshape(generation[pop,(WW[0].shape[0] * WW[0].shape[1]):((WW[0].shape[0] * WW[0].shape[1])+(WW[1].shape[0] * WW[1].shape[1]))],WW[1].shape)
        
        for sample in range(X.shape[1]):
        
            OO[0][1:,0] = X[:, sample]
            O[0]=ff_layer(WW[0], OO[0])
            OO[1][1:,0]=O[0][:,0]
            O[1]=ff_layer(WW[1], OO[1])
            e =t[0][sample] - O[1][0][0]
              
            generation_error_array[:, sample] = e    
    
        total_error = np.sum(generation_error_array**2, axis = 0)/2
        average_error = np.mean(total_error)
        generation_error[0, pop] = average_error
        minimum_error = np.min(generation_error) 
    
        if (minimum_error<desire_error):
            criteria_satisfied = 1
        else:
            criteria_satisfied = 0
         
    generation_accuracy = 1-2*generation_error
    generation_accuracy_array.append(generation_accuracy)
    
    best = np.argmin(generation_error)
    
    generation [0, :] = generation[best,:]
    probability = 1/generation_error
    probability = probability / np.sum(probability)
    
    # bests
    best_flags = (generation_error == generation_error[0, np.argmin(generation_error)])
    best_indexs = np.where(best_flags)[1]
    best_samples = np.where(best_flags)[1].shape[0]
    pop_best = generation[best_indexs, :]
    
    # random generation
    pop_random = np.random.randn(random_samples, 9)
    
    roulette_number = chromosome_number - (pop_best.shape[0] + random_samples)
    # roulette selection
    roulette_indexs = np.random.choice(np.arange(0, chromosome_number) , size=(roulette_number,), p = probability[0])
    pop_roulette = generation[roulette_indexs, :]
    

    # crossover
    crossover_samples = int(crossover_ratio * roulette_number)
    crossover_samples = crossover_samples + 1 if crossover_samples % 2 == 1 else crossover_samples
    
    pop_crossover = np.zeros((crossover_samples,9))
    crossover_chromosome_indexs = np.random.randint(roulette_number, size = (int(crossover_samples/2),2))
    
    for chromosome in range(int(crossover_samples/2)):
        for gene in range(9):
            crossover_new_genes = np.random.uniform(low = pop_roulette[crossover_chromosome_indexs[chromosome,0],gene]  ,
                                                    high =pop_roulette[crossover_chromosome_indexs[chromosome,1],gene] , size = 2)
            pop_crossover[chromosome*2  ,gene] = crossover_new_genes[0]
            pop_crossover[chromosome*2+1,gene] = crossover_new_genes[1]
        
    
    # mutation
    mutation_samples = roulette_number - crossover_samples
    mutation_gene_index = np.random.randint(9, size = mutation_samples)
    mutation_chromosome_indexs =  np.random.randint(roulette_number, size = mutation_samples)
    mutation_new_genes = np.random.randn(mutation_samples, 1)    
    pop_mutation = pop_roulette[mutation_chromosome_indexs,:]
    
    # gene muutation
    for i in range(mutation_samples):
        pop_mutation[i, mutation_gene_index[i]] = mutation_new_genes[i]

    # new generation
    generation[0                                           : best_samples                                                  ]= pop_best
    generation[best_samples                                : best_samples+random_samples                                   ]= pop_random
    generation[best_samples+random_samples                 : best_samples+random_samples+mutation_samples                  ]= pop_mutation
    generation[best_samples+random_samples+mutation_samples: best_samples+random_samples+mutation_samples+crossover_samples]= pop_crossover   
    print('Generation ={}   Accuracy={}'.format(generation_number, np.max(generation_accuracy)))    
    generation_number = generation_number + 1
    

plt.figure(num = 1, figsize=(8,6))
for gen in range(generation_number):
    plt.scatter(gen*np.ones((1,chromosome_number)), generation_accuracy_array[gen].T, s = 10 ,marker = '.')

plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
y1 = np.min(generation_accuracy_array)/1.2
y2 = np.max(generation_accuracy_array)
plt.axis((0,generation_number,y1,1))
plt.savefig('..\images\GA_XOR2.jpg')






