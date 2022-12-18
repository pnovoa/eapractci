import numpy as np
from tqdm import tqdm

from qap import QAProblem
import gaoper as op


class TraditionalGA:
    def __init__(self, qa_problem:QAProblem, pop_size:int=50, crossover_rate:float=0.8, mutation_rate:float=0.05, replace_fn=op.replace_fitness_based) -> None:
        self.qa_problem = qa_problem
        self.pop_size = pop_size
        self.cross_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = np.empty((self.pop_size, self.qa_problem.n),  dtype=int)
        self.fitness = np.zeros(self.pop_size, dtype=int)
        self.replace = replace_fn

    def initialize(self):
        for i in range(self.pop_size):
            x = np.random.permutation(self.qa_problem.n)
            f = self.qa_problem(x)
            self.population[i,:] = x
            self.fitness[i] = f

    def select(self):
        # Parent selection for crossover
        par1_idx, par2_idx = op.selection_tournament(self.fitness, perc_tourn=0.3)
        par1 = self.population[par1_idx]
        par2 = self.population[par2_idx]
        return par1, par2
    
    def crossover(self, par1, par2):
        # Crossover
        if(np.random.random() < self.cross_rate):
            off1, off2 = op.cross_order(par1, par2)
        else:
            off1, off2 = par1.copy(), par2.copy()
        return off1, off2


    def mutate(self, off1, off2):
        # Mutation
        off1 = op.mut_swap(off1, self.mutation_rate)
        off2 = op.mut_swap(off2, self.mutation_rate)
        return off1, off2

    def evaluate(self, individual):
        return self.qa_problem(individual)



    def evolve(self):
        
        offspring = []
        off_fitness = []

        for i in range(self.pop_size//2):
            
            par1, par2 = self.select()
            off1, off2 = self.crossover(par1, par2)
            off1, off2 = self.mutate(off1, off2)

            # Offspring fitness evaluation
            fit1 = self.evaluate(off1)
            fit2 = self.evaluate(off2)

            # Adding the new individuals to the ofsspring pool
            offspring.append(off1)
            offspring.append(off2)
            off_fitness.append(fit1)
            off_fitness.append(fit2)

        return offspring, off_fitness
    
    def __call__(self, max_gens):
        
        self.initialize()
        # best_fit = np.min(self.fitness)
        # stag_times = 0
        
        for i in tqdm(range(1, max_gens)): 
        # for gen in range(1, max_gens):
            
            offspring, off_fitness = self.evolve()

            # Create an extender pool of individuals (population + offspring)
            all_fits = np.append(self.fitness, off_fitness)
            all_pop = np.append(self.population, offspring, axis=0)
            
            # Replacement: select the top p_size fittest individuals            
            rep_indxs = self.replace(all_fits, self.pop_size)
            
            self.population = all_pop[rep_indxs]
            self.fitness = all_fits[rep_indxs]

            self.population = all_pop[rep_indxs]
            self.fitness = all_fits[rep_indxs]
            

            


class BaldwinianGA(TraditionalGA):
    
    def __init__(self, qa_problem: QAProblem, pop_size: int = 50, crossover_rate: float = 0.8, mutation_rate: float = 0.05, replace_fn=op.replace_fitness_based, ls_deep=10) -> None:
        super().__init__(qa_problem, pop_size, crossover_rate, mutation_rate, replace_fn)


    