import numpy as np

# Defining the problem
class QAProblem:
    def __init__(self, n, distances, flows):
        self.n = n
        self.distances = distances
        self.flows = flows

    def trad_eval(self, solution):
        cost = 0.
        for i in range(self.n):
            for j in range(self.n):
                dist = self.distances[i][j]
                flow = self.flows[solution[i]][solution[j]]
                cost += flow * dist
        return cost

    def symm_eval(self, solution):
        cost = 0
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                dist = self.distances[i][j]
                flow = self.flows[solution[i]][solution[j]]
                cost += 2*(flow * dist)
        return cost

    def __call__(self, solution):
        perm_flow = self.flows[solution,:][:,solution]
        return int(np.multiply(self.distances, perm_flow).sum())

    def create_from_file(path):
        with open(path, "r") as f:
            n = int(f.readline().strip())
            distances, flows = np.zeros((n, n), dtype=int), np.zeros((n, n), dtype=int)
            _ = f.readline()
            for i in range(n):
                flows[i,:] = (list(map(int, f.readline().split())))
            for j in range(n):
                distances[j,:] = (list(map(int, f.readline().split())))
        return QAProblem(n, distances, flows)

# Reading data

qaProblem = QAProblem.create_from_file(path="tai256c.dat")

def cross_order(parent1, parent2, low_rate=0.2, upp_rate=0.8):
    l = len(parent1)
    par_size = int(np.ceil( l * np.random.uniform(low=low_rate, high=upp_rate, size=1)))
    
    start_point = np.random.randint(l)
    cop_points = (start_point + np.array(range(par_size))) % l
    end_point = (start_point + par_size) % l
    rep_points_all = (end_point + np.array(range(l))) % l
    rep_points = rep_points_all[0:(l - par_size)]

    off1 = np.zeros(l, dtype=int)
    off2 = np.zeros(l, dtype=int)

    off1[cop_points] = parent1[cop_points]
    off2[cop_points] = parent2[cop_points]

    parent2_cont = np.setdiff1d(parent2[rep_points_all], off1[cop_points], assume_unique=True)
    parent1_cont = np.setdiff1d(parent1[rep_points_all], off2[cop_points], assume_unique=True)

    off1[rep_points] = parent2_cont[0:(l - par_size)]
    off2[rep_points] = parent1_cont[0:(l - par_size)]

    return off1, off2




def mut_swap(individual, strength=1, inplace=True):
    if inplace:
        mut_ind = individual
    else:
        mut_ind = individual.copy()

    l = len(individual)
    for i in range(strength):
        r = np.random.permutation(l)[[0,1]]
        s,e = np.min(r), np.max(r)
        temp = mut_ind[s]
        mut_ind[s] = mut_ind[e]
        mut_ind[e] = temp
    
    return mut_ind



def selection_tournament(pop_fitness, perc_tourn=0.2):
    perc_tourn = max(0.1, perc_tourn)
    perc_tourn = min(1., perc_tourn)
    
    p_size = len(pop_fitness)

    tourn_size = int(np.ceil(perc_tourn * p_size))

    indx1 = np.random.permutation(p_size)[range(tourn_size)]
    best1indx = np.argmin(pop_fitness[indx1])
    true_best1_indx = indx1[best1indx]

    true_best2_indx = true_best1_indx

    while(true_best1_indx ==  true_best2_indx):
        indx2 = np.random.permutation(p_size)[range(tourn_size)]
        best2indx = np.argmin(pop_fitness[indx2])
        true_best2_indx = indx2[best2indx]

    return true_best1_indx, true_best2_indx


def replacement_best_n(pop_fitness, sel_size:int):
    return np.argsort(pop_fitness)[range(sel_size)]


class GAforQA:
    def __init__(self, qa_problem:QAProblem, pop_size:int=50, crossover_rate:float=0.9, mutation_rate:float=0.1) -> None:
        self.qa_problem = qa_problem
        self.pop_size = pop_size
        self.cross_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = np.empty((self.pop_size, self.qa_problem.n),  dtype=int)
        self.fitness = np.zeros(self.pop_size, dtype=int)

    def initialize(self):
        for i in range(self.pop_size):
            x = np.random.permutation(self.qa_problem.n)
            f = self.qa_problem(x)
            self.population[i,:] = x
            self.fitness[i] = f
            

    def run(self, max_runs, max_gens):

        stats_fitness = np.zeros((max_gens, max_runs), dtype=int)

        for r in range(max_runs):

            np.random.seed(r)

            self.initialize()

            best_fit = np.min(self.fitness)

            stats_fitness[0, r] = best_fit

            print("{:d}\t{:d}\t{:d}\t{:d}".format(r+1, 1, best_fit, abs(best_fit-44759294)))

            stag_times = 0

            for gen in range(1, max_gens):

                # Selecting pairs of parents

                offspring = []
                off_fitness = []

                for i in range(self.pop_size//2):
                    par1_idx, par2_idx = selection_tournament(self.fitness, perc_tourn=0.2)
                    par1 = self.population[par1_idx]
                    par2 = self.population[par2_idx]
                    if(np.random.random() < self.cross_rate):
                        off1, off2 = cross_order(par1, par2)
                    else:
                        off1, off2 = par1.copy(), par2.copy()

                    off1 = mut_swap(off1, np.random.randint(low=0, high=2))
                    off2 = mut_swap(off2, np.random.randint(low=0, high=2))

                    fit1 = self.qa_problem(off1)
                    fit2 = self.qa_problem(off2)

                    offspring.append(off1)
                    offspring.append(off2)
                    off_fitness.append(fit1)
                    off_fitness.append(fit2)

                    

                all_fits = np.append(self.fitness, off_fitness)
                all_pop = np.append(self.population, offspring, axis=0)
                
                rep_indxs = replacement_best_n(all_fits, self.pop_size)

                self.population = all_pop[rep_indxs]
                self.fitness = all_fits[rep_indxs]

                new_best_fit = self.fitness[0]

                if new_best_fit == best_fit:
                    stag_times += 1
                else:
                    stag_times = 0
                    best_fit = self.fitness[0]

                if stag_times > 10:
                    repindx = np.random.randint(low=self.pop_size-3, high=self.pop_size)
                    self.population[repindx] = np.random.permutation(self.qa_problem.n)
                    self.fitness[repindx] = self.qa_problem(self.population[repindx])
                    stag_times = 0

                stats_fitness[gen, r] = best_fit

                print("{:d}\t{:d}\t{:d}\t{:d}".format(r+1, gen+1, best_fit, abs(best_fit-44759294)))
        
        return stats_fitness
            

ga_alg = GAforQA(qaProblem, pop_size=800)

results = ga_alg.run(max_runs=10, max_gens=20000)



