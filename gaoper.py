import numpy as np
from qap import QAProblem

def cross_order(parent1, parent2, low_rate=0.3, upp_rate=0.7):
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

def mut_swap(individual, mut_rate, inplace=True):
    
    if inplace:
        mut_ind = individual
    else:
        mut_ind = individual.copy()

    l = len(individual)

    mut_size = int(np.ceil(np.random.uniform(0,mut_rate)*l))

    rand_pos = np.random.permutation(l)[0:mut_size]

    swap_pos = np.random.permutation(rand_pos)

    mut_ind[rand_pos] = mut_ind[swap_pos]
    
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


def replace_fitness_based(pop_fitness, sel_size:int):
    return np.argsort(pop_fitness)[range(sel_size)]


def replacement_age_based(pop_fitness, sel_size:int):
    return range(sel_size, len(pop_fitness))


def max_flow_min_dist(loc_i:int, fac_i:int, qaprob:QAProblem):
    newsolution = np.zeros(qaprob.n, dtype=int) - 1
    
    start_fac = fac_i
    start_loc = loc_i
    
    newsolution[loc_i] = fac_i
    for i in range(1, qaprob.n):

        # Decreasing sort by flow
        flows = qaprob.flows[:, start_fac]
        fac_max_flow = np.flip(np.argsort(flows))

        avail_fac = np.setdiff1d(fac_max_flow, newsolution, assume_unique=True)

        dists = qaprob.distances[:, start_loc]
        loc_min_dist = np.argsort(dists)
        used_loc = np.argwhere(newsolution != -1).flatten()
        avail_loc = np.setdiff1d(loc_min_dist, used_loc, assume_unique=True)

        start_fac = avail_fac[0]
        start_loc = avail_loc[0]

        newsolution[start_loc] = start_fac

    return newsolution

    


if __name__ == '__main__':
    qaprob = QAProblem.build_from_file("tai256c.dat")

    np.random.seed(331)
    fits = []
    orig_fit = []
    for j in range(4):
        orig_sol = np.random.permutation(qaprob.n)
        orig_fit.append(qaprob(orig_sol))
        for i in range(256):
            loc_i = i
            fac_i = orig_sol[loc_i]
            newsol = max_flow_min_dist(loc_i, fac_i, qaprob)
            fits.append(qaprob(newsol))
            print(f"{j}\t{i}\t{fits[-1]}")

    print(f"Best:{sorted(fits)[0]}")

    print(orig_fit)
    

    



