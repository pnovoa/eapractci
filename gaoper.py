import numpy as np
from qap import QAProblem
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def max_flow_min_dist(parent, qaprob:QAProblem, rep_perc=0.2):
    # newsolution = np.zeros(qaprob.n, dtype=int) - 1

    newsolution = parent.copy()
    newsolution[np.random.random(len(parent)) < rep_perc] = -1

    used_locs = np.argwhere(newsolution != -1).flatten()
    
    start_loc = used_locs[0]
    start_fac = newsolution[start_loc]
    
    for i in range(qaprob.n-len(used_locs)):

        # Decreasing sort by flow
        flows = qaprob.flows[:, start_fac]
        fac_max_flow = np.argsort(flows)

        avail_fac = np.setdiff1d(fac_max_flow, newsolution, assume_unique=True)

        dists = qaprob.distances[:, start_loc]
        loc_min_dist = np.flip(np.argsort(dists))
        used_loc = np.argwhere(newsolution != -1).flatten()
        avail_loc = np.setdiff1d(loc_min_dist, used_loc, assume_unique=True)

        start_fac = avail_fac[0]
        start_loc = avail_loc[0]

        newsolution[start_loc] = start_fac

    return newsolution



def swap_if_better(parent, parfit, qaprob:QAProblem):
    # TODO: Improve
    rpos = np.random.permutation(qaprob.n)[0:2]
    rfac = parent[rpos]
    
    newsol = parent.copy()

    temp = newsol[rpos[0]]
    newsol[rpos[0]] = newsol[rpos[1]]
    newsol[rpos[1]] = newsol[temp]

    



    


    

    



