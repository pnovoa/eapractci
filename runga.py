import argparse
import numpy as np
import pandas as pd
from qap import QAProblem
from gas import TraditionalGA, BaldwinianGA, LamarckianGA
import os

parser = argparse.ArgumentParser(description='Run a GA for solving a QA problem.')
parser.add_argument('algname', type=str, help='GA name')
parser.add_argument('psize', type=int, help='an integer for setting the population size of the GA')
parser.add_argument('crossr', type=float, help='a float value between 0 and 1 (exclusive) for setting the crossover rate')
parser.add_argument('mutr', type=float, help='a float value between 0 and 1 (exclusive) for setting the mutation rate')
parser.add_argument('fes', type=int, help='an integer indicating the maximum number of allowed calls to the fitness function')
parser.add_argument('run', type=int, help='an integer corresponding to the execution number')
args = parser.parse_args()

exec_identity = f"(A:{args.algname}, PS:{args.psize}, CR:{args.crossr}, MR:{args.mutr}, MF:{args.fes}, R:{args.run})"

print(f"Starting {exec_identity}")

# Controlling the random seed. Set it as equal to the execution number.
np.random.seed(args.run)

# Computing the number of generations for the algorithm as a function of MAX_FES and psize

MAX_GEN = args.fes // args.psize

# Creating the problem instance
qa_problem = QAProblem.build_from_file(path="tai256c.dat")


# Creating the algorithm
if args.algname == "TRAD":
    ga_alg = TraditionalGA(qa_problem=qa_problem, pop_size=args.psize, crossover_rate=args.crossr, mutation_rate=args.mutr)
elif args.algname == "BAL":
    ga_alg = BaldwinianGA(qa_problem=qa_problem, pop_size=args.psize, crossover_rate=args.crossr, mutation_rate=args.mutr)
    MAX_GEN = MAX_GEN // ga_alg.ls_depth
elif args.algname == "LAM":
    ga_alg = LamarckianGA(qa_problem=qa_problem, pop_size=args.psize)


# Executing the algorithm
ga_alg(max_gens=MAX_GEN)

# Retrieving statistics and saving them in the archives

file_name = f"{args.algname}_{args.psize}_{args.fes}_{args.crossr}_{args.mutr}_{args.run}"
file_name = os.path.join("gaout", file_name)

# Saving fitness evolution over time
l = len(qa_problem.best_fit_record)
df_fit = pd.DataFrame({
    "ALG":np.repeat(args.algname, l),
    "PSIZE":np.repeat(args.psize, l),
    "CR": np.repeat(args.crossr, l),
    "MR": np.repeat(args.mutr, l),
    "MAXFES":np.repeat(args.fes, l),
    "GENS": np.repeat(MAX_GEN, l),
    "RUN":np.repeat(args.run, l),
    "FES": list(range(1, l + 1)),
    "BFIT":qa_problem.best_fit_record
})

df_fit.to_csv(f"{file_name}_evol.csv", index=False)



# Saving the best solution

df_bs = pd.DataFrame(
    {
    "ALG":[args.algname],
    "PSIZE":[args.psize],
    "CR": [args.crossr],
    "MR": [args.mutr],
    "MAXFES":[args.fes],
    "GENS": [MAX_GEN],
    "RUN":[args.run],
    "BFIT":[int(qa_problem.best_fit)]
 }
)

bs_names = ["X" + str(i) for i in range(qa_problem.n)]

bs_values = [[v for v in qa_problem.best_sol]]

df_bs = pd.concat([df_bs, pd.DataFrame(bs_values, index=df_bs.index, columns=bs_names)], axis=1)

df_bs.to_csv(f"{file_name}_best.csv", index=False)

print(f"Finishing {exec_identity}. Best Fitness:{qa_problem.best_fit}")