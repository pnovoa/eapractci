import numpy as np
import matplotlib.pyplot as plt

class QAProblem:
    def __init__(self, n, distances, flows):
        self.n = n
        self.distances = distances
        self.flows = flows
        self.best_fit_record = []
        self.best_fit = np.Inf
        self.best_sol = []

    def __call__(self, solution):
        perm_flow = self.flows[solution,:][:,solution]
        fit = int(np.multiply(self.distances, perm_flow).sum())

        # Recording the best fit over each function call
        if fit < self.best_fit:
            self.best_fit = fit
            self.best_sol = solution.tolist()
        
        self.best_fit_record.append(self.best_fit)

        return fit

    def build_from_file(path):
        with open(path, "r") as f:
            n = int(f.readline().strip())
            distances, flows = np.zeros((n, n), dtype=int), np.zeros((n, n), dtype=int)
            _ = f.readline()
            for i in range(n):
                flows[i,:] = (list(map(int, f.readline().split())))
            for j in range(n):
                distances[j,:] = (list(map(int, f.readline().split())))
        return QAProblem(n, distances, flows)
    
    def reset_stats(self):
        self.best_fit_record = []
        self.best_fit = np.Inf
        self.best_sol = []




def main():
    problem = QAProblem.build_from_file(path="tai256c.dat")

    solutions = np.array([np.random.permutation(problem.n) for i in range(10)], dtype=int)

    fitnessess = np.array([problem(solution) for solution in solutions], dtype=int)

    plt.imshow(solutions, interpolation="none")
    plt.show()
    plt.imshow([fitnessess], interpolation="none")
    plt.show()



if __name__ == '__main__':
    main()