import pulp 
import numpy as np
import scipy.stats as ss
import utility as ut

class EMD:
    def __init__(self, ensemble_1, ensemble_2, weights_1=None, weights_2=None, normalize=False):
        self.ensemble_1 = np.array(ensemble_1)
        self.ensemble_2 = np.array(ensemble_2)
        self.m = len(ensemble_1)
        self.n = len(ensemble_2)
        
        if weights_1 is None:
            self.weights_1 = np.ones(self.m)/self.m
        elif normalize:
            self.weights_1 = np.array(weights_1)/np.sum(weights_1)
        else:
            self.weights_1 = np.array(weights_1)
        
        if weights_2 is None:
            self.weights_2 = np.ones(self.n)/self.n
        elif normalize:
            self.weights_2 = np.array(weights_2)/np.sum(weights_2)
        else:
            self.weights_2 = np.array(weights_2)



    def set_distance(self, dist_matrix=None, p=None):
        if dist_matrix is not None:
            self.dm = dist_matrix
        elif p is not None:
            self.dm = np.zeros((self.m, self.n))
            for i in range(self.m):
                for j in range(self.n):
                    self.dm[i][j] = np.linalg.norm(self.ensemble_1[i] - self.ensemble_2[j], ord=p)**p

    @ut.timer
    def compute(self):
        # create the linear programming problem
        emd = pulp.LpProblem('earth_mover\'s_distance', pulp.LpMinimize)
        # create the flow matrix
        keys = []
        for i in range(self.m):
            for j in range(self.n):
                keys.append((i, j))
        f = pulp.LpVariable.dicts('f', keys, 0.0, 1.0, cat='Continuous')
        # add row and column sum constriants
        for i in range(self.m):
            emd += sum([f[(i, j)] for j in range(self.n)]) == self.weights_1[i]
        for j in range(self.n):
            emd += sum([f[(i, j)] for i in range(self.m)]) == self.weights_2[j]
        
        emd += sum([f[(i, j)] for i in range(self.m) for j in range(self.n)]) == min(np.sum(self.weights_1), np.sum(self.weights_2))
        # add objective function
        emd += sum([f[(i, j)] * self.dm[i][j] for i in range(self.m) for j in range(self.n)])

        #print(emd)
        #Build the solverModel for your preferred
        emd.solve(pulp.PULP_CBC_CMD(timeLimit=100.0))
        return emd.objective.value()


# test
d = 3
mean_1 = [1.0] * d
cov_1 = np.identity(d)
mean_2 = [1.0] * d
cov_2 = np.identity(d)
ensemble_1 = np.random.multivariate_normal(mean_1, cov_1, size=50)
ensemble_2 = np.random.multivariate_normal(mean_2, cov_2, size=50)
weights_1 = [ss.multivariate_normal.pdf(x, mean_1, cov_1) for x in ensemble_1]
weights_2 = [ss.multivariate_normal.pdf(x, mean_2, cov_2) for x in ensemble_2]
emd = EMD(ensemble_1, ensemble_2, weights_1, weights_2, True)
emd.set_distance(p=2)
print(emd.compute())
diff = np.average(ensemble_1, weights=weights_1, axis=0) - np.average(ensemble_2, weights=weights_2, axis=0)
print(np.linalg.norm(diff)**2)