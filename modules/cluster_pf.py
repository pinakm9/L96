import numpy as np
import utility as ut
import filter as fl
from scipy.optimize import linprog
from scipy.spatial import Delaunay
import copy


class ClusterPF(fl.ParticleFilter):
    def __init__(self, index_map, cluster_map, model, particle_count, folder = None, particles = None):
        super().__init__(model, particle_count, folder, particles)
        self.index_map = index_map
        self.cluster_map = cluster_map
        self.sigma2 = np.diag(self.model.observation.sigma)
        self.H = self.model.H#np.zeros((self.model.observation.dimension, self.model.hidden_state.dimension))
        #for i in range(self.model.observation.dimension):
        #   self.H[i, self.index_map[i]] = 1.0
        self.weights = np.ones((self.model.observation.dimension, self.particle_count)) / self.particle_count



    def compute_adjustment(self, y, obs_index):
        # compute R_f and U
        idx = self.cluster_map[obs_index]
        particles = [particle[[idx]] for particle in self.particles]
        R_f = 0.
        U = np.ones((int(self.model.hidden_state.dimension / self.model.observation.dimension), self.particle_count))
        
        x_f = np.average(particles, weights = self.weights[obs_index, :], axis = 0)
        for k in range(self.particle_count):
            p = (particles[k] - x_f)
            R_f += self.weights[obs_index, k] * np.outer(p, p)
            U[:, k] = np.sqrt(self.weights[obs_index, k]) * p

        # compute V, Sigma
        Sigma, V = np.linalg.eig(R_f)
        Sigma = np.diag(np.sqrt(Sigma))

        # Compute W, D
        H = self.H[obs_index, [idx]]
        r_o = self.sigma2[obs_index]
        #print(np.linalg.multi_dot([Sigma, V.T, H.T, H, V, Sigma])/r_o)
        D, W = np.linalg.eig(np.linalg.multi_dot([Sigma, V.T, H.T, H, V, Sigma])/r_o)
        D = np.diag(D) 

        # Compute A
        D_ = np.diag([1.0/np.sqrt(1.0 + d) for d in np.diag(D)])
        Sigma_ = np.diag([1.0/d for d in np.diag(Sigma)])
        A =  np.linalg.multi_dot([V, Sigma, W, D_, Sigma_, V.T])

        # Compute adjusted particles
        P = np.dot(R_f, H.T)
        Q = np.dot(H, P) + r_o 
        #print(P, Q)
        G = np.dot(P, np.linalg.inv(Q))
        #print(np.dot(G, y - np.dot(H, x_f)))
        x_a = x_f + np.dot(G, y - np.dot(H, x_f))
        for i in range(self.particle_count):
            self.particles[i, [idx]] = x_a + np.dot(A, particles[i] - x_f)

    def partial_resample(self, particles, weights):
        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(self.particle_count)) / self.particle_count
        indices = np.zeros(self.particle_count, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < self.particle_count:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        self.resampled_particles = np.array([particles[i] for i in indices])
        return self.resampled_particles, len(np.unique(indices))


    def one_step_update(self, observation):
        if self.current_time > 0:
            self.particles = np.array([self.model.hidden_state.sims[self.current_time].algorithm(self.current_time, particle)\
                                         for particle in self.particles]) 
        for j, y in enumerate(observation):
            xs = [particle[self.index_map[j]] for particle in self.particles]
            print(y, max(xs), min(xs))
            # hard threshold criterion 
            if (y > min(xs) and y < max(xs)):
                print('htc met')
                self.compute_adjustment(y, j)
            else:
                # standard PF update
                self.weights[j, :] *= np.exp(-0.5 * (y - self.particles[:, self.index_map[j]])**2 / self.sigma2[j])
                print('sum', self.weights[j, :].sum())
                self.weights[j, :] /= self.weights[j, :].sum()
                
                # resampling
                if 1.0 / (self.weights[j, :]**2).sum() < self.particle_count / 2.:
                    print('resampling')
                    idx = self.cluster_map[j]
                    particles = [particle[[idx]] for particle in self.particles]
                    particles, _ = self.partial_resample(particles, self.weights[j, :]) 
                    for i in range(self.particle_count):
                        # post-regularization
                        self.particles[i, [idx]] = particles[i] + np.random.normal(loc=0.5, size=len(idx))
                    self.weights[j, :] = 1. / self.particle_count

    
    @ut.timer
    def update(self, observations, method = 'mean', record_path = None, **params):
        """
        Description:
            Updates using all the obeservations using self.one_step_update 
        Args:
            observations: list/np.array of observations to pass to self.one_step_update
            method: method for computing trajectory, default = 'mean'
            resampling method: method for resampling, default = 'systematic'
            record_path: file path for storing evolution of particles
        Returns:
            self.weights
        """
        self.observed_path = observations
        for observation in self.observed_path:
            self.one_step_update(observation = observation)
            self.resampling_tracker.append(True)
            if method is not None:
                self.compute_trajectory(method = method)
            self.record(observation)
            print('Assimilation done for time:{}'.format(self.current_time))
            if self.status == 'failure':
                break
            else:
               self.status = 'success' 
            self.current_time += 1
        return self.status


    def compute_trajectory(self, method = 'mean'):
        """
        Description:
            Computes hidden trajectory
        """
        if method == 'mean':
            mean = np.zeros(self.model.hidden_state.dimension)
            for j in range(self.model.observation.dimension):
                idx = self.cluster_map[j]
                particles = [particle[[idx]] for particle in self.particles]
                mean[[idx]] = np.average(particles, weights = self.weights[j, :], axis = 0)
        self.computed_trajectory = np.append(self.computed_trajectory, [mean], axis = 0)
        return self.computed_trajectory
              
               

            




            