import numpy as np
import utility as ut
import filter as fl
from scipy.optimize import minimize_scalar,  minimize
import scipy.optimize as so
import copy

def gasp_cohn(x, y, c):
    """
    Gaspari-Cohn taper function
    """
    r = abs(x - y) / c
    if r >= 0. and r < 1.:
        return 1. - 5./3. * r**2 + 5./8.* r**3 + r**4 / 2. - r**5/4.
    elif r >= 1. and r < 2.:
        return 4. - 5.*r + 5./3. * r**2 + 5./8.* r**3 - r**4 / 2. + r**5/12. - 2./(3. * r)
    else:
        return 0.

class LocalPF(fl.ParticleFilter):
    def __init__(self, n_eff_t, r_loc, mixing_param, index_map, model, particle_count, folder = None, particles = None):
        super().__init__(model, particle_count, folder, particles)
        self.N_eff_t = n_eff_t
        self.r_loc = r_loc 
        self.mixing_param = mixing_param
        self.index_map = index_map
        self.sigma2 = np.diag(self.model.observation.sigma)
        self.obs_zero = np.zeros(self.model.observation.dimension)
        self.H = lambda x: self.model.observation.func(0, x, self.obs_zero)
        self.weights = np.ones((self.particle_count, self.model.hidden_state.dimension)) / self.particle_count
        
 
        
    def eic_objective(self, beta, observation, i):
        term_1, term_2 = 0., 0.
        for n in range(self.particle_count):
            y_ = self.H(self.particles[n])[i]
            p = np.exp(-(y_ - observation[i])**2 /(2.0 * beta * self.sigma2[i]))
            term_1 += p
            term_2 += p**2
        N_beta = term_1**2/term_2
        print('N_beta', N_beta)
        return (self.N_eff_t - N_beta)**2

    def find_eic(self, observation):
        self.beta = np.ones(self.model.observation.dimension)
        beta_hat = np.zeros(self.model.observation.dimension)
        for i in range(self.model.observation.dimension):
            fun = lambda x: self.eic_objective(x, observation, i)
            res = minimize_scalar(fun, bounds=(1., 10.), method='bounded')
            beta_hat[i] = res.x
            print('fun', res.fun**0.5)
            for k in range(self.model.observation.dimension):
                self.beta[k] += (beta_hat[i] - 1.) * gasp_cohn(self.index_map[i], self.index_map[k], self.r_loc)
        print(self.beta)
        return self.beta


    def partial_resample(self, particles, weights):
        """
        Description:
            Performs the systemic resampling algorithm used by particle filters.
            This algorithm separates the sample space into N divisions. A single random
            offset is used to to choose where to sample from for all divisions. This
            guarantees that every sample is exactly 1/N apart.

        Returns:
            number of unique particles after resampling
        """
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
            # pre-regularization
            self.particles = np.array([self.model.hidden_state.sims[self.current_time].algorithm(self.current_time, particle)\
                                      for particle in self.particles]) 
        self.weights = np.ones((self.particle_count, self.model.hidden_state.dimension)) / self.particle_count
        #elif len(self.particles) != self.particle_count:
        #    self.particles = self.model.hidden_state.sims[0].generate(self.particle_count)
        self.find_eic(observation)
        self.new_particles = copy.deepcopy(self.particles)
        w_hat = np.ones((self.particle_count, self.model.observation.dimension))
        w_tilde = np.ones((self.particle_count, self.model.observation.dimension))
        Omega_hat = np.ones((self.model.hidden_state.dimension, self.model.observation.dimension))
        for i in range(self.model.observation.dimension):
            
            for n in range(self.particle_count):
                w_hat[n][i] = np.exp(-(self.H(self.particles[n])[i] - observation[i])**2 /(2.0 * self.beta[i] * self.sigma2[i]))
                w_tilde[n][i] = np.exp(-(self.H(self.new_particles[n])[i] - observation[i])**2 /(2.0 * self.beta[i] * self.sigma2[i]))
            
            #print('sums', w_hat[:, i].sum(), w_tilde[:, i].sum())
            w_hat[:, i] /= w_hat[:, i].sum()
            w_tilde[:, i] /= w_tilde[:, i].sum()
            
            self.partial_resample(self.new_particles, w_tilde[:, i])
            
            for j in range(self.model.hidden_state.dimension):
                Omega_hat[j][i] = np.dot(w_hat[:, i], self.weights[:, j])
            
                for n in range(self.particle_count):
                    self.weights[n][j] *=  ((self.particle_count * w_hat[n][i] - 1.) * gasp_cohn(self.index_map[i], j, self.r_loc) + 1.)/self.particle_count
                    
                self.weights[:, j] /= self.weights[:, j].sum()
                
                # compute sigma_j
                bar_x_j = np.dot(self.weights[:, j], self.particles[:, j])
                sigma2_j = np.dot(self.weights[:, j], (self.particles[:, j] - bar_x_j)**2) / (1. - (self.weights[:, j]**2).sum())
         
                #print('denom {}'.format((1. - (self.weights[:, j]**2).sum())))

                l =  gasp_cohn(self.index_map[i], j, self.r_loc)
                if l > 1e-3:
                    c = (1. - l) / (Omega_hat[j][i] * self.particle_count * l)
                    r1 = np.sqrt(sigma2_j * (self.particle_count - 1.) / ((self.resampled_particles[:, j] - bar_x_j + c * (self.new_particles[:, j] - bar_x_j))**2).sum())
                    r2 = c * r1
                    #print('sea is', c, (self.resampled_particles[:, j] - bar_x_j + c * (self.new_particles[:, j] - bar_x_j)).sum())
                else:
                    r1 = 0.
                    r2 = 1.

                self.new_particles[:, j] = bar_x_j + self.mixing_param * r1 * (self.resampled_particles[:, j] - bar_x_j) +\
                                           (self.mixing_param * (r2 - 1.) + 1.) * (self.new_particles[:, j] - bar_x_j) +  np.random.normal(loc=0.0, scale=0.5, size=self.particle_count)
        self.particles = copy.deepcopy(self.new_particles)
        

    
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
            for j in range(self.model.hidden_state.dimension):
                self.particles[:, j] += np.random.normal(loc=0.0, scale=0.26, size=self.particle_count)
            self.resampling_tracker.append(True)
            if method is not None:
                self.compute_trajectory(method = method)
            self.record(observation)
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
        if method == 'mode':
            # for each time find the most likely particle
            new_hidden_state = self.particles[np.array(list(map(self.filtering_pdf, self.particles))).argmax()]
            self.computed_trajectory = np.append(self.computed_trajectory, [new_hidden_state], axis = 0)
        elif method == 'mean':
            new_hidden_state = np.array([np.dot(self.weights[:, j], self.particles[:, j]) for j in range(self.model.hidden_state.dimension)])
            self.computed_trajectory = np.append(self.computed_trajectory, [new_hidden_state], axis = 0)
        return self.computed_trajectory
    


class LocalPF2(LocalPF):
    def __init__(self, n_eff_t, r_loc, mixing_param, index_map, model, particle_count, folder = None, particles = None):
        super().__init__(n_eff_t, r_loc, mixing_param, index_map, model, particle_count, folder, particles)

    
    def n_eff(self, beta, observation, i):
        term_1, term_2 = 0., 0.
        for n in range(self.particle_count):
            y_ = self.H(self.particles[n])[i]
            p = np.exp(-(y_ - observation[i])**2 /(2.0 * beta * self.sigma2[i]))
            term_1 += p
            term_2 += p**2
        return term_1**2 / term_2


    def find_eic(self, observation):
        self.beta = np.ones(self.model.observation.dimension)
        beta_hat = np.zeros(self.model.observation.dimension)
        for i in range(self.model.observation.dimension):
            if self.n_eff(1., observation, i) < self.N_eff_t:
                fun = lambda x: self.eic_objective(x, observation, i) 
                res = minimize_scalar(fun, bounds=(1., 10.), method='bounded')
                beta_hat[i] = res.x
                print('fun', self.n_eff(res.x, observation, i))
            else:
                beta_hat[i] = 1.
            for k in range(self.model.observation.dimension):
                self.beta[k] += (beta_hat[i] - 1.) * gasp_cohn(self.index_map[i], self.index_map[k], self.r_loc)
        print(self.beta)
        return self.beta

    def compute_trajectory(self, method = 'mean'):
        """
        Description:
            Computes hidden trajectory
        """
        if method == 'mean':
            new_hidden_state = np.array([self.particles[:, j].sum()/self.particle_count for j in range(self.model.hidden_state.dimension)])
            self.computed_trajectory = np.append(self.computed_trajectory, [new_hidden_state], axis = 0)
        return self.computed_trajectory