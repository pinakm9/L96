import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
import utility as ut
import tables
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import bpf_plotter as bpfp
import tensorflow as tf 
import wasserstein as ws

class DistComparison:
    """
    Class for comparing distributions
    """
    def __init__(self, ensemble_1, ensemble_2, weights_1=None, weights_2=None):
        self.ensemble_1 = ensemble_1
        self.ensemble_2 = ensemble_2
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        self.n = len(self.ensemble_1)
        self.m = len(self.ensemble_2)
        self.dim = len(ensemble_1[0])

    #@ut.timer
    def compute_KL(self, k):
        """
        k: for nearest neighbor computation
        """
        neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(self.ensemble_1)
        dist, _ = neigh.kneighbors(self.ensemble_1)
        r_k = [dist[i][k] for i in range(self.n)]
        #print(dist.shape)
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(self.ensemble_2)
        dist, _ = neigh.kneighbors(self.ensemble_1)
        s_k = [dist[i][k - 1] for i in range(self.n)]
        #print(dist.shape)
        kl = sum([np.log(s_k[i]/r_k[i]) for i in range(self.n)]) * self.dim / self.n +  self.dim * np.log(self.m/(self.n-1))
        return kl

    def systematic_noisy_resample(self, ensemble, weights, num_samples, noise_cov=0.01):
        """
        resamples new particles from a given ensemble
        """
        ensemble_size = len(ensemble)
        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(ensemble_size)) / ensemble_size
        indices = np.zeros(ensemble_size, 'i')
        cumulative_sum = np.cumsum(weights)
        # figure out significant particles
        i, j = 0, 0
        while i < ensemble_size:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        indices = list(set(indices))
        # figure out the number of offsprings for each particle
        offsprings = [0] * len(indices)
        weight_sum = sum([weights[i] for i in indices])
        for k, i in enumerate(indices):
            offsprings[k] = math.ceil(num_samples * weights[i] / weight_sum)
        new_particles = np.zeros((sum(offsprings), self.dim))
        mean = np.zeros(self.dim)
        cov = noise_cov * np.identity(self.dim)
        # resample
        j = 0
        for k, i in enumerate(indices):
            new_particles[j] = ensemble[i]
            new_particles[j+1: j+offsprings[k]]= ensemble[i] + np.random.multivariate_normal(mean, cov, size=offsprings[k] - 1)
            j += offsprings[k]
        return np.array([new_particles[i] for i in np.random.choice(sum(offsprings), num_samples, replace=False)])

    def compute_KL_with_weights(self, num_samples=1000, k=100, noise_cov=0.01):
        if self.weights_1 is None:
            self.weights_1 = np.ones(self.n) / self.n
        if self.weights_2 is None:
            self.weights_2 = np.ones(self.m) / self.m
        print('Resampling from first distribution ...')
        ensemble_1 = self.systematic_noisy_resample(self.ensemble_1, self.weights_1, num_samples + 1, noise_cov)
        print('Resampling from second distribution ...')
        ensemble_2 = self.systematic_noisy_resample(self.ensemble_2, self.weights_2, num_samples, noise_cov)
        print('Computing nearest neighbors ...')
        neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(ensemble_1)
        dist, _ = neigh.kneighbors(ensemble_1)
        r_k = [dist[i][k] for i in range(num_samples + 1)]
        #print(dist.shape)
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(ensemble_2)
        dist, _ = neigh.kneighbors(ensemble_1)
        s_k = [dist[i][k - 1] for i in range(num_samples + 1)]
        kl = sum([np.log(s_k[i]/r_k[i]) for i in range(num_samples + 1)]) * self.dim / (num_samples + 1)
        print('Estimated KL divergence = {}'.format(kl))
        return kl


class PFComparison:
    def __init__(self, assml_file_1, assnl_file_2):
        self.file_1 = assml_file_1
        self.file_2 = assnl_file_2

    @ut.timer
    def compare_with_kde(self, num_samples, k, bandwidth, saveas=None):
        hdf5_1 = tables.open_file(self.file_1, 'r')
        hdf5_2 = tables.open_file(self.file_2, 'r')
        iterations = len(hdf5_1.root.observation.read().tolist())
        kl_dist = np.zeros(iterations, dtype=np.float32)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = np.array(getattr(hdf5_2.root.particles, 'time_' + str(itr)).read().tolist())
            weights_1 = np.array(getattr(hdf5_1.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            weights_2 = np.array(getattr(hdf5_2.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            # remove zero weights
            idx_1 = np.where(weights_1 > 1e-10)
            idx_2 = np.where(weights_2 > 1e-10)
            kde_1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(ensemble_1[idx_1], sample_weight=weights_1[idx_1])
            kde_2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(ensemble_2[idx_2], sample_weight=weights_2[idx_2])
            ensemble_1 = kde_1.sample(num_samples + 1)
            ensemble_2 = kde_2.sample(num_samples)
            dist_comp = DistComparison(ensemble_1, ensemble_2)
            kl_dist[itr] = dist_comp.compute_KL(k)
        hdf5_1.close()
        hdf5_2.close()
        pd.DataFrame(kl_dist).to_csv(saveas + '.csv' if saveas is not None else 'filter_comparison.csv', header=None, index=None)
        plt.figure(figsize = (8, 8))
        x = np.array(list(range(iterations)))
        idx_1 = np.where(kl_dist >= 0.0)
        idx_2 = np.where(kl_dist < 0.0)
        x_, y_ = x[idx_1], kl_dist[idx_1]
        plt.scatter(x_, y_, color='blue')
        plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
        """
        try:
            ransac = linear_model.RANSACRegressor()
            x__ = x_.reshape((-1, 1))
            ransac.fit(x__, y_)
            plt.plot(x_, ransac.predict(x__), '--r', color='blue', label='RANSAC trend (computed from blue dots)')
        except:
            x_2 = (x_**2).sum()
            x_1 = x_.sum()
            A = np.array([[x_2, x_1], [x_1, len(x_)]])
            a, b = np.linalg.solve(A, [(x_*y_).sum(), y_.sum()])
            plt.plot(x_, a*x_ + b, '--r', color='blue', label='linear trend (computed from blue dots)')
        #"""
        plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
        plt.xlabel('assimilation step')
        plt.ylabel('approximate KL divergence')
        plt.title('{} vs {}'.format(self.file_1[:-3], self.file_2[:-3]))
        plt.legend()
        if saveas is None:
            saveas = 'filter_comparison.png'
        else:
            saveas = saveas + '.png'
        plt.savefig(saveas)

    @ut.timer
    def compare_with_resampling(self, num_samples, k, noise_cov=0.01, saveas=None):
        hdf5_1 = tables.open_file(self.file_1, 'r')
        hdf5_2 = tables.open_file(self.file_2, 'r')
        iterations = len(hdf5_1.root.observation.read().tolist())
        kl_dist = np.zeros(iterations, dtype=np.float32)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = np.array(getattr(hdf5_2.root.particles, 'time_' + str(itr)).read().tolist())
            weights_1 = np.array(getattr(hdf5_1.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            weights_2 = np.array(getattr(hdf5_2.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            dist_comp = DistComparison(ensemble_1, ensemble_2, weights_1, weights_2)
            kl_dist[itr] = dist_comp.compute_KL_with_weights(num_samples, k, noise_cov)
        hdf5_1.close()
        hdf5_2.close()
        pd.DataFrame(kl_dist).to_csv(saveas + '.csv' if saveas is not None else 'filter_comparison.csv', header=None, index=None)
        if saveas is not None:
            plt.figure(figsize = (8, 8))
            x = np.array(list(range(iterations)))
            idx_1 = np.where(kl_dist >= 0.0)
            idx_2 = np.where(kl_dist < 0.0)
            x_, y_ = x[idx_1], kl_dist[idx_1]
            plt.scatter(x_, y_, color='blue')
            plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
            plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
            plt.xlabel('assimilation step')
            plt.ylabel('approximate KL divergence')
            plt.title('{} vs {}'.format(self.file_1[:-3], self.file_2[:-3]))
            plt.legend()
            if saveas is None:
                saveas = 'filter_comparison.png'
            else:
                saveas = saveas + '.png'
            plt.savefig(saveas)
        else:
            return kl_dist

    @ut.timer
    def compare_KL(self, k, saveas=None):
        hdf5_1 = tables.open_file(self.file_1, 'r')
        hdf5_2 = tables.open_file(self.file_2, 'r')
        iterations = len(hdf5_1.root.observation.read().tolist())
        kl_dist = np.zeros(iterations, dtype=np.float32)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = np.array(getattr(hdf5_2.root.particles, 'time_' + str(itr)).read().tolist())
            dist_comp = DistComparison(ensemble_1, ensemble_2)
            kl_dist[itr] = dist_comp.compute_KL(k)
        hdf5_1.close()
        hdf5_2.close()
        #pd.DataFrame(kl_dist).to_csv(saveas + '.csv' if saveas is not None else 'filter_comparison.csv', header=None, index=None)
        if saveas is not None:
            plt.figure(figsize = (8, 8))
            x = np.array(list(range(iterations)))
            idx_1 = np.where(kl_dist >= 0.0)
            idx_2 = np.where(kl_dist < 0.0)
            x_, y_ = x[idx_1], kl_dist[idx_1]
            plt.scatter(x_, y_, color='blue')
            plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
            plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
            plt.xlabel('assimilation step')
            plt.ylabel('approximate KL divergence')
            plt.title('{} vs {}'.format(self.file_1[:-3], self.file_2[:-3]))
            plt.legend()
            if saveas is None:
                saveas = 'filter_comparison.png'
            else:
                saveas = saveas + '.png'
            plt.savefig(saveas)
        else:
            return kl_dist


    @ut.timer
    def compare_W(self, epsilon=0.01, num_iter=200, p=2, saveas=None):
        hdf5_1 = tables.open_file(self.file_1, 'r')
        hdf5_2 = tables.open_file(self.file_2, 'r')
        iterations = len(hdf5_1.root.observation.read().tolist())
        kl_dist = np.zeros(iterations, dtype=np.float32)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = np.array(getattr(hdf5_2.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_1 = tf.convert_to_tensor(ensemble_1, dtype=tf.float32)
            ensemble_2 = tf.convert_to_tensor(ensemble_2, dtype=tf.float32)
            kl_dist[itr] = ws.sinkhorn_loss(ensemble_1, ensemble_2, epsilon=epsilon, num_iter=num_iter, p=p) ** (1.0/p)
        hdf5_1.close()
        hdf5_2.close()
        #pd.DataFrame(kl_dist).to_csv(saveas + '.csv' if saveas is not None else 'filter_comparison.csv', header=None, index=None)
        if saveas is not None:
            plt.figure(figsize = (8, 8))
            x = np.array(list(range(iterations)))
            idx_1 = np.where(kl_dist >= 0.0)
            idx_2 = np.where(kl_dist < 0.0)
            x_, y_ = x[idx_1], kl_dist[idx_1]
            plt.scatter(x_, y_, color='blue')
            plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
            plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
            plt.xlabel('assimilation step')
            plt.ylabel('approximate KL divergence')
            plt.title('{} vs {}'.format(self.file_1[:-3], self.file_2[:-3]))
            plt.legend()
            if saveas is None:
                saveas = 'filter_comparison.png'
            else:
                saveas = saveas + '.png'
            plt.savefig(saveas)
        else:
            return kl_dist

class PFvsKF:
    def __init__(self, pf_file, kf_file):
        self.pf_file = pf_file
        self.kf_file = kf_file

    @ut.timer
    def compare_with_resampling(self, num_samples, k, noise_cov=0.01, saveas=None):
        hdf5_1 = tables.open_file(self.pf_file, 'r')
        npy_2 = np.load(self.kf_file)
        iterations = len(hdf5_1.root.observation.read().tolist())
        kl_dist = np.zeros(iterations, dtype=np.float32)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = npy_2[itr].T
            weights_1 = np.array(getattr(hdf5_1.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            dist_comp = DistComparison(ensemble_1, ensemble_2, weights_1)
            kl_dist[itr] = dist_comp.compute_KL_with_weights(num_samples, k, noise_cov)
        hdf5_1.close()
        
        if saveas is not None:
            pd.DataFrame(kl_dist).to_csv(saveas + '.csv', header=None, index=None)
            plt.figure(figsize = (8, 8))
            x = np.array(list(range(iterations)))
            idx_1 = np.where(kl_dist >= 0.0)
            idx_2 = np.where(kl_dist < 0.0)
            x_, y_ = x[idx_1], kl_dist[idx_1]
            plt.scatter(x_, y_, color='blue')
            plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
            plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
            plt.xlabel('assimilation step')
            plt.ylabel('approximate KL divergence')
            plt.title('{} vs {}'.format(self.pf_file[:-3], self.kf_file[:-4]))
            plt.legend()
            if saveas is None:
                saveas = 'filter_comparison.png'
            else:
                saveas = saveas + '.png'
            plt.savefig(saveas)
            del npy_2
        else:
            return kl_dist

    

    @ut.timer
    def compare_with_ensemble_plots(self, saveas=None):
        hdf5_1 = tables.open_file(self.pf_file, 'r')
        npy_2 = np.load(self.kf_file)
        iterations = len(hdf5_1.root.observation.read().tolist())
        ens_plotter = bpfp.EnsemblePlotter(pt_size=80)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = npy_2[itr].T
            weights_1 = np.array(getattr(hdf5_1.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            weights_2 = np.ones(len(ensemble_2)) / len(ensemble_2)
            ens_plotter.plot_weighted_ensembles_2D([ensemble_1, ensemble_2], labels=['Particle Filter', 'EnKF'],\
                                                     weights=[weights_1, weights_2], weight_histogram=False, file_path=saveas)
        hdf5_1.close()
        
        