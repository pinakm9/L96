import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import os
import warnings
warnings.filterwarnings("error")

@ut.timer
def experiment(model, filter, particle_counts, num_exprs, resampling_threshold, titles = None, file_paths = None, final_exact_mean = None, max_path = 1e4, **filter_kwargs):
    # compute average error for each particle count
    rmse, err_in_mean = [], []
    for particle_count in particle_counts:
        error, err_m, expr = 0.0, 0.0, 0
        while expr < num_exprs:
            # Generate paths
            hidden_path = model.hidden_state.generate_path()
            observed_path = model.observation.generate_path(hidden_path)
            if np.all(abs(hidden_path) < max_path):
                print('\rusing {:04} particles: experiment# {:03d}'.format(particle_count, expr), end = '')
                # create particle filter
                fltr = getattr(fl, filter)(model, particle_count, **filter_kwargs)
                try:
                    if filter == 'EnsembleKF':
                        fltr.update(observed_path)
                    else:
                        fltr.update(observed_path , threshold_factor = resampling_threshold, method = 'mean')
                    fltr.compute_error(hidden_path)
                    error += fltr.rmse
                    if final_exact_mean is not None:
                        err_m += np.linalg.norm(final_exact_mean(observed_path[1:]) - fltr.computed_trajectory[-1])
                    expr += 1
                except:
                    pass
        rmse.append(error/num_exprs)
        if final_exact_mean is not None:
            err_in_mean.append(err_m/num_exprs)
    # print a newline for clarity
    print()
    # plot average error vs particle_count
    if file_paths is not None:
        x = np.linspace(min(particle_counts), max(particle_counts), 100)
        #y = scipy.interpolate.make_interp_spline(particle_counts, rmse)(x)
        plt.plot(particle_counts, rmse)
        plt.xlabel('number of particles')
        plt.ylabel('average rmse')
        if titles is not None:
            plt.title(titles[0])
        plt.savefig(file_paths[0])
        # plot error in mean vs particle_count
        if len(file_paths) > 0 and final_exact_mean is not None:
            plt.clf()
            #y = scipy.interpolate.make_interp_spline(particle_counts, err_in_mean)(x)
            plt.plot(particle_counts, err_in_mean)
            plt.xlabel('number of particles')
            plt.ylabel('error in mean')
            if titles is not None and len(titles) > 0:
                plt.title(titles[1])
            plt.savefig(file_paths[1])
    return rmse, err_in_mean
