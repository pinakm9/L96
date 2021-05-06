"""
Plots evolution of ensembles
"""
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import filter as fl
import config as cf
import numpy as np
import Lorenz63_xy
import json
from bpf_plotter import plot_ensemble_evol
import os
import pandas as pd
# locate config files to create models
config_folder = 'config'
config_files = os.listdir(config_folder)

# set number of observation realizations
num_obs = 10

for i in range(num_obs):
    # set up random seed
    seed = np.random.randint(int(1e8))
    np.random.seed(seed)
    for file in config_files[:2]:
        print('loading configuration from {}'.format(config_folder + '/' + file))
        with open(config_folder + '/' + file) as f:
            config = json.load(f)
        # set model
        ev_time = config["Number of assimilation steps"]
        prior_cov = config["Prior covariance"]
        obs_cov = config["Observation covariance"]
        shift = config['Shift'][0]
        obs_gap = config["Observation gap"]
        model, gen_path = Lorenz63_xy.get_model(size=ev_time, prior_cov=prior_cov, obs_cov=obs_cov, shift=shift, obs_gap=obs_gap)

        # set filter parameters
        particle_count = config["Particle count"]
        resampling_method = config["Resampling method"]
        resampling_threshold = config["Resampling threshold"]
        noise = config["Resampling covariance"]

        # assimilation using a bootstrap particle filter
        hidden_path = gen_path(ev_time)
        #pd.DataFrame(hidden_path).to_csv(cc.res_path + '/hidden_path.csv', header=None, index=None)
        config_id = file[:-5].split('_')[1]
        
        # set up logging
        expr_name = 'obs_{}_config_{}'.format(i, config_id)
        cc = cf.ConfigCollector(expr_name = expr_name, folder = str(script_dir) + '/results')
        
        # generate observation and assimilate
        observed_path = model.observation.generate_path(hidden_path)
        print("starting assimilation for observation realization {} and configuration {} ... ".format(i, config_id))
        bpf = fl.ParticleFilter(model, particle_count = particle_count, record_path = 'results/{}.h5'.format(expr_name))
        bpf.update(observed_path, resampling_method = resampling_method, threshold_factor = resampling_threshold, method = 'mean', noise=noise)
        print('assimilation status: {}'.format(bpf.status))
        if bpf.status == 'faliure':
            exit()
        else:
            bpf.plot_trajectories(hidden_path, coords_to_plot=[0, 1, 2], file_path=cc.res_path + '/trajectories.png')
            bpf.compute_error(hidden_path)
            bpf.plot_error(file_path=cc.res_path + '/error.png')
        config['Status'] = bpf.status
        config['Numpy Seed'] = seed
        cc.add_params(config)
        cc.write(mode='json')
