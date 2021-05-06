"""
Plots evolution of ensembles
"""
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent.parent.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import genetic_pf as gpf
import config as cf
import numpy as np
import Lorenz96
import json
from bpf_plotter import plot_ensemble_evol
import os
import pandas as pd
# locate config files to create models
config_folder = 'config'
config_files = os.listdir(config_folder)

for file in config_files[1:]:
    print('loading configuration from {}'.format(config_folder + '/' + file))
    with open(config_folder + '/' + file) as f:
        config = json.load(f)
    # set random seed
    seed = config["Numpy seed"]
    np.random.seed(seed)
    # set model
    dim = config["Hidden state dimension"]
    ev_time = config["Number of assimilation steps"]
    prior_cov = config["Prior covariance"]
    obs_cov = config["Observation covariance"]
    shift = config['Shift'][0]
    obs_gap = config["Observation gap"]
    x0_file = "path_{}.csv".format(dim)
    x0 = np.array(pd.read_csv(x0_file))[-1]
    model, gen_path = Lorenz96.get_model(x0=x0, size=ev_time, prior_cov=prior_cov, obs_cov=obs_cov,  shift=shift, obs_gap=obs_gap)

    # set filter parameters
    particle_count = config["Particle count"]

    # set up logging
    config_id = file[:-5].split('_')[1]
    expr_name = '{}_pc_{}_s_{}'.format(config_id, prior_cov, shift)
    cc = cf.ConfigCollector(expr_name = expr_name, folder = str(script_dir) + '/results')

    # assimilation using a bootstrap particle filter
    hidden_path = gen_path(ev_time)
    pd.DataFrame(hidden_path).to_csv(cc.res_path + '/hidden_path.csv', header=None, index=None)
    observed_path = model.observation.generate_path(hidden_path)

    # assimilate
    print("starting assimilation ... ")
    pf = gpf.GeneticPF(model, particle_count = particle_count, max_generations_per_step=10, record_path = cc.res_path + '/assimilation.h5')
    pf.update(observed_path, method = 'mean')

    # document results
    if False:
        plot_ensemble_evol(cc.res_path + '/assimilation.h5', hidden_path, time_factor=1, pt_size=80, obs_inv=True)
    
    pf.plot_trajectories(hidden_path, coords_to_plot=[0, 9, 39], file_path=cc.res_path + '/trajectories.png', measurements=False)
    pf.compute_error(hidden_path)
    pf.plot_error(file_path=cc.res_path + '/l2_error.png')
    config['Status'] = pf.status
    cc.add_params(config)
    cc.write(mode='json')
