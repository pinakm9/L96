"""
Plots evolution of ensembles
"""
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
print(module_dir)
sys.path.insert(0, module_dir + '/models')
sys.path.insert(0, module_dir + '/modules')
# import remaining modules
import filter as fl
import config as cf
import numpy as np
import Lorenz96_alt
import json
from bpf_plotter import plot_ensemble_evol
import os
import pandas as pd
# locate config files to create models
config_folder = 'config'
config_files = os.listdir(config_folder)

for file in config_files[2:]:
    print('loading configuration from {}'.format(config_folder + '/' + file))
    with open(config_folder + '/' + file) as f:
        config = json.load(f)
    # set random seed
    seed = config["Numpy seed"]
    np.random.seed(2022)
    # set model
    dim = config["Hidden state dimension"]
    ev_time = 500#config["Number of assimilation steps"]
    prior_cov = config["Prior covariance"]
    obs_cov = config["Observation covariance"]
    shift = config['Shift'][0]
    obs_gap = config["Observation gap"]
    x0_file = "path_{}.csv".format(dim)
    x0 = np.array(pd.read_csv(x0_file))[-1]
    model, gen_path = Lorenz96_alt.get_model(x0=x0, size=ev_time, prior_cov=prior_cov, obs_cov=obs_cov,  shift=shift, obs_gap=obs_gap)
    index_map = [2*i for i in range(int(dim/2))]
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
    pf = fl.ParticleFilter(model, particle_count = 1000, folder = cc.res_path)
    pf.update(observed_path, method = 'mean', resampling_method = 'systematic_noisy', pre_reg=0.5, post_reg=0.5)
    #print(pf.H(hidden_path[3]))
    #print(pf.find_eic(observed_path[1]))
    #print(pf.one_step_update(observed_path[1]))

    # document results
    if pf.status == 'success':
        #plot_ensemble_evol(cc.res_path + '/assimilation.h5', hidden_path, time_factor=1, pt_size=80, obs_inv=True)
        print(hidden_path - pf.computed_trajectory)
        pf.plot_trajectories(hidden_path, coords_to_plot=[0, 1, 38, 39], file_path=cc.res_path + '/trajectories.png', measurements=False)
        pf.compute_error(hidden_path)
        pf.plot_error(semilogy=True)
        config['Status'] = pf.status
        cc.add_params(config)
        cc.write(mode='json')