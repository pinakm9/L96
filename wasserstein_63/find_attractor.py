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
import filter as fl
import config as cf
import numpy as np
import Lorenz96
import json
from bpf_plotter import plot_ensemble_evol
import os
import pandas as pd
# locate config files to create models
config_folder = 'config_0.4'
config_files = os.listdir(config_folder)

for file in config_files:
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
    x0 = np.random.normal(size=dim)
    model, gen_path = Lorenz96.get_model(x0=x0, size=ev_time, prior_cov=prior_cov, obs_cov=obs_cov,  shift=shift, obs_gap=obs_gap)


    # assimilation using a bootstrap particle filter
    hidden_path = gen_path(10000)
    pd.DataFrame(hidden_path).to_csv('path_{}.csv'.format(dim), header=None, index=None)
    print(hidden_path[-1])

    