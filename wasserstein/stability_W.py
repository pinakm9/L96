import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import compare_dist as cd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2021)
ev_time = 100
num_obs = 1

expr_folder = 'results_b'
config_folder = 'config'
plot_folder = 'dist plots'
configs = os.listdir(config_folder)[:2]
print(configs)
#np.zeros((num_obs, ev_time))
for i in range(len(configs)):
    config_id_1 = configs[i].split('_')[1][:-5]
    for j in range(i+1, len(configs), 1):
        config_id_2 = configs[j].split('_')[1][:-5]
        data = {'time': [], 'obs_id':[], 'Wasserstein_2': []}
        for obs in range(num_obs):
            assml_file_1 = expr_folder + '/{}_seed_{}#5/assimilation.h5'.format(config_id_1, obs)
            assml_file_2 = expr_folder + '/{}_seed_{}#5/assimilation.h5'.format(config_id_2, obs)
            print('comparing {} vs {} for observation realization {} ...'.format(config_id_1, config_id_2, obs))
            pf_comp = cd.PFComparison(assml_file_1, assml_file_2)
            data['time'] += list(range(ev_time))
            data['obs_id'] += [obs] * ev_time 
            data['Wasserstein_2'] += list(pf_comp.compare_W(saveas=None, num_iter=50)) #pf_comp.compare_with_resampling(num_samples=1000, k=100, noise_cov=0.01, saveas=None)
            #print(data['Wasserstein_2'])
        df = pd.DataFrame(data)
        df.to_csv('{}/W_{}_vs_{}.csv'.format(expr_folder, config_id_1, config_id_2), index=False)