import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import compare_dist as cd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2021)
ev_time = 100
num_obs = 10
k = 4

expr_folder = 'results'
config_folder = 'config'
plot_folder = 'KL plots'
configs = os.listdir(config_folder)[:2]
print(configs)
#np.zeros((num_obs, ev_time))
for i in range(len(configs)):
    config_id_1 = configs[i].split('_')[1][:-5]
    for j in range(i+1, len(configs), 1):
        config_id_2 = configs[j].split('_')[1][:-5]
        data = {'time': [], 'obs_id':[], 'KL_div': []}
        for obs in range(num_obs):
            assml_file_1 = expr_folder + '/obs_{}_config_{}.h5'.format(obs, config_id_1)
            assml_file_2 = expr_folder + '/obs_{}_config_{}.h5'.format(obs, config_id_2)
            print('comparing {} vs {} for observation realization {} ...'.format(config_id_1, config_id_2, obs))
            pf_comp = cd.PFComparison(assml_file_1, assml_file_2)
            data['time'] += list(range(ev_time))
            data['obs_id'] += [obs] * ev_time 
            data['KL_div'] += list(pf_comp.compare_KL(k=k, saveas=None)) #pf_comp.compare_with_resampling(num_samples=1000, k=100, noise_cov=0.01, saveas=None)
        
        df = pd.DataFrame(data)
        df.to_csv('results/KL_{}_{}_vs_{}.csv'.format(k, config_id_1, config_id_2), index=False)
        
        
        
        
        """
        kl_dist = kl_dist / num_obs
        fig = plt.figure(figsize=(8, 8))
        x = np.array(list(range(ev_time)))
        idx_1 = np.where(kl_dist >= 0.0)
        idx_2 = np.where(kl_dist < 0.0)
        x_, y_ = x[idx_1], kl_dist[idx_1]
        plt.scatter(x_, y_, color='blue')
        plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
        plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
        plt.xlabel('assimilation step')
        plt.ylabel('approximate KL divergence')
        plt.title('{} vs {}'.format(config_id_1, config_id_2))
        plt.legend()
        plt.savefig(plot_folder + '/{}_vs_{}.png'.format(config_id_1, config_id_2))
        """
    