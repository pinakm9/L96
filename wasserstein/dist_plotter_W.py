import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

config_id_1 = 1
config_id_2 = 2
ev_time = 500
dim = 5
pf = 'bpf_200_sd'
df = pd.read_csv('results_b/W_{}_vs_{}.csv'.format(config_id_1, config_id_2))
df = df.loc[df['time'].isin([4*i for i in range(int(ev_time/4))])]
sns.boxplot(x=df['time'], y=df['Wasserstein_2'])
plt.show()
plt.savefig('dist_plots/W_{}_box_plot_{}_vs_{}_{}.png'.format(dim, config_id_1, config_id_2, pf))
plt.clf()
sns.pointplot(x=df['time'], y=df['Wasserstein_2'])
plt.savefig('dist_plots/W_{}_error_bar_{}_vs_{}_{}.png'.format(dim, config_id_1, config_id_2, pf))