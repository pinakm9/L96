import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

config_id_1 = 1
config_id_2 = 2
ev_time = 100

df = pd.read_csv('results/W_{}_vs_{}.csv'.format(config_id_1, config_id_2))
df = df.loc[df['time'].isin([4*i for i in range(int(ev_time/4))])]
sns.boxplot(x=df['time'], y=df['Wasserstein_2'])
plt.savefig('dist plots/W_box_plot_{}_vs_{}.png'.format(config_id_1, config_id_2))
plt.clf()
sns.pointplot(x=df['time'], y=df['Wasserstein_2'])
plt.savefig('dist plots/W_error_bar_{}_vs_{}.png'.format(config_id_1, config_id_2))