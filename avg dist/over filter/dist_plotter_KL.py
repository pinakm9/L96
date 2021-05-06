import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

k = 4
config_id_1 = 1
config_id_2 = 2
ev_time = 100

df = pd.read_csv('results/KL_{}_{}_vs_{}.csv'.format(k, config_id_1, config_id_2))
df = df.loc[df['time'].isin([4*i for i in range(int(ev_time/4))])]
sns.boxplot(x=df['time'], y=df['KL_div'])
plt.savefig('dist plots/KL_box_plot_{}_{}_vs_{}.png'.format(k, config_id_1, config_id_2))
plt.clf()
sns.pointplot(x=df['time'], y=df['KL_div'])
plt.savefig('dist plots/KL_error_bar_{}_{}_vs_{}.png'.format(k, config_id_1, config_id_2))