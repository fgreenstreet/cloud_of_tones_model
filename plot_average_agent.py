import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import label_axes_change_over_time
import matplotlib


font = {'size': 7}
matplotlib.rc('font', **font)
num_agents = 100
peaks = np.load('/Users/francesca/Documents/Model_of_2AC_task_figs/{}_agents_classic_exp.npy'.format(num_agents), allow_pickle=True)

APE_fig, APE_axs = plt.subplots(1, 1, figsize=[2, 2])
APE_mean = np.mean(peaks.item().get('APE contra'), axis=0)
APE_axs.plot(APE_mean, color='#3F888F')
APE_fig.set_tight_layout(True)
APE_axs = label_axes_change_over_time(APE_axs)
plt.savefig('/Users/francesca/Documents/Model_of_2AC_task_figs/{}_agents_classic_exp_APE_contra.pdf'.format(num_agents))


RPE_fig, RPE_axs = plt.subplots(1, 1, figsize=[2, 2])
RPE_mean = np.mean(peaks.item().get('RPE cues'), axis=0)
RPE_axs.plot(RPE_mean, color='#3F888F')
RPE_fig.set_tight_layout(True)
RPE_axs = label_axes_change_over_time(RPE_axs)
plt.savefig('/Users/francesca/Documents/Model_of_2AC_task_figs/{}_agents_classic_exp_RPE_cue.pdf'.format(num_agents))

plt.show()