import numpy as np
import matplotlib.pyplot as plt
RPE_fig, RPE_axs = plt.subplots(1, 3)
RPE_rewards = np.load('all_RPE_rewards.npy')
RPE_lowtones = np.load('all_RPE_lowtones.npy')
RPE_hightones = np.load('all_RPE_hightones.npy')
APE_left = np.load('all_APE_left.npy')
APE_right = np.load('all_APE_right.npy')
RPE_axs[0].plot(np.nanmean(RPE_rewards, axis=0)[: -50], color='#3F888F')
RPE_axs[1].plot(np.nanmean(RPE_lowtones, axis=0)[: -50], color='#3F888F')
RPE_axs[2]. plot(np.nanmean(RPE_hightones, axis=0)[: -50], color='#3F888F')

APE_fig, APE_axs = plt.subplots(1, 2)
APE_axs[0].plot(np.nanmean(APE_left, axis=0)[: -50], color='#3F888F')
APE_axs[1].plot(np.nanmean(APE_right, axis=0)[: -50], color='#3F888F')


for ax in RPE_axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Response size')


for ax in APE_axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Response size')
plt.show()