import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pdb
from tqdm import tqdm, trange
from agent import Mouse
from classic_task import Box
from plotting_functions import *
np.random.seed(0)



if __name__ == '__main__':
    import pandas as pd

    n_trials = 2000
    x = np.linspace(0,50, n_trials  * 10)
    cue_reaction_times = np.random.geometric(0.01, x.shape[0]) #(np.exp(-x)*3+ np.random.rand(x.shape[0])) + 5
    movement_times = np.random.geometric(0.01, x.shape[0]) * 2
    e = Box(punish=True)
    a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.01, psi=0.1)

    all_PEs = []
    all_APEs = []
    all_trial_types = []
    all_actions = []
    all_states = []
    all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])

    all_rewards, all_MSs, all_Ns, all_Ss, all_Vs = [], [], [], [], []
    with trange(n_trials) as t:
        for i in t:
            _, PEs, trial_type, action, states, state_changes, apes, trial_r, m_signals, values, novelties, saliences = a.one_trial(i)
            # pdb.set_trace()
            all_rewards.append(trial_r)
            mean_r = sum(all_rewards) / (i + 1.)
            t.set_description(f"avg. reward = {mean_r}")
            e.reset()
            all_states.append(states)
            all_PEs.append(PEs)
            all_trial_types.append(trial_type)
            #################################
            all_MSs.append(m_signals)
            all_Ns.append(novelties)
            all_Ss.append(saliences)
            all_Vs.append(values)
            ################################
            all_actions.append(action)
            all_state_changes = all_state_changes.append(state_changes, ignore_index=True)
            all_APEs.append(apes)

continuous_time_PEs = np.concatenate(all_PEs).ravel()
continuous_time_APEs = np.concatenate(all_APEs).ravel()
continuous_time_MSs = np.squeeze(np.concatenate(all_MSs))[:,0] # np.concatenate(all_MSs).ravel()
continuous_time_Ns = np.squeeze(np.concatenate(all_Ns))
continuous_time_Ss = np.squeeze(np.concatenate(all_Ss))
continuous_time_Vs = np.squeeze(np.concatenate(all_Vs))

reward_times = np.where(np.asarray(a.reward_history) == 1)[0]

left_inds = all_state_changes[all_state_changes['action taken'] == 'Left'].index
cue_left = all_state_changes.shift(periods=1).iloc[left_inds]
cue_left_times = cue_left['time stamp'].values

right_inds = all_state_changes[all_state_changes['action taken'] == 'Right'].index
cue_right = all_state_changes.shift(periods=1).iloc[left_inds]
cue_right_times = cue_right['time stamp'].values

low_tone_times = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['state name'] == 'HighLeft'].index.values].values
high_tone_times = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['state name'] == 'LowRight'].index.values].values
left_choices = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['action taken'] == 'Left'].index.values].values
right_choices = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['action taken'] == 'Right'].index.values].values


# novelties, values, saliences
fig, axs = plt.subplots(5, 4, figsize=[10, 8])
states = ['Left cues', 'Right cues', 'Reward', 'Contra', 'Ipsi']
time_stamps = {'Left cues': cue_left_times, 'Right cues': cue_right_times, 'Reward': reward_times,
               'Contra': left_choices, 'Ipsi': right_choices}
models = {'APE': continuous_time_APEs, 'RPE': continuous_time_PEs, 'Novelty': np.sum(continuous_time_Ns, axis=1),
          'Salience': np.sum(continuous_time_Ss, axis=1), 'Movement': continuous_time_MSs}

axs[0, 0].set_ylabel('RPE')
axs[1, 0].set_ylabel('Salience')
axs[2, 0].set_ylabel('Novelty')
axs[3, 0].set_ylabel('Movement')
axs[4, 0].set_ylabel('APE')

axs[0, 0].set_title('Left cues')
axs[0, 1].set_title('Right cues')
axs[0, 2].set_title('Reward')

plot_early_and_late(models['RPE'],  time_stamps['Left cues'], axs[0, 0], ' ', window=6)
plot_early_and_late(models['RPE'],  time_stamps['Right cues'], axs[0, 1], ' ', window=6)
plot_early_and_late(models['RPE'],  time_stamps['Reward'], axs[0, 2], ' ', window=6)
plot_change_over_time(models['RPE'], time_stamps['Left cues'], axs[0, 3])


plot_early_and_late(models['Salience'],  time_stamps['Left cues'], axs[1, 0], ' ', window=6)
plot_early_and_late(models['Salience'],  time_stamps['Right cues'], axs[1, 1], ' ', window=6)
plot_early_and_late(models['Salience'],  time_stamps['Reward'], axs[1, 2], ' ', window=6)
plot_change_over_time(models['Salience'], time_stamps['Left cues'], axs[1, 3])


plot_early_and_late(models['Novelty'],  time_stamps['Left cues'], axs[2, 0], ' ', window=6)
plot_early_and_late(models['Novelty'],  time_stamps['Right cues'], axs[2, 1], ' ', window=6)
plot_early_and_late(models['Novelty'],  time_stamps['Reward'], axs[2, 2], ' ', window=6)
plot_change_over_time(models['Novelty'], time_stamps['Left cues'], axs[2, 3])

axs[3, 0].set_title('Contra choices')
axs[3, 1].set_title('Ipsi choices')
axs[3, 2].set_title('Reward')

plot_early_and_late(models['Movement'],  time_stamps['Contra'], axs[3, 0], ' ', window=6)
plot_early_and_late(models['Movement'],  time_stamps['Ipsi'], axs[3, 1], ' ', window=6)
plot_early_and_late(models['Movement'],  time_stamps['Reward'], axs[3, 2], ' ', window=6)
plot_change_over_time(models['Movement'], time_stamps['Contra'], axs[3, 3])


plot_early_and_late(models['APE'],  time_stamps['Contra'], axs[4, 0], ' ', window=6)
plot_early_and_late(models['APE'],  time_stamps['Ipsi'], axs[4, 1], ' ', window=6)
plot_early_and_late(models['APE'],  time_stamps['Reward'], axs[4, 2], ' ', window=6)
plot_change_over_time(models['APE'], time_stamps['Contra'], axs[4, 3])

for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


plt.tight_layout()
plt.savefig("/Users/francesca/Documents/Model_of_2AC_task_figs/all_models.pdf")
plt.show()