import matplotlib.pyplot as plt
import matplotlib
from tqdm import trange
from agent import Mouse
from state_change.state_change_env import WhiteNoiseBox
from helper_functions.plotting_functions import *
import os
from directories import save_dir
np.random.seed(0)



if __name__ == '__main__':
    import pandas as pd

    font = {'size': 7}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family']

    n_trials = 3000
    x = np.linspace(0, 50, n_trials * 10)
    cue_reaction_times = np.random.geometric(0.01, x.shape[0]) #(np.exp(-x)*3+ np.random.rand(x.shape[0])) + 5
    movement_times = np.random.geometric(0.01, x.shape[0]) * 2
    e = WhiteNoiseBox(punish=True)
    a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.01, psi=0.2)

    all_PEs = []
    all_APEs = []
    all_trial_types = []
    all_actions = []
    all_states = []
    all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])

    all_rewards, all_MSs, all_Ns, all_Ss, all_Vs = [], [], [], [], []
    with trange(n_trials) as t:
        for i in t:
            _, PEs, trial_type, action, states, state_changes, apes, trial_r, m_signals, values, novelties, saliences, _ = a.one_trial(i)
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
white_noise_times = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['state name'] == 'WhiteNoiseLeft'].index.values].values

# novelties, values, saliences
fig, axs = plt.subplots(5, 1, figsize=[2, 6])
states = ['Left cues', 'Right cues', 'White noise', 'Reward', 'Contra', 'Ipsi']
#time_stamps = {'High tones': high_tone_times, 'Low tones': low_tone_times, 'White noise': white_noise_times, 'Reward': reward_times,
#               'Contra': left_choices, 'Ipsi': right_choices}

time_stamps = {'Left cues': cue_left_times, 'Right cues': cue_right_times, 'White noise': white_noise_times, 'Reward': reward_times,
               'Contra': left_choices, 'Ipsi': right_choices}
models = {'APE': continuous_time_APEs, 'RPE': continuous_time_PEs, 'Novelty': np.sum(continuous_time_Ns, axis=1),
          'Salience': np.sum(continuous_time_Ss, axis=1), 'Movement': continuous_time_MSs}

axs[0].set_ylabel('RPE')
axs[1].set_ylabel('Salience')
axs[2].set_ylabel('Novelty')
axs[3].set_ylabel('Movement')
axs[4].set_ylabel('APE')


plot_change_over_time(models['RPE'], time_stamps['Left cues'], axs[0])
plot_change_over_time(models['Salience'], time_stamps['Left cues'], axs[1])
plot_change_over_time(models['Novelty'], time_stamps['Left cues'], axs[2])
plot_change_over_time(models['Movement'], time_stamps['Contra'], axs[3])
plot_change_over_time(models['APE'], time_stamps['Contra'], axs[4])

for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


plt.tight_layout()
plt.savefig(os.path.join(save_dir, "all_models_white_noise.pdf"))

state_change_left_choices = all_state_changes['time stamp'][
    all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'] >= 2000) ].index.values].values
pre_left_choices = all_state_changes['time stamp'][
    all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'] < 2000) ].index.values].values
cue_stamps = (low_tone_times, white_noise_times)
movement_stamps = (pre_left_choices, state_change_left_choices)
model_timestamp_type = {'APE': movement_stamps, 'RPE': cue_stamps, 'Novelty': cue_stamps,
          'Salience': cue_stamps, 'Movement': movement_stamps}
fig, axs = plt.subplots(5, 1, figsize=[2, 6])
colours = cm.viridis(np.linspace(0, 0.8, 3))
axs[0].set_ylabel('RPE')
axs[1].set_ylabel('Salience')
axs[2].set_ylabel('Novelty')
axs[3].set_ylabel('Movement')
axs[4].set_ylabel('APE')

for ax_num, ax in enumerate(axs):
    model_type = list(models)[ax_num]
    plot_average_response(models[model_type], model_timestamp_type[model_type][0], ax, color=colours[1])
    plot_average_response(models[model_type], model_timestamp_type[model_type][1], ax, color=colours[0])
    ax.set_ylabel(model_type)
    #ax.set_ylim([0, 1])

for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "state_change_trace_view_rescaled_y_axis.pdf"))
plt.show()