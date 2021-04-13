import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pandas as pd
from tqdm import tqdm


class Box(object):
    def __init__(self, punish=False):
        self.high_to_left = True  # toggles whether High tone corresponds to correct action being left
        self.high_sound_prob = .5
        self.env_states = ['Start', 'High', 'Low', 'Outcome']
        self.actions = ['Left', 'Centre', 'Right', 'Idle']
        action_states = ['HighLeft', 'HighRight', 'LowLeft', 'LowRight']
        self.states = self.env_states + action_states
        self.n_states = len(self.states)
        self.state_idx = {s: idx for s, idx in zip(self.states, range(self.n_states))}
        self.n_actions = len(self.actions)
        self.action_idx = {a: idx for a, idx in zip(self.actions, range(self.n_actions))}

        self.current_state = self.states[0]
        self.time_in_state = np.zeros(self.n_states, dtype=int)
        self.timer = 0
        self.punish = punish

    def act(self, action, time_is_not_up):
        # maybe make animals not able to act every time step
        next_state = self.get_next_state(self.current_state, action, time_is_not_up)
        reward = self.get_reward(self.current_state, action, next_state)

        # adjust state timer
        if next_state != self.current_state:
            self.time_in_state = np.zeros(self.n_states)
        else:
            self.time_in_state[self.state_idx[self.current_state]] += 1

        self.current_state = next_state
        return next_state, reward

    def get_reward(self, state, action, next_state):
        if next_state != 'Outcome':
            reward_amount = 0
        else:
            if self.high_to_left:
                if state == 'HighLeft' or state == 'LowRight':
                    reward_amount = 1
                else:
                    reward_amount = 0
            else:
                if state == 'HighRight' or state == 'LowLeft':
                    reward_amount = 1
                else:
                    reward_amount = 0
        return reward_amount

    def get_next_state(self, state, action, timer_is_not_up):
        if state == 'Start':
            if action == 'Centre':
                if np.random.rand() <= self.high_sound_prob:
                    next_state = 'High'
                else:
                    next_state = 'Low'
            else:
                next_state = 'Start'

        elif state == 'High' or state == 'Low':
            if action == 'Idle':
                next_state = state
            elif action == 'Centre':
                if self.punish:
                    next_state = 'Outcome'
                else:
                    next_state = state
            else:
                next_state = state + action
        elif state == 'HighLeft' or state == 'HighRight' or state == 'LowLeft' or state == 'LowRight':
            if timer_is_not_up:
                next_state = state
            else:
                next_state = 'Outcome'
        elif state == 'Outcome':
            next_state = None

        else:
            raise ValueError('No valid state input.')

        return next_state

    def reset(self):
        """Reset environment to initial state.

        :return:
        """
        self.current_state = self.states[0]

    def in_terminal_state(self):
        """Return True if current state is terminal, False otherwise.

        :return:
        """
        return self.current_state == 'Outcome'


class Mouse(object):
    def __init__(self, reaction_times, movement_times, env=Box(), critic_learning_rate=.1, actor_learning_rate=.1, habitisation_rate =.1, inv_temp=5., psi=0.1):
        self.env = env
        self.inv_temp = inv_temp  # inverse temperature param for softmax decision func
        self.critic_lr = critic_learning_rate
        self.critic_value = np.zeros(self.env.n_states)  # weights parameterising value function
        self.actor_lr = actor_learning_rate
        self.actor_value = np.zeros([self.env.n_actions, self.env.n_states])
        self.habit_strength = np.zeros([self.env.n_actions, self.env.n_states])
        self.habitisation_rate = habitisation_rate
        self.reward_history = []
        self.cue_reaction_times = reaction_times
        self.movement_times = movement_times
        self.dwell_time_history = []
        self.rho = 0
        self.r_k = []
        self.k = 0
        self.dwell_timer = movement_times[0] #np.random.geometric(0.35)*3
        self.psi = psi
        self.num_movements = 1
        self.num_cues = 0

    def get_dwell_time(self, state):
        if state == 'High' or state == 'Low':
            dwell_time = self.cue_reaction_times[self.num_cues]
            self.num_cues +=  1
        else:
            dwell_time = self.movement_times[self.num_movements]
            self.num_movements += 1
        return dwell_time

    def compute_value(self, features):
        return np.dot(self.critic_weights, features)

    def compute_action_value(self, features, action_idx):
        return np.dot(features, self.actor_weights[action_idx])

    def compute_habit_prediction_error(self, action, state_num):
        action_idx = self.env.action_idx[action]
        if action == 'Idle':
            delta_a = np.zeros([self.env.n_actions])  # means action 'Idle' does not cause APEs
        elif action == 'Centre':
            delta_a = np.zeros([self.env.n_actions])
        else:
            action_vector = np.eye(self.env.n_actions)[action_idx]
            delta_a = action_vector - self.habit_strength[:, state_num]
        if np.any(delta_a < 0):

            delta_a[np.where(delta_a < 0)] = 0
        return delta_a

    def compute_average_reward_per_timestep(self, n=500):
        if self.k == 0:
            rho_k = 0
        else:
            rho_k = sum(self.r_k[-n:])/ sum(self.dwell_time_history[-n:])
        return rho_k

    def compute_average_reward_per_timestep_fast(self):
        rho_k = (self.rho * self.env.timer + self.reward_history[-1]) / (self.env.timer + 1)
        return rho_k

    def one_trial(self):
        k = 0
        t = 0
        prediction_errors = []
        apes = []
        tone = None
        a = None
        actions = []
        states = []
        rectified_prediction_errors = []
        state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'action taken'])
        while not self.env.in_terminal_state() and t < 1000:
            current_state_num = self.env.state_idx[self.env.current_state]
            current_state = self.env.current_state
            dwell_time = self.env.time_in_state[current_state_num]
            policy = self.softmax(self.actor_value[:, current_state_num])
            a = self.choose_action(policy, dwell_time)
            next_state, reward = self.env.act(a, dwell_time < self.dwell_timer)
            next_state_num = self.env.state_idx[next_state]

            rho2 = 0
            delta_k = 0
            rectified_delta_k = 0
            delta_action = self.compute_habit_prediction_error(a, current_state_num)

            if current_state != next_state:  # only updates value at state transitions
                self.r_k.append(reward)
                rho2 = self.compute_average_reward_per_timestep()
                delta_k = reward - rho2 * dwell_time + self.critic_value[next_state_num] - self.critic_value[current_state_num]
                rectified_delta_k = rectify(delta_k + self.psi)
                self.k += 1
                self.dwell_timer = self.get_dwell_time(next_state)
                self.critic_value[current_state_num] += self.critic_lr * delta_k
                self.actor_value[self.env.action_idx[a], current_state_num] += self.actor_lr * delta_k
                self.habit_strength[:, current_state_num] += self.habitisation_rate * delta_action
                k += 1  # transition index increases
                new_state_changes = pd.DataFrame([[next_state, self.env.timer, dwell_time, a]], columns=['state name', 'time stamp', 'dwell time', 'action taken'])
                state_changes = state_changes.append(new_state_changes)
                self.dwell_time_history.append(dwell_time)



            if next_state == 'High':
                tone = 'High'
            elif next_state == 'Low':
                tone = 'Low'
            prediction_errors.append(delta_k)
            rectified_prediction_errors.append(rectified_delta_k)
            apes.append(delta_action[0])
            actions.append(a)
            states.append(self.env.current_state)
            self.reward_history.append(reward)
            t += 1
            self.env.timer += 1
        return prediction_errors,rectified_prediction_errors, tone, actions, states, state_changes, apes

    def choose_action(self, policy, dwell_time, random_policy=False, optimal_policy=False):
        if dwell_time < self.dwell_timer:
            a = 'Idle'
        elif self.env.current_state == 'HighLeft' or self.env.current_state == 'LowLeft':
                a = 'Idle'
        elif self.env.current_state == 'HighRight' or self.env.current_state == 'LowRight':
                a = 'Idle'
        else:
            if random_policy:
                a = np.random.choice(self.env.actions)
            elif optimal_policy:
                if self.env.current_state == 'High':
                    a = 'Left'
                elif self.env.current_state == 'Low':
                    a = 'Right'
                else:
                    a = 'Centre'
            else:
                a = np.random.choice(self.env.actions, p=policy)

        return a

    def softmax(self, state_action_values):
        exps = np.exp(self.inv_temp * state_action_values)
        policy = [e / sum(exps) for e in exps]
        return policy


def align_PEs(PEs, trial_types, choices):
    trial_lengths = [len(trial) for trial in PEs]
    max_trial_length = max(trial_lengths)
    cue_aligned_PEs = np.full([len(PEs), max_trial_length], np.nan)
    reward_aligned_PEs = np.full([len(PEs), max_trial_length], np.nan)
    for trial in range(0, len(PEs)):
        trial_PE = PEs[trial]
        trial_length = len(trial_PE)
        cue_aligned_PEs[trial, :trial_length] = trial_PE
        reward_aligned_PEs[trial, -trial_length:] = trial_PE
    cue_aligned_df = pd.DataFrame(cue_aligned_PEs)
    cue_aligned_df['Type'] = trial_types
    cue_aligned_df['Choice'] = choices
    reward_aligned_df = pd.DataFrame(reward_aligned_PEs)
    reward_aligned_df['Type'] = trial_types
    reward_aligned_df['Choice'] = choices
    return cue_aligned_df, reward_aligned_df

def plot_heat_maps_over_trials(PEs, time_stamps,ax, title, window=10, delta_range=[-1, 1]):
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    im = ax.imshow(aligned_PEs, extent=[-(window/2), (window/2), trial_num, 0], aspect='auto',vmin=delta_range[0], vmax=delta_range[1])
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Trial number')
    ax.set_title(title)
    return


def plot_early_and_late(PEs, time_stamps,ax, title, window=10, chunk_prop=.33):
    colours = cm.viridis(np.linspace(0, 0.8, 3))
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    early_aligned_PEs = aligned_PEs[:int(aligned_PEs.shape[0] * chunk_prop)].mean(axis=0)
    mid_aligned_PEs = aligned_PEs[int(aligned_PEs.shape[0] * chunk_prop):-int(aligned_PEs.shape[0] * chunk_prop)].mean(axis=0)
    late_aligned_PEs = aligned_PEs[-int(aligned_PEs.shape[0] * chunk_prop):].mean(axis=0)
    timesteps = np.arange(-(window/2), (window/2))
    ax.plot(timesteps, early_aligned_PEs, color=colours[0], label= 'early')
    ax.plot(timesteps, mid_aligned_PEs, color=colours[1], label= 'mid')
    ax.plot(timesteps, late_aligned_PEs, color=colours[2], label= 'late')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Response')
    ax.set_title(title)
    return


def plot_change_over_time(PEs, time_stamps,ax, title):
    PEs_peak = np.zeros([len(time_stamps)-3])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        PEs_peak[trial_num] = PEs[time_stamp]
    rolling_av_peaks = moving_average(PEs_peak, n=50)
    ax.plot(rolling_av_peaks, color='#3F888F')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Response size')
    ax.set_title(title)
    return


def rectify(num_to_rectify):
    if num_to_rectify < 0:
        return 0
    else:
        return num_to_rectify

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':

    n_trials = 2000
    x = np.linspace(0,50, n_trials  * 10)
    cue_reaction_times = np.random.geometric(0.01, x.shape[0]) #(np.exp(-x)*3+ np.random.rand(x.shape[0])) + 5
    movement_times = np.random.geometric(0.01, x.shape[0]) * 3
    e = Box(punish=True)
    a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.01, psi=0.1)

    all_PEs = []
    all_APEs = []
    all_trial_types = []
    all_actions = []
    all_states = []
    all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])
    for i in tqdm(range(n_trials)):
        _, PEs, trial_type, action, states, state_changes, apes = a.one_trial()
        e.reset()
        all_states.append(states)
        all_PEs.append(PEs)
        all_trial_types.append(trial_type)
        all_actions.append(action)
        all_state_changes = all_state_changes.append(state_changes, ignore_index=True)
        all_APEs.append(apes)

    continuous_time_PEs = np.concatenate(all_PEs).ravel()
    continuous_time_APEs = np.concatenate(all_APEs).ravel()
    rewarded_trials = np.where(np.asarray(a.reward_history) == 1)[0]

    low_states = all_state_changes[all_state_changes['state name'] == 'Low']
    #short_low_states = low_states.loc[low_states['dwell time'] > np.mean(dwell_times)]
    low_tone_times = all_state_changes['time stamp'][all_state_changes[all_state_changes['state name']=='Low'].index.values].values
    high_tone_times = all_state_changes['time stamp'][all_state_changes[all_state_changes['state name']=='High'].index.values].values
    left_choices = all_state_changes['time stamp'][all_state_changes[all_state_changes['action taken']=='Left'].index.values].values
    right_choices = all_state_changes['time stamp'][all_state_changes[all_state_changes['action taken']=='Right'].index.values].values

    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    min_PE = min(np.concatenate(all_PEs).ravel())
    max_PE = max(np.concatenate(all_PEs).ravel())
    reward_plot = plot_heat_maps_over_trials(continuous_time_PEs, rewarded_trials, axs[0], 'reward', window=6, delta_range=[min_PE, max_PE])
    low_plot = plot_heat_maps_over_trials(continuous_time_PEs, low_tone_times, axs[1], 'low cues', window=6, delta_range=[min_PE, max_PE])
    high_plot = plot_heat_maps_over_trials(continuous_time_PEs, high_tone_times, axs[2], 'high cues', window=6, delta_range=[min_PE, max_PE])
    plt.tight_layout()

    fig1, axs1 = plt.subplots(nrows=1, ncols=3)
    min_PE = min(np.concatenate(all_PEs).ravel())
    max_PE = max(np.concatenate(all_PEs).ravel())
    reward_plot = plot_early_and_late(continuous_time_PEs, rewarded_trials, axs1[0], 'reward', window=6)
    low_plot = plot_early_and_late(continuous_time_PEs, low_tone_times, axs1[1], 'low cues', window=6)
    high_plot = plot_early_and_late(continuous_time_PEs, high_tone_times, axs1[2], 'high cues', window=6)
    axs1[2].legend(bbox_to_anchor=(1., .8, .15, .2), loc='upper left')
    plt.tight_layout()



    for ax in axs1:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig2, axs2 = plt.subplots(nrows=1, ncols=2)
    min_APE = min(np.concatenate(all_APEs).ravel())
    max_APE = max(np.concatenate(all_APEs).ravel())
    left_plot = plot_heat_maps_over_trials(continuous_time_APEs, left_choices, axs2[0], 'contra', window=6, delta_range=[min_APE, max_APE])
    low_plot = plot_heat_maps_over_trials(continuous_time_APEs, right_choices, axs2[1], 'ipsi', window=6, delta_range=[min_APE, max_APE])

    plt.tight_layout()


    fig3, axs3 = plt.subplots(nrows=1, ncols=2, sharey=True)
    min_APE = min(np.concatenate(all_APEs).ravel())
    max_APE = max(np.concatenate(all_APEs).ravel())
    left_plot = plot_early_and_late(continuous_time_APEs, left_choices, axs3[0], 'contra', window=6)
    low_plot = plot_early_and_late(continuous_time_APEs, right_choices, axs3[1], 'ipsi', window=6)
    axs3[1].legend(bbox_to_anchor=(1., .8, .15, .2), loc='upper left')

    for ax in axs3:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig4, axs4 = plt.subplots(nrows=1, ncols=3)
    reward_plot = plot_change_over_time(continuous_time_PEs, rewarded_trials, axs4[0], 'reward')
    low_plot = plot_change_over_time(continuous_time_PEs, low_tone_times, axs4[1], 'low cues')
    high_plot = plot_change_over_time(continuous_time_PEs, high_tone_times, axs4[2], 'high cues')
    plt.tight_layout()

    for ax in axs4:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig5, axs5 = plt.subplots(nrows=1, ncols=2, sharey=True)
    left_plot = plot_change_over_time(continuous_time_APEs, left_choices, axs5[0], 'contra')
    low_plot = plot_change_over_time(continuous_time_APEs, right_choices, axs5[1], 'ipsi')

    for ax in axs5:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)



    plt.tight_layout()
    plt.show()
