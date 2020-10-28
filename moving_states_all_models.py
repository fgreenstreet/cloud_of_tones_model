import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pdb
from tqdm import tqdm, trange

np.random.seed(0)


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
    def __init__(self, reaction_times, movement_times, env=Box(), critic_learning_rate=.2, actor_learning_rate=.1, habitisation_rate=.1, inv_temp=5., psi=0.1):
        self.env = env
        self.inv_temp = inv_temp  # inverse temperature param for softmax decision func
        self.critic_lr = critic_learning_rate
        self.critic_value = np.zeros(self.env.n_states)  # weights parameterising value function
        self.actor_lr = actor_learning_rate
        self.actor_value = np.zeros([self.env.n_actions, self.env.n_states])
        self.habit_strength = np.zeros([self.env.n_actions, self.env.n_states])
        self.habitisation_rate = habitisation_rate
        self.reward_history = []
        self.dwell_time_history = []
        self.k = 0
        self.dwell_timer = movement_times[0]
        self.t_per_state = np.zeros(self.env.n_states)
        self.instances_in_state = np.zeros(self.env.n_states)
        self.saliences = np.zeros(self.env.n_states)
        self.cue_reaction_times = reaction_times
        self.movement_times = movement_times
        self.rho = 0
        self.r_k = []
        self.psi = psi
        self.num_cues = 0
        self.num_movements = 1


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
        return np.dot(features, self.actor_weights[action_idx])  # is this right?

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

    def compute_movement_signal(self, action, *args):
        """fire in response to movement"""
        action_idx = self.env.action_idx[action]
        if action == 'Idle':
            signal = np.zeros([self.env.n_actions])  # no movement
        elif action == 'Centre':
            signal = np.zeros([self.env.n_actions])  # no movement
        else:
            signal = np.eye(self.env.n_actions)[action_idx]

        return signal

    def compute_novelty(self, gamma=0.01):  # .33 if reset every trial, 0.01 if maintain across
        # just a decaying exponential of time spent in a state
        return np.exp(-gamma * self.instances_in_state)

    def compute_salience(self, value, novelty, state_idx, beta=0.5):
        # TODO: novelty + rpe
        # return beta * value + (1 - beta) * novelty
        return value[state_idx] / beta + novelty[state_idx]

    def compute_average_reward_per_timestep(self, n=500):
        if self.k == 0:
            rho_k = 0
        else:
            rho_k = sum(self.r_k[-n:]) / sum(self.dwell_time_history[-n:])
        return rho_k

    def one_trial(self):
        k = 0
        t = 0
        rectified_prediction_errors, prediction_errors, apes, actions, states, m_signals, novelties, salience_hist, values = [], [], [], [], [], [], [], [], []
        tone = None
        a = None
        state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])
        total_reward = 0.
        self.instances_in_state[0] = 1
        novelty = self.compute_novelty()
        self.saliences[0] = self.compute_salience(self.critic_value, novelty, 0)
        while not self.env.in_terminal_state() and t < 1000:
            current_state_num = self.env.state_idx[self.env.current_state]
            self.t_per_state[current_state_num] += 1
            current_state = self.env.current_state
            dwell_time = self.env.time_in_state[current_state_num]
            policy = self.softmax(self.actor_value[:, current_state_num])
            a = self.choose_action(policy, dwell_time)
            next_state, reward = self.env.act(a, dwell_time < self.dwell_timer)
            next_state_num = self.env.state_idx[next_state]
            rho2 = 0
            delta_k = 0
            rectified_delta_k = 0
            ################################################################
            movement_signal = self.compute_movement_signal(a)
            novelty = np.zeros(self.env.n_states)
            self.saliences = np.zeros(self.env.n_states)

            delta_action = np.zeros([self.env.n_actions])


            ################################################################
            if current_state != next_state:  # only updates value at state transitions
                novelty = self.compute_novelty()
                self.saliences[next_state_num] = self.compute_salience(self.critic_value, novelty, next_state_num)
                self.instances_in_state[next_state_num] += 1
                self.r_k.append(reward)
                rho2 = self.compute_average_reward_per_timestep()
                delta_k = reward - rho2 * dwell_time + self.critic_value[next_state_num] - self.critic_value[current_state_num]
                rectified_delta_k = rectify(delta_k + self.psi)
                self.k += 1
                self.dwell_timer = self.get_dwell_time(next_state)
                self.critic_value[current_state_num] += self.critic_lr * delta_k
                self.actor_value[self.env.action_idx[a], current_state_num] += self.actor_lr * delta_k
                delta_action = self.compute_habit_prediction_error(a, current_state_num)
                self.habit_strength[:, current_state_num] += self.habitisation_rate * delta_action
                k += 1  # transition index increases
                new_state_changes = pd.DataFrame([[next_state, self.env.timer, dwell_time, a]], columns=['state name', 'time stamp', 'dwell time', 'action taken'])
                state_changes = state_changes.append(new_state_changes)
                self.dwell_time_history.append(dwell_time)


            if next_state == 'High':
                tone = 'High'
            elif next_state == 'Low':
                tone = 'Low'
            self.last_action = a
            prediction_errors.append(delta_k)
            rectified_prediction_errors.append(rectified_delta_k)
            apes.append(delta_action[0])
            actions.append(a)
            values.append(self.critic_value.reshape(-1, 1))
            m_signals.append(movement_signal.reshape(-1, 1))
            novelties.append(novelty.reshape(-1, 1))
            salience_hist.append(self.saliences.reshape(-1, 1))
            states.append(self.env.current_state)
            total_reward += reward
            self.reward_history.append(reward)
            t += 1
            self.env.timer += 1
        return prediction_errors, rectified_prediction_errors, tone, actions, states, state_changes, apes, \
               total_reward, m_signals, values, novelties, salience_hist  # novelties, saliences is 4 x trial_len

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


def plot_heat_maps_over_trials(PEs, time_stamps, ax, title, window=10, delta_range=[-1, 1]):
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    im = ax.imshow(aligned_PEs, extent=[-(window / 2), (window / 2), trial_num, 0], aspect='auto', vmin=delta_range[0],
                   vmax=delta_range[1])
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Trial number')
    ax.set_title(title)
    return


def plot_early_and_late(PEs, time_stamps, ax, max_val, window=10, chunk_prop=.33):
    colours = cm.viridis(np.linspace(0, 0.8, 3))
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    early_aligned_PEs = aligned_PEs[:int(aligned_PEs.shape[0] * chunk_prop)].mean(axis=0)
    mid_aligned_PEs = aligned_PEs[int(aligned_PEs.shape[0] * chunk_prop):-int(aligned_PEs.shape[0] * chunk_prop)].mean(
        axis=0)
    late_aligned_PEs = aligned_PEs[-int(aligned_PEs.shape[0] * chunk_prop):].mean(axis=0)
    timesteps = np.arange(-(window / 2), (window / 2))
    ax.plot(timesteps, early_aligned_PEs, color=colours[0], label='early')
    ax.plot(timesteps, mid_aligned_PEs, color=colours[1], label='mid')
    ax.plot(timesteps, late_aligned_PEs, color=colours[2], label='late')
    ax.set_ylim([0, 1])
    #ax.set_xlabel('Time steps')
   # ax.set_ylabel('Response')

    return


def plot_change_over_time(PEs, stamps,ax):
    PEs_peak = np.zeros([len(stamps)-3])
    for trial_num, time_stamp in enumerate(stamps[1:-2]):
        PEs_peak[trial_num] = PEs[time_stamp]
    rolling_av_peaks = moving_average(PEs_peak, n=50)
    ax.plot(rolling_av_peaks, color='#3F888F')
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
            _, PEs, trial_type, action, states, state_changes, apes, trial_r, m_signals, values, novelties, saliences = a.one_trial()
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
low_tone_times = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['state name'] == 'Low'].index.values].values
high_tone_times = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['state name'] == 'High'].index.values].values
left_choices = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['action taken'] == 'Left'].index.values].values
right_choices = all_state_changes['time stamp'][
    all_state_changes[all_state_changes['action taken'] == 'Right'].index.values].values


# novelties, values, saliences
fig, axs = plt.subplots(5, 4, figsize=[10, 8])
states = ['High tones', 'Low tones', 'Reward', 'Contra', 'Ipsi']
time_stamps = {'High tones': high_tone_times, 'Low tones': low_tone_times, 'Reward': reward_times,
               'Contra': left_choices, 'Ipsi': right_choices}
models = {'APE': continuous_time_APEs, 'RPE': continuous_time_PEs, 'Novelty': continuous_time_Ns,
          'Salience': continuous_time_Ss, 'Movement': continuous_time_MSs}

axs[0, 0].set_ylabel('RPE')
axs[1, 0].set_ylabel('Salience')
axs[2, 0].set_ylabel('Novelty')
axs[3, 0].set_ylabel('Movement')
axs[4, 0].set_ylabel('APE')

axs[0, 0].set_title('High tones')
axs[0, 1].set_title('Low tones')
axs[0, 2].set_title('Reward')

plot_early_and_late(models['RPE'],  time_stamps['High tones'], axs[0, 0], ' ', window=6)
plot_early_and_late(models['RPE'],  time_stamps['Low tones'], axs[0, 1], ' ', window=6)
plot_early_and_late(models['RPE'],  time_stamps['Reward'], axs[0, 2], ' ', window=6)
plot_change_over_time(models['RPE'], time_stamps['High tones'], axs[0, 3])


plot_early_and_late(models['Salience'][:, 1],  time_stamps['High tones'], axs[1, 0], ' ', window=6)
plot_early_and_late(models['Salience'][:, 2],  time_stamps['Low tones'], axs[1, 1], ' ', window=6)
plot_early_and_late(models['Salience'][:, 3],  time_stamps['Reward'], axs[1, 2], ' ', window=6)
plot_change_over_time(models['Salience'][:, 1], time_stamps['High tones'], axs[1, 3])


plot_early_and_late(models['Novelty'][:, 1],  time_stamps['High tones'], axs[2, 0], ' ', window=6)
plot_early_and_late(models['Novelty'][:, 2],  time_stamps['Low tones'], axs[2, 1], ' ', window=6)
plot_early_and_late(models['Novelty'][:, 3],  time_stamps['Reward'], axs[2, 2], ' ', window=6)
plot_change_over_time(models['Novelty'][:, 1], time_stamps['High tones'], axs[2, 3])

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