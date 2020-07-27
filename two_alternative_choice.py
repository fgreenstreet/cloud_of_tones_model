import numpy as np
import matplotlib.pyplot as plt


class Box(object):
    def __init__(self):
        self.window_size = 20  # number of time steps features stay on
        self.n_features = 1 + self.window_size * 2

        self.high_to_left = True  # toggles whether High tone corresponds to correct action being left
        self.high_sound_prob = .5

        self.states = ['Start', 'High', 'Low', 'Outcome']
        self.n_states = len(self.states)
        self.state_idx = {s: idx for s, idx in zip(self.states, range(self.n_states))}
        self.actions = ['Left', 'Centre', 'Right', 'Idle']
        self.n_actions = len(self.actions)
        self.action_idx = {a: idx for a, idx in zip(self.actions, range(self.n_actions))}

        self.current_state = self.states[0]
        self.time_in_state = np.zeros(self.n_states, dtype=int)

    def act(self, action):
        # maybe make animals not able to act every time step
        next_state = self.get_next_state(self.current_state, action)
        reward = self.get_reward(self.current_state, action, next_state)

        # adjust state timer
        if next_state != self.current_state:
            self.time_in_state = np.zeros(self.n_states)
        else:
            self.time_in_state[self.state_idx[self.current_state]] += 1

        self.current_state = next_state
        return next_state, reward

    def get_next_state(self, state, action):
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
                next_state = state
            else:
                next_state = 'Outcome'

        elif state == 'Outcome':
            next_state = None

        else:
            raise ValueError('No valid state input.')

        return next_state

    def get_reward(self, state, action, next_state):
        if next_state != 'Outcome':
            reward_amount = 0
        else:
            if self.high_to_left:
                if state == 'High' and action == 'Left':
                    reward_amount = 1
                elif state == 'High' and action == 'Right':
                    reward_amount = 0
                elif state == 'Low' and action == 'Right':
                    reward_amount = 1
                elif state == 'Low' and action == 'Left':
                    reward_amount = 0
                else:
                    reward_amount = 0
            else:
                if state == 'High' and action == 'Left':
                    reward_amount = 0
                elif state == 'High' and action == 'Right':
                    reward_amount = 1
                elif state == 'Low' and action == 'Right':
                    reward_amount = 0
                elif state == 'Low' and action == 'Left':
                    reward_amount = 1
                else:
                    reward_amount = 0
            if action == 'Centre':
                reward_amount = 0
        return reward_amount

    def get_features(self, state):
        start_f = 0
        high_f = np.zeros(self.window_size)
        low_f = np.zeros(self.window_size)
        if state == 'Start':
            start_f = 1
        elif state == 'High':
            time_spent_in_state = self.time_in_state[self.state_idx[state]]
            high_f = np.eye(self.window_size)[int(time_spent_in_state)]
        elif state == 'Low':
            time_spent_in_state = self.time_in_state[self.state_idx[state]]
            low_f = np.eye(self.window_size)[int(time_spent_in_state)]
        else:
            return np.zeros(self.n_features)
        return np.concatenate([[start_f], high_f, low_f])

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
    def __init__(self, env=Box(), gamma=.9, critic_learning_rate=.1, actor_learning_rate=.1, habitisation_rate =.1, inv_temp=5.):
        self.env = env
        self.gamma = gamma  # discount factor
        self.inv_temp = inv_temp  # inverse temperature param for softmax decision func
        self.critic_lr = critic_learning_rate
        self.critic_weights = np.zeros(self.env.n_features)  # weights parameterising value function
        self.actor_lr = actor_learning_rate
        self.actor_weights = np.zeros([self.env.n_actions, self.env.n_features])
        self.habit_weights = np.zeros([self.env.n_actions, self.env.n_features])
        self.habitisation_rate = habitisation_rate

    def compute_value(self, features):
        return np.dot(self.critic_weights, features)

    def compute_action_value(self, features, action_idx):
        return np.dot(features, self.actor_weights[action_idx])

    def compute_habit_prediction_error(self, action, features):
        action_idx = self.env.action_idx[action]
        action_vector = np.eye(self.env.n_actions)[action_idx]
        current_habit = np.dot(self.habit_weights, features)
        delta_a = action_vector - current_habit
        return delta_a

    def one_trial(self):
        prediction_errors = []
        left_habit_errors = []
        tone = None
        t = 0
        self.env.reset()
        state = self.env.current_state
        features = self.env.get_features(state)
        while not self.env.in_terminal_state() and t < 1000:
            # choose action
            policy = self.softmax(features)
            a = self.choose_action(t, policy)

            # take action
            next_state, reward = self.env.act(a)
            if next_state == 'High':
                tone = 'High'
            elif next_state == 'Low':
                tone = 'Low'

            next_features = self.env.get_features(next_state)

            delta = reward + self.gamma * np.dot(next_features, self.critic_weights) - np.dot(features, self.critic_weights)
            self.critic_weights += self.critic_lr * delta * features
            self.actor_weights[self.env.action_idx[a]] += self.actor_lr * delta * features
            delta_action = self.compute_habit_prediction_error(a, features)
            self.habit_weights += self.habitisation_rate * np.outer(delta_action, features)

            left_habit_errors.append(delta_action[0])
            prediction_errors.append(delta)

            state = next_state
            features = next_features
            t += 1
        return prediction_errors, tone, reward, left_habit_errors, a

    def choose_action(self, time, policy):
        # a = np.random.choice(self.env.actions)
        if time == 0:
            a = 'Centre'
        elif 0 < time < 10:
            a = 'Idle'
        else:
            a = np.random.choice(self.env.actions, p=policy)
        return a

    def softmax(self, features):
        exps = [np.exp(self.inv_temp * np.dot(self.actor_weights[a], features)) for a in range(self.env.n_actions)]
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


def plot_early_and_late_PE(PEs_df, ax, title, num_trials=10):
    trial_type_PEs = PEs_df
    first_n_trials = trial_type_PEs.iloc[0:num_trials].T
    mean_first_n = np.mean(first_n_trials, axis=1)
    last_n_trials = trial_type_PEs.iloc[-num_trials:].T
    mean_last_n = np.mean(last_n_trials, axis=1)
    ax.plot(mean_first_n, label='first {} trials'.format(num_trials))
    ax.plot(mean_last_n, label='last {} trials'.format(num_trials))
    ax.set_xlabel('time')
    ax.set_ylabel('prediction error size')
    ax.set_title(title)

    plt.legend()


def plot_heat_maps_over_trials(PEs_df, ax, title):
    ax.imshow(PEs_df, aspect='auto')
    ax.set_xlabel('time')
    ax.set_ylabel('trial number')
    ax.set_title(title)

    
def plot_summary(cue_aligned_RPEs, reward_aligned_RPEs, cue_aligned_APEs, reward_aligned_APEs, tone='High', choice='Left'):
    if choice == 'Left':
        laterality = 'Contralateral'
    else:
        laterality = 'Ipsilateral'
    cue_aligned_filtered_RPEs = cue_aligned_RPEs[(cue_aligned_RPEs.Type == tone) & (cue_aligned_RPEs.Choice == choice)].drop(
        ['Type', 'Choice'], axis=1)
    cue_aligned_filtered_APEs = cue_aligned_APEs[(cue_aligned_APEs.Type == tone) & (cue_aligned_APEs.Choice == choice)].drop(
        ['Type', 'Choice'], axis=1)
    reward_aligned_filtered_RPEs = reward_aligned_RPEs[
        (cue_aligned_RPEs.Type == tone) & (reward_aligned_RPEs.Choice == choice)].drop(
        ['Type', 'Choice'], axis=1)
    reward_aligned_filtered_APEs = reward_aligned_APEs[
        (cue_aligned_APEs.Type == tone) & (reward_aligned_APEs.Choice == choice)].drop(
        ['Type', 'Choice'], axis=1)

    line_fig, line_ax = plt.subplots(2, 2)
    plot_early_and_late_PE(cue_aligned_filtered_RPEs, line_ax[0, 0], 'cue-aligned RPE')
    plot_early_and_late_PE(reward_aligned_filtered_RPEs, line_ax[0, 1], 'reward-aligned RPE')
    plot_early_and_late_PE(cue_aligned_filtered_APEs, line_ax[1, 0], 'cue-aligned APE')
    plot_early_and_late_PE(reward_aligned_filtered_APEs, line_ax[1, 1], 'reward-aligned APE')
    line_fig.suptitle('{side} choice, {sound} tones'.format(side=laterality, sound=tone))
    plt.tight_layout()

    fig, ax = plt.subplots(2, 2, squeeze=True)
    plot_heat_maps_over_trials(cue_aligned_filtered_RPEs, ax[0, 0], 'cue-aligned RPE')
    plot_heat_maps_over_trials(reward_aligned_filtered_RPEs, ax[0, 1], 'reward-aligned RPE')
    plot_heat_maps_over_trials(cue_aligned_filtered_APEs, ax[1, 0], 'cue-aligned APE')
    plot_heat_maps_over_trials(reward_aligned_filtered_APEs, ax[1, 1], 'reward-aligned APE')
    fig.suptitle('{side} choice, {sound} tones'.format(side=laterality, sound=tone))
    plt.tight_layout()


if __name__ == '__main__':
    import pandas as pd

    n_trials = 300

    e = Box()
    e.get_features('High')
    a = Mouse(env=e, critic_learning_rate=0.05, actor_learning_rate=0.05, gamma=0.99)

    all_PEs = []
    all_APEs = []
    all_trial_types = []
    all_rewards = []
    all_actions = []

    for i in range(n_trials):
        high_features = a.env.get_features('High')
        PEs, trial_type, rewards, APEs, action = a.one_trial()
        all_PEs.append(PEs)
        all_APEs.append(APEs)
        all_trial_types.append(trial_type)
        all_rewards.append(rewards)
        all_actions.append(action)

    cue_aligned_RPEs, reward_aligned_RPEs = align_PEs(all_PEs, all_trial_types, all_actions)
    cue_aligned_APEs, reward_aligned_APEs = align_PEs(all_APEs, all_trial_types, all_actions)

    plot_summary(cue_aligned_RPEs, reward_aligned_RPEs, cue_aligned_APEs, reward_aligned_APEs, choice='Right', tone='Low')
    plt.show()

