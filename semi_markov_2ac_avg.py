import pandas as pd
from semi_markov_2ac_all_models import Box, Mouse
from tqdm import tqdm
import numpy as np

n_trials = 800
n_agents = 100


all_RPE_rewards = []
all_RPE_lowtones = []
all_RPE_hightones = []

all_APE_left = []
all_APE_right = []

for a in tqdm(range(n_agents)):
    e = Box()
    a = Mouse(env=e, critic_learning_rate=0.0025, actor_learning_rate=0.0025, habitisation_rate=0.01)

    all_PEs = []
    all_APEs = []
    all_trial_types = []
    all_actions = []
    all_states = []
    all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'action taken'])
    for i in range(n_trials):
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
    reward_times = np.where(np.asarray(a.reward_history) == 1)[0]
    low_tone_times = all_state_changes['time stamp'][
        all_state_changes[all_state_changes['state name'] == 'Low'].index.values].values
    high_tone_times = all_state_changes['time stamp'][
        all_state_changes[all_state_changes['state name'] == 'High'].index.values].values
    left_choice_times = all_state_changes['time stamp'][
        all_state_changes[all_state_changes['action taken'] == 'Left'].index.values].values
    right_choice_times = all_state_changes['time stamp'][
        all_state_changes[all_state_changes['action taken'] == 'Right'].index.values].values


    RPE_rewards = [continuous_time_PEs[ts] for ts in reward_times]
    RPE_lowtones = [continuous_time_PEs[ts] for ts in low_tone_times]
    RPE_hightones = [continuous_time_PEs[ts] for ts in high_tone_times]

    APE_left = [continuous_time_APEs[ts] for ts in left_choice_times]
    APE_right = [continuous_time_APEs[ts] for ts in right_choice_times]

    all_RPE_rewards.append(RPE_rewards)
    all_RPE_lowtones.append(RPE_lowtones)
    all_RPE_hightones.append(RPE_hightones)

    all_APE_left.append(APE_left)
    all_APE_right.append(APE_right)


def asarray_and_pad_with_nans(listoflist):
    max_length = max([len(l) for l in listoflist])
    arr = np.empty([len(listoflist), max_length])
    arr[:] = np.nan
    for i, l in enumerate(listoflist):
        arr[i, :len(l)] = l
    return arr


all_RPE_rewards = asarray_and_pad_with_nans(all_RPE_rewards)
all_RPE_lowtones = asarray_and_pad_with_nans(all_RPE_lowtones)
all_RPE_hightones = asarray_and_pad_with_nans(all_RPE_hightones)

all_APE_left = asarray_and_pad_with_nans(all_APE_left)
all_APE_right = asarray_and_pad_with_nans(all_APE_right)

np.save('all_RPE_rewards', all_RPE_rewards)
np.save('all_RPE_lowtones', all_RPE_lowtones)
np.save('all_RPE_hightones', all_RPE_hightones)
np.save('all_APE_left', all_APE_left)
np.save('all_APE_right', all_APE_right)