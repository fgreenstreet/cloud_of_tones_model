import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from agent import AgentWithTimeOuts
from classic_task import TaskWithTimeOuts


# Start by sampling some reaction and movement times
n_trials = 2000  # have experimented with 10000
x = np.linspace(0, 50, n_trials * 10)
cue_reaction_times = np.random.geometric(0.01, x.shape[0])  # (np.exp(-x)*3+ np.random.rand(x.shape[0])) + 5
movement_times = np.random.geometric(0.01, x.shape[0]) * 2

e = TaskWithTimeOuts()
a = AgentWithTimeOuts(cue_reaction_times, movement_times, e,
                      critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.01, psi=0.2)

a.one_trial(1)
a.one_trial(1)