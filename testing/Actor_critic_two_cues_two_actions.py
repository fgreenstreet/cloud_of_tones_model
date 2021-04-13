import numpy as np
import random
import matplotlib.pyplot as plt
import math

beta = 0.0001
num_trials = 600
states = 6
e = 0.5
turns = np.zeros([num_trials])
num_actions = 2
num_cues = 2
value_high, value_low, high_left, high_right, low_left, low_right, probs_high_left, probs_high_right, probs_low_left, probs_low_right = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]

d = 0
v = np.zeros([states])
r = np.zeros([states])
m = np.zeros([num_cues, num_actions])
sound = np.zeros([num_trials])

for trial in range(0, num_trials):
    sound[trial] = random.choice([0, 1])
    if sound[trial] == 0: # high
        noise = np.random.rand(num_actions)/100
        prob_high_left = noise[0] + math.exp(beta * m[0, 0]) / (math.exp(beta * m[0, 1]) + math.exp(beta * m[0, 0]))
        prob_high_right = noise[1] + math.exp(beta * m[0, 1]) / (math.exp(beta * m[0, 1]) + math.exp(beta * m[0, 0]))
        probs_high_left.append(prob_high_left)
        probs_high_right.append(prob_high_right)
        maxP = max([prob_high_left, prob_high_right])
        if prob_high_left == prob_high_right:
            turns[trial] = random.choice([0, 1])
        elif maxP == prob_high_left:
            turns[trial] = 0
        elif maxP == prob_high_right:
            turns[trial] = 1

        if turns[trial] == 0: # left
            r[0] = 0
            d = r[0] + v[2] - v[0]
            v[0] = v[0] + e * d
            m[0, 0] = m[0, 0] + e * (1 - prob_high_left) * d
            m[0, 1] = m[0, 1] + e * (0 - prob_high_right) * d

        if turns[trial] == 1: # right
            r[0] = 5
            d = r[0] + v[3] - v[0]
            v[0] = v[0] + e * d
            m[0, 1] = m[0, 1] + e * (1 - prob_high_right) * d
            m[0, 0] = m[0, 0] + e * (0 - prob_high_left) * d

    if sound[trial] == 1: # low
        noise = np.random.rand(num_actions)/100
        prob_low_left = noise[0] + math.exp(beta * m[1, 0]) / (math.exp(beta * m[1, 1]) + math.exp(beta * m[1, 0]))
        prob_low_right = noise[1] + math.exp(beta * m[1, 1]) / (math.exp(beta * m[1, 1]) + math.exp(beta * m[1, 0]))
        probs_low_left.append(prob_low_left)
        probs_low_right.append(prob_low_right)
        maxP = max([prob_low_left, prob_low_right])
        if prob_low_left == prob_low_right:
            turns[trial] = random.choice([0, 1])
        elif maxP == prob_low_left:
            turns[trial] = 0
        elif maxP == prob_low_right:
            turns[trial] = 1

        if turns[trial] == 0: # left
            r[1] = 5
            d = r[1] + v[4] - v[1]
            v[1] = v[1] + e * d
            m[1, 0] = m[1, 0] + e * (1 - prob_low_left) * d
            m[1, 1] = m[1, 1] + e * (0 - prob_low_right) * d

        if turns[trial] == 1: # right
            r[1] = 0
            d = r[1] + v[5] - v[1]
            v[1] = v[1] + e * d
            m[1, 1] = m[1, 1] + e * (1 - prob_low_right) * d
            m[1, 0] = m[1, 0] + e * (0 - prob_low_left) * d

    value_high.append(v[0])
    value_low.append(v[1])
    high_left.append(m[0, 0])
    high_right.append(m[0, 1])
    low_left.append(m[1, 0])
    low_right.append(m[1, 1])


fig, axs = plt.subplots(1, 3, figsize=(10,6))
axs[0].plot(value_high, 'g')
axs[0].plot(value_low, 'r')
axs[0].set_xlabel('Trial Number')
axs[0].set_ylabel('Cue value')
axs[0].legend(['High', 'Low'])


axs[1].plot(high_left, 'b')
axs[1].plot(high_right, 'g')
axs[1].set_xlabel('Trial Number')
axs[1].set_ylabel('Action value')
axs[1].legend(['Left', 'Right'])
axs[1].set_title('High cue')

axs[2].plot(low_left, 'b')
axs[2].plot(low_right, 'g')
axs[2].set_xlabel('Trial Number')
axs[2].set_ylabel('Action value')
axs[2].legend(['Left', 'Right'])
axs[2].set_title('Low cue')


plt.show()
