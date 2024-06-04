import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import random
import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn



mu, L, r= 1,1, 1
desired = 5
noise = 0.1

#env
#states s_t = [Q (current flow rate), dP \ dz (pressure gradient), previous pressure setting, error (from eq), noise (unsure)]
actions = [desired/2 + i*desired/50 for i in range(100)]


def get_pressure(prevPressure, setP, deltaT):
    return prevPressure + (setP - prevPressure)*(1 - np.e^(-deltaT/2))

def get_flow(pressure):
    return pressure * np.pi * r**4 / (8 * mu * L) * np.random.uniform(1 - noise, 1 + noise)

def get_reward(flow):
    return -(desired - flow)**2

def new_state(state, action, pressure, t):
    flow = get_flow(pressure)
    return [flow, get_pressure(pressure, action, t), action, desired - flow]

def reset():
    return [0,0,0,0]

def reinforcementLearner(eps, alpha, ep, start, replay, r, N):
    value_network = nn.initialize_model(in_features=len(start), out_features=len(actions))
    policy_network = nn.initialize_model(in_features=len(start), out_features=1)
    done = False
    state = reset()
    pressure = state[1]
    rewardTotal = 0
    for i in range(ep):
        
        action_probs = policy_network(state)
        if (np.random.uniform(0,1) < eps):
            action = actions[np.random.randint(0, len(actions))]
        else:
            action = actions[torch.multinomial(action_probs, num_samples=1)]
        print(f'Action probabilities: {action_probs}')

        # Sample an action based on the probabilities
        
        new_state = new_state(state, action, pressure, 0.01)
        pressure = get_pressure(pressure, action, 0.01)
        flow = get_flow(pressure)
        reward = get_reward(flow)
        rewardTotal += reward

        replay.append([state, action, reward, action_probs])

        if done: 
            #print("TOTAL REWRD", rewardTotal)
            #print("DONE")
            break
    for i in range(N):
    #print(q_table)

    return rewardTotal

reset = [0,0,0,0,0]
eps = 1
total = 0
incremental = 0
for i in range(10000):
    reward, q = reinforcementLearner(q, eps, 0.9, 100, reset)
    if (reward > 0):
        eps*=0.9
        total+=reward
        incremental+= reward
    if (i % 1000 == 0):
        print(incremental/1000)
        incremental = 0
    plt.plot(i, total/(i+1), 'ro')
print(q)
plt.title('Plot of points')
plt.show()





        



    




'''
a1 = 2
a2 = 3
a3 = 7
value = 12
def func1(b):
    if (b < a1 + 11): return b - value
    return value - b

def func2(b):
    if (b < a2 + 11): return b - value
    return value - b

def func3(b):
    if (b < a3 + 11): return b - value
    return value - b
'''