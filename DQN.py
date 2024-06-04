import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import random
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

reset = [0,0,0,0]

def compute_returns_and_advantages(policy, values, episode_transitions, gamma=0.99, lam=0.95):
    rewards = [trans[2] for trans in episode_transitions]
    returns = []
    advantages = []
    G = 0
    A = 0
    expected =0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        expected += np.log(policy[i])*advantages[i]
        delta = rewards[i] + gamma * values[i + 1] - values[i] if i + 1 < len(values) else rewards[i] - values[i]
        A = delta + gamma * lam * A
        

        returns.insert(0, G)
        advantages.insert(0, A)
    expected /= len(rewards)
    return advantages, returns, expected


def reinforcementLearner(eps, alpha, ep, start, replay, r, N):
    value_network = nn.initialize_model(in_features=len(start), out_features=len(actions))
    policy_network = nn.initialize_model(in_features=len(start), out_features=1)
    done = False
    state = reset
    pressure = state[1]
    rewardTotal = 0
    for i in range(ep):
        
        action_probs = policy_network(state)
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
    values = value_network(torch.FloatTensor([trans[0] for trans in replay])).detach().numpy()
    policy_vals = policy_network(torch.FloatTensor([trans[0] for trans in replay])).detach().numpy()

    advantages, returns, expected = compute_returns_and_advantages(policy_vals, values, replay)
    value_loss = 0
    policy_loss = 0
    for i in range(N):
        # Sample a random transition from the replay buffer
        n = np.random.randint(0, len(replay))
        state, action, reward, actions_prob = replay[n]

        # Convert to torch tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        advantage = torch.FloatTensor([advantages[n]])
        return_value = torch.FloatFloatTensor([returns[n]])

        # Compute value loss
        value_pred = value_network(state_tensor)
        value_loss += advantages[n]**2 / N

        # Compute log probabilities for the policy network

        # Compute policy loss for PPO
        '''
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_
        '''

    # Average the accumulated losses

    # Call your separate backpropagation function
    nn.train_with_loss(value_network, value_loss, lr=0.01)
    nn.train_with_loss(policy_network, policy_loss, lr=0.01)


    



    return rewardTotal

reset = [0,0,0,0,0]
eps = 1
total = 0
incremental = 0
replay = []
start = reset
for i in range(10000):
    reward = reinforcementLearner(eps, 0.99, 100, start, replay, 0.95, 20)
    total+= reward
    incremental += reward
    if (i % 1000 == 0):

        print(incremental/1000)
        incremental = 0
    plt.plot(i, total/(i+1), 'ro')
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