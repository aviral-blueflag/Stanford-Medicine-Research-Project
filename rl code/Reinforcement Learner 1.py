import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import random
import Network

'''
env = gym.make('Blackjack-v1', natural=False, sab=False)
actions = env.action_space.
spaces = env.observation_space.n

q = np.zeros((spaces, actions))
print(q)  
'''

#env
#states s_t = [Q (current flow rate), dP \ dz (pressure gradient), previous pressure setting, error (from eq), noise (unsure)]
#actions = [increase pressure,decrease pressure,maintain pressure]

def reinforcementLearner(q_table, eps, alpha, ep, start):
    done = False
    state = start
    rewardTotal = 0
    for i in range(ep):
        if (np.random.uniform(0,1) < eps):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        new_state, reward, done = env.step(action)[:3]
        #print("state", state, "action", action, "reward", reward)
        q_table[state][action]= q_table[state][action] * (1 - alpha) + alpha * (reward + 0.9 * np.max(q_table[new_state]))

        state = new_state
        rewardTotal += reward

        if done: 
            #print("TOTAL REWRD", rewardTotal)
            #print("DONE")
            break
    #print(q_table)

    return rewardTotal, q_table

eps = 1
total = 0
incremental = 0
for i in range(10000):
    reward, q = reinforcementLearner(q, eps, 0.9, 100, env.reset()[0])
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