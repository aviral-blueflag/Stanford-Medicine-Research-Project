import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nn import initialize_model, train_with_loss

mu, L, r = 1, 1, 1
desired = 5
noise = 0.1

# Environment
actions = [desired / 2 + i * desired / 10 for i in range(10)]

def get_pressure(prevPressure, setP, deltaT):
    return prevPressure + (setP - prevPressure) * (1 - np.exp(-deltaT / 2))

def get_flow(pressure):
    return pressure * np.pi * r**4 / (8 * mu * L) * np.random.uniform(1 - noise, 1 + noise)

def get_reward(flow):
    return -(desired - flow) ** 2

def new_state(state, action, pressure, t):
    flow = get_flow(pressure)
    return [flow, get_pressure(pressure, action, t), action, desired - flow, state[4]]

reset = [0, 0, 0, 0, desired]

def compute_returns_and_advantages(values, episode_transitions, gamma=0.99, lam=0.95):
    rewards = [trans[2] for trans in episode_transitions]
    returns = []
    advantages = []
    G = 0
    A = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        returns.insert(0, G)
        delta = rewards[i] + gamma * values[i + 1] - values[i] if i + 1 < len(values) else rewards[i] - values[i]
        A = delta + gamma * lam * A
        advantages.insert(0, A)
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

value_network = initialize_model(in_features=len(reset), out_features=1)
policy_network = initialize_model(in_features=len(reset), out_features=len(actions))

value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.01)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)

def reinforcementLearner(eps, alpha, ep, start, replay, r, N):
    done = False
    state = reset
    pressure = state[1]
    rewardTotal = 0
    
    for i in range(ep):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_network(state_tensor, policy=True)
        action_index = torch.multinomial(action_probs, num_samples=1).item()  # Sample action index
        action = actions[action_index]  # Get the actual action value

        state = new_state(state, action, pressure, 0.01)
        pressure = get_pressure(pressure, action, 0.01)
        flow = get_flow(pressure)
        reward = get_reward(flow)
        rewardTotal += reward

        replay.append([state, action_index, reward, action_probs])  # Store action_index instead of action

        if done:
            break

    values = value_network(torch.FloatTensor([trans[0] for trans in replay]))
    policy_vals = policy_network(torch.FloatTensor([trans[0] for trans in replay]), policy=True)

    advantages, returns = compute_returns_and_advantages(values.detach().numpy(), replay)

    for i in range(N):
        n = np.random.randint(0, len(replay))
        state, action_index, reward, actions_prob = replay[n]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        advantage = advantages[n]
        G = returns[n]

        value_estimate = value_network(state_tensor)
        value_loss = F.mse_loss(value_estimate, torch.tensor([[G]], dtype=torch.float32))

        policy_prob = policy_network(state_tensor, policy=True)
        policy_loss = -torch.log(policy_prob[0, action_index]) * advantage

        train_with_loss(value_network, value_loss, value_optimizer, lr=0.01)
        train_with_loss(policy_network, policy_loss, policy_optimizer, lr=0.01)

    return rewardTotal

# Training example
episodes = 1000
replay_memory = []
total_rewards = []

for episode in range(episodes):
    reward = reinforcementLearner(1.0, 0.99, 100, reset, replay_memory, 0.95, 20)
    total_rewards.append(reward)
    print(f"Episode {episode + 1}, Total Reward: {reward}")

plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
