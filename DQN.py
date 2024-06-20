import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nn import initialize_model, train_with_loss
from env import *


def compute_returns_and_advantages(values, rewards, gamma=0.95):
    returns = []
    advantages = []
    G = 0
    A = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        returns.insert(0, G)
        delta = rewards[i] + gamma * values[i + 1] - values[i] if i + 1 < len(values) else rewards[i] - values[i]
        A = delta + gamma * A
        advantages.insert(0, A)
    advantages = np.array(advantages)
    returns = np.array(returns)
    return advantages, returns


value_network = initialize_model(in_features=len(reset), out_features=1)
policy_network = initialize_model(in_features=len(reset), out_features=len(actions))

value_optimizer = torch.optim.Adam(value_network.parameters())
policy_optimizer = torch.optim.Adam(policy_network.parameters())

# Implementing a learning rate scheduler
#value_scheduler = torch.optim.lr_scheduler.StepLR(value_optimizer, step_size=100, gamma=0.9)
#policy_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=100, gamma=0.9)

def pickAction(ep, actions, plot=False):
    replay = []
    state = reset
    pressure = state[1]
    rewardTotal = 0
    flows = []
    for i in range(ep):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_network(state_tensor, policy=True)
        action_index = torch.multinomial(action_probs, num_samples=1).item()  # Sample action index
        action = actions[action_index]  # Get the actual action value
        pressure = get_pressure(pressure, action)
        flow = get_flow(pressure)
        state = new_state(action, pressure, flow)
        if plot:
            flows.append(flow)
        reward = get_reward(flow)
        rewardTotal += reward

        replay.append((state, action_index, reward, action_probs[0, action_index]))

    return rewardTotal, replay, flows


def sampleReplay(replay, batch_size):
    indices = np.random.choice(len(replay), batch_size, replace=False)
    sampled_replay = [replay[i] for i in indices]
    return sampled_replay


def train(values, advantages, returns, states, action_indices, action_probs):
    values = torch.tensor(values, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    states = torch.tensor(states, dtype=torch.float32)
    action_indices = torch.tensor(action_indices, dtype=torch.int64)

    action_probs = torch.stack(action_probs)

    if len(states.shape) == 1:
        states = states.unsqueeze(0)
    if len(values.shape) == 1:
        values = values.unsqueeze(0)
    if len(advantages.shape) == 1:
        advantages = advantages.unsqueeze(0)
    if len(returns.shape) == 1:
        returns = returns.unsqueeze(0)

    if action_probs.dim() != 1 or action_probs.shape[0] != action_indices.shape[0]:
        raise ValueError("action_probs must be a 1D tensor with the same length as action_indices")

    value_predictions = value_network(states)
    
    if value_predictions.shape != returns.shape:
        value_predictions = value_predictions.view_as(returns)

    value_loss = F.mse_loss(value_predictions, returns)
    log_probs = torch.log(action_probs)
    policy_loss = -(log_probs * advantages).mean()

    value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
    policy_optimizer.step()

    return value_loss.item(), policy_loss.item()


def reinforcementLearner(actions, N, ep, plot):
    rewardTotal, replay, flows = pickAction(ep, actions, plot)
    replay = sampleReplay(replay, N)
    states, indices, rewards, probs = zip(*replay)
    states = np.stack(states)
    indices = np.array(indices)
    rewards = np.array(rewards)
    probs = list(probs)
    values = value_network(torch.FloatTensor(states))
    advantages, returns = compute_returns_and_advantages(values.detach().numpy(), rewards)
    train(values, advantages, returns, states, indices, probs)

    # Update the learning rate
    #value_scheduler.step()
    #policy_scheduler.step()

    return rewardTotal, flows


# Training example
episodes = 1000
replay_memory = []
total_rewards = []

counter = 0
for episode in range(episodes):
    plot = (episode == episodes - 1)
    reward, flows = reinforcementLearner(actions, 20, 200, plot)
    if plot:
        plt.figure()
        plt.plot(flows)
    total_rewards.append(reward)
    counter += reward
    if episode % 10 == 9:
        print(f"Episode {episode + 1}, Avg Reward: {counter / 10}")
        counter = 0

plt.figure()
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()