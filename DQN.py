import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nn import initialize_model, train_with_loss
from env import *
torch.autograd.set_detect_anomaly(True)


def compute_returns_and_advantages(values, rewards, gamma=0.99, lam=0.95):
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
    advantages = np.array(advantages)
    returns = np.array(returns)
    return advantages, returns

value_network = initialize_model(in_features=len(reset), out_features=1)
policy_network = initialize_model(in_features=len(reset), out_features=len(actions))

value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.01)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)

def pickAction(ep, actions):
    replay = []
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
        

        replay.append((state, action_index, reward, action_probs[0, action_index].item()))


    replay = np.array(replay, dtype=object)
    return rewardTotal, replay

def sample_replays(replay_buffer, n):
    # Convert replay buffer to a NumPy array if it is not already
    if not isinstance(replay_buffer, np.ndarray):
        replay_buffer = np.array(replay_buffer)
    # Randomly sample n indices from the replay buffer
    indices = np.random.choice(len(replay_buffer), size=n, replace=False)
    # Use the indices to select the replays
    sampled_replays = replay_buffer[indices]
    return sampled_replays
def train(values, advantages, returns, states, action_indices, action_probs):
    # Convert NumPy arrays to PyTorch tensors correctly
    values = torch.tensor(values, dtype=torch.float32)  # No need for requires_grad here
    advantages = torch.tensor(advantages, dtype=torch.float32)  # No need for requires_grad here
    returns = torch.tensor(returns, dtype=torch.float32)  # No need for requires_grad here
    states = torch.tensor(states, dtype=torch.float32)  # No need for requires_grad here
    action_indices = torch.tensor(action_indices, dtype=torch.int64)  # No need for requires_grad here
    action_probs = torch.tensor(action_probs, dtype=torch.float32)  # No need for requires_grad here

    # Debug prints to check shapes
    print("Values shape:", values.shape)
    print("Advantages shape:", advantages.shape)
    print("Returns shape:", returns.shape)
    print("States shape:", states.shape)
    print("Action indices shape:", action_indices.shape)
    print("Action probs shape:", action_probs.shape)

    # Ensure all tensors have the correct shape (adding batch dimension if necessary)
    if len(states.shape) == 1:
        states = states.unsqueeze(0)
    if len(values.shape) == 1:
        values = values.unsqueeze(0)
    if len(advantages.shape) == 1:
        advantages = advantages.unsqueeze(0)
    if len(returns.shape) == 1:
        returns = returns.unsqueeze(0)

    # Ensure action_probs is a 1D tensor matching action_indices
    if action_probs.dim() != 1 or action_probs.shape[0] != action_indices.shape[0]:
        raise ValueError("action_probs must be a 1D tensor with the same length as action_indices")

    # Compute value predictions
    value_predictions = value_network(states)
    
    # Ensure value_predictions and returns have the same shape
    if value_predictions.shape != returns.shape:
        value_predictions = value_predictions.view_as(returns)

    # Compute value loss (Mean Squared Error)
    value_loss = F.mse_loss(value_predictions, returns)

    # Compute log probabilities of the selected actions
    log_probs = torch.log(action_probs)

    # Compute policy loss
    policy_loss = -(log_probs * advantages).mean()

    # Perform backpropagation and optimization step for value network
    value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    value_optimizer.step()

    # Perform backpropagation and optimization step for policy network
    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()

    return value_loss.item(), policy_loss.item()


def reinforcementLearner(actions, N, ep):
    rewardTotal, replay = pickAction(ep, actions)
    replay = sample_replays(replay, N)
    states = np.stack(replay[:, 0])
    indices = np.stack(replay[:, 1])
    rewards = np.stack(replay[:, 2])
    probs = np.stack(replay[:, 3])
    values = value_network(torch.FloatTensor(states))
    advantages, returns = compute_returns_and_advantages(values.detach().numpy(), rewards)
    train(values, advantages, returns, states, indices, probs)
    return rewardTotal



# Training example
episodes = 1000
replay_memory = []
total_rewards = []

for episode in range(episodes):
    reward = reinforcementLearner(actions, 10, 100)
    total_rewards.append(reward)
    print(f"Episode {episode + 1}, Total Reward: {reward}")

plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()