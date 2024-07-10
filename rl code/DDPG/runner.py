import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import Actor, Critic
from ReplayBuffer import ReplayBuffer
from environment import reset, get_pressure, get_flow, get_reward, new_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(state):
    return state.flatten()

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.005

        self.total_it = 0
        self.actor_losses = []
        self.critic_losses = []
        self.noise = OUNoise(action_dim)

    def select_action(self, state, noise=0.1):
        flat_state = flatten_state(state)
        flat_state = normalize(flat_state)
        state = torch.FloatTensor(flat_state).reshape(1, -1).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = action + self.noise.noise()
        return action.clip(-self.max_action, self.max_action)

    def train(self, print_updates=False):
        if self.replay_buffer.size() < self.batch_size:
            return

        self.total_it += 1

        for _ in range(2):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape(-1, 1).to(device)

            noise = (torch.randn_like(action) * 0.1).clamp(-0.3, 0.3)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.discount * target_q

            current_q = self.critic(state, action)

            critic_loss = F.mse_loss(current_q, target_q.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        if self.total_it % 2 == 0:
            actor_loss = -self.critic(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)

            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.actor_losses.append(actor_loss.item())

        self.critic_losses.append(critic_loss.item())

state_dim = len(flatten_state(reset()))
action_dim = 1
max_action = 13 # Updated max_action based on flow calculations

agent = DDPG(state_dim, action_dim, max_action)

num_episodes = 1000
episode_length = 200
warm_up_steps = 1000

rewards_list = []
flow_list = []

state = reset()
pressure = 0
for step in range(warm_up_steps):
    action = np.random.uniform(-max_action, max_action, action_dim)
    next_state, flow = new_state(action, pressure, state[2])
    reward = get_reward(flow)
    done = False

    agent.replay_buffer.add(flatten_state(state), action, float(reward), flatten_state(next_state), float(done))
    state = next_state
    pressure = state[0]
    if done:
        state = reset()
        pressure = 0

for episode in range(num_episodes):
    state = reset()
    pressure = 0
    episode_reward = 0
    episode_flows = []

    for t in range(episode_length):
        action = agent.select_action(state)
        next_state, flow = new_state(action, pressure, state[2])
        reward = get_reward(flow)
        done = False

        agent.replay_buffer.add(flatten_state(state), action, float(reward), flatten_state(next_state), float(done))
        state = next_state
        pressure = state[0]
        episode_reward += reward
        episode_flows.append(flow)

        print_updates = episode % 10 == 0 and t == episode_length - 1
        agent.train(print_updates=print_updates)

        if done:
            break

    rewards_list.append(episode_reward)
    flow_list.append(episode_flows)
    if episode % 10 == 0:
        if len(agent.critic_losses) > 0 and len(agent.actor_losses) > 0:
            print(f"Episode: {episode}, Reward: {episode_reward}, Critic Loss: {agent.critic_losses[-1]}, Actor Loss: {agent.actor_losses[-1]}")

torch.save(agent.actor.state_dict(), 'DDPG_actor.pth')
print("Training finished!")

# Plotting the rewards
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward Over Time')

# Plotting the critic losses
plt.subplot(1, 3, 2)
plt.plot(agent.critic_losses)
plt.xlabel('Training Step')
plt.ylabel('Critic Loss')
plt.title('Critic Loss Over Time')

# Plotting the actor losses
plt.subplot(1, 3, 3)
plt.plot(agent.actor_losses)
plt.xlabel('Training Step')
plt.ylabel('Actor Loss')
plt.title('Actor Loss Over Time')

plt.tight_layout()
plt.show()

# Plotting the flow of the last episode
plt.figure(figsize=(6, 5))
plt.plot(flow_list[-1])
plt.xlabel('Time Step')
plt.ylabel('Flow')
plt.title('Flow Over Time in the Last Episode')
plt.show()
