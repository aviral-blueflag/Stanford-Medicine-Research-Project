from network import NN

import torch
from torch.optim import Adam
import numpy as np
from torch.distributions import MultivariateNormal
class PPO:
    
    def __init__(self, env):
        self.env = env
        self.obs_dimensions = env.observation_space.shape[0]
        self.action_dimensions = env.action_space.shape[0]
        self.actor = NN(self.obs_dimensions, self.action_dimensions)
        self.critic = NN(self.obs_dimensions, 1)
        self._init_hyperparameters()

        self.cov_var = torch.full(size=(self.action_dimensions,), fill_value=0.5)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    
    def _init_hyperparameters(self):
        self.batch_size = 1000
        self.timesteps = 200
        self.gamma = 0.98
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def compute_rtgs(self, batch_rews):

        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        t = 0

        while t < self.batch_size:
            ep_rews = []
            obs = self.env.reset()
            done = False

            for ep_t in range(self.timesteps):
                t += 1

                obs_tensor = torch.tensor(obs, dtype=torch.float32)  # Ensure obs is a tensor
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                #print(f"rollout - obs_tensor shape before get_action: {obs_tensor.shape}")  # Debugging line
                batch_obs.append(obs_tensor)
                action, log_prob = self.get_action(obs_tensor)  # Calls get_action
                #print(f"rollout - action taken: {action}, log_prob: {log_prob}")  # Debugging line
                obs, rew, done, _ = self.env.step(action)
                #print(f"rollout - new obs from env: {obs}")  # Debugging line

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.stack(batch_obs)  # Changed to stack tensors
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)


        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)  # Ensure obs is a tensor
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        #print(f"get_action - obs shape: {obs.shape}")
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()


    
    def evaluate(self, batch_obs, batch_acts):
        #print(f"evaluate - batch_obs shape: {batch_obs.shape}, batch_acts shape: {batch_acts.shape}")
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
    
    def learn(self, time):
        t = 0
        while t < time:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t += np.sum(batch_lens)
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            A = batch_rtgs - V.detach()
            A = (A - A.mean()) / (A.std() + 1e-10)
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A

                actor_loss = (-torch.min(surr1, surr2)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward(retain_graph=True)    
                self.critic_optim.step()

                
                
                #self.logger['actor_losses'].append(actor_loss.detach())

from env import FlowControlEnv

env = FlowControlEnv()
model = PPO(env)
model.learn(10000)
