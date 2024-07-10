
from network import NN
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

class PPO:
    
    def __init__(self, policy_class, env, **hyperparameters):
        self.env = env
        self.obs_dimensions = env.observation_space.shape[0]
        self.action_dimensions = env.action_space.shape[0]
        self.actor = policy_class(self.obs_dimensions, self.action_dimensions)
        self.critic = policy_class(self.obs_dimensions, 1)
        self._init_hyperparameters(hyperparameters)

        self.cov_var = torch.full(size=(self.action_dimensions,), fill_value=0.2)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=0.005)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.01)
        #self.actor_scheduler = StepLR(self.actor_optim, step_size=1000, gamma=0.95)
        #self.critic_scheduler = StepLR(self.critic_optim, step_size=1000, gamma=0.95)

    def _init_hyperparameters(self, hyperparameters):
        self.batch_size = 200
        self.timesteps = 200
        self.gamma = 0.98
        self.n_updates_per_iteration = 3
        self.clip = 0.2
        self.lr = 0.001

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
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
        episode_count = 0

        while t < self.batch_size:
            ep_rews = []
            obs = self.env.reset()
            done = False
            if isinstance(obs, tuple):
                obs = obs[0]
            for ep_t in range(self.timesteps):
                if ep_t == self.timesteps - 1:
                    done = True
                t += 1
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                batch_obs.append(obs_tensor)
                action, log_prob = self.get_action(obs_tensor)
                step_result = self.env.step(action)
                if len(step_result) == 3:
                    obs, rew, done = step_result
                else:
                    obs, rew, done, *_ = step_result 
                if isinstance(obs, tuple):
                    obs = obs[0]
                if done:
                    break
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            print(f"Episode finished after {ep_t + 1} timesteps with reward {sum(ep_rews)}")

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

            episode_count += 1

        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        obs = obs.clone().detach()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean = self.actor(obs, is_actor=True)
        #print(f"Actor network output (mean): {mean}")  # Debugging actor output
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        low = self.env.action_space.low[0]
        high = self.env.action_space.high[0]
        action = low + (0.5 * (action + 1.0) * (high - low))
        #action = torch.clamp(action, low, high)
        
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs, is_actor=False).squeeze()
        #print(f"Critic network output (V): {V}")  # Debugging critic output
        mean = self.actor(batch_obs, is_actor=True)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
    
    def learn(self, time):
        t = 0
        while t < time:
            print(f"Learning iteration, timesteps so far: {t}")
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t += np.sum(batch_lens)
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            A = batch_rtgs - V.detach()
            A = (A - A.mean()) / (A.std() + 1e-10)
            
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                print(f"Actor loss: {actor_loss.item()} Critic loss: {critic_loss.item()}")
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        print(f"Actor gradient for {name}: {param.grad.norm().item()}")
                    else:
                        print(f"Actor gradient for {name}: NONE") 

                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                """ for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        #print(f"Critic gradient for {name}: {param.grad.norm().item()}")
                    else:
                        print(f"Critic gradient for {name}: NONE") """

                self.critic_optim.step()

                #self.actor_scheduler.step()
                #self.critic_scheduler.step()

# Example usage
from env import FlowControlEnv
env = FlowControlEnv()
model = PPO(NN, env)
model.learn(10000)