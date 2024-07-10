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
        self.actor = NN(self.obs_dimensions, self.action_dimensions)
        self.critic = NN(self.obs_dimensions, 1)
        self._init_hyperparameters(hyperparameters)

        self.cov_var = torch.full(size=(self.action_dimensions,), fill_value=0.2)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self, hyperparameters):
        self.batch_size = 200
        self.timesteps = 200
        self.gamma = 0.98
        self.n_updates_per_iteration = 3
        self.clip = 0.2
        self.lr = 0.01
        self.mini_batch_size = hyperparameters.get('mini_batch_size', 64)

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
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
                #obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
                #if obs_tensor.dim() == 1:
                    #obs_tensor = obs_tensor.unsqueeze(0)
                batch_obs.append(np.array(obs))
                action, log_prob = self.get_action(torch.tensor(np.array(obs)))
                step_result = self.env.step(action)
                if len(step_result) == 3:
                    obs, rew, done = step_result
                else:
                    obs, rew, done, *_ = step_result 
                if isinstance(obs, tuple):
                    obs = obs[0]
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

            print(f"Episode finished after {ep_t + 1} timesteps with reward {sum(ep_rews)}")

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            episode_count += 1
        
        #batch_obs = np.array(batch_obs, dtype=np.float32)
        batch_acts = np.array(batch_acts, dtype=np.float32)
        #batch_log_probs = np.array(batch_log_probs, dtype=np.float32)

        print(f"batch_obs shape: {(batch_obs)}")
        print(f"batch_acts shape: {(batch_acts)}")
        print(f"batch_log_probs shape: {(batch_log_probs)}")
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
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
            
            A = batch_rtgs - V.detach()  # Ensure advantage is of correct shape
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
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optim.step()

    def mini_batch_indices(self, batch_size):
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        for start in range(0, batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            yield indices[start:end]

# Example usage
from env import FlowControlEnv
env = FlowControlEnv()
model = PPO(NN, env, lr_actor=0.01, lr_critic=0.02, mini_batch_size=64)
model.learn(10000)