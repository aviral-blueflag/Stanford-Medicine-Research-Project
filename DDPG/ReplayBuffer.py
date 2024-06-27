import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            torch.FloatTensor(np.array(batch_states)).to(device),
            torch.FloatTensor(np.array(batch_actions)).to(device),
            torch.FloatTensor(np.array(batch_rewards)).to(device),
            torch.FloatTensor(np.array(batch_next_states)).to(device),
            torch.FloatTensor(np.array(batch_dones)).to(device)
        )

    def size(self):
        return len(self.storage)
