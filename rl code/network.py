import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NN(nn.Module):
    def __init__(self, inlayers, out):
        super(NN, self).__init__()
        self.l1 = nn.Linear(inlayers, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            x = torch.tensor(obs, dtype=torch.float)
        x = obs
        #print(f"Input shape: {x.shape}")  # Debugging line
        act1 = F.relu(self.l1(x))
        #print(f"Shape after l1: {act1.shape}")  # Debugging line
        act2 = F.relu(self.l2(act1))
        #print(f"Shape after l2: {act2.shape}")  # Debugging line
        act3 = F.relu(self.l3(act2))
        #print(f"Shape after l3: {act3.shape}")  # Debugging line
        return act3
    