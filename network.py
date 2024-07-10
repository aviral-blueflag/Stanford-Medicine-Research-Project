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
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        activation1 = F.relu(self.l1(obs))
        activation2 = F.relu(self.l2(activation1))
        output = self.l3(activation2)
        return output
    