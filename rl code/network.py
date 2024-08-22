import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class FeedForwardNN(nn.Module):
    def __init__(self, inlayers, out, is_actor=False):
        super(FeedForwardNN, self).__init__()
        self.l1 = nn.Linear(inlayers, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out)
        self.is_actor = is_actor
    
    
    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        #print("HELLO?")
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        activation1 = F.relu(self.l1(obs))
        #print("act1", activation1)
        activation2 = F.relu(self.l2(activation1))
        #print("act2", activation2)
        output = self.l3(activation2)
        if self.is_actor:
            output = F.sigmoid(output) * 20
        #print("output", output)
        return output
    