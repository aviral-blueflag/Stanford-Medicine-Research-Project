import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Model class
class Model(nn.Module):
    def __init__(self, in_features=10, h1=8, h2=6, out_features=12):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x, policy=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        if policy:
            return F.softmax(x, dim=1)
        else:
            return x

def initialize_model(in_features, out_features, h1=8, h2=6):
    model = Model(in_features=in_features, h1=h1, h2=h2, out_features=out_features)
    return model

def train_with_loss(network, loss, lr=0.01):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    optimizer.zero_grad()
    
    # Ensure the network parameters have requires_grad=True
    for param in network.parameters():
        param.requires_grad = True

    # Proper backward pass for loss
    loss.backward()
    
    # Print gradients after backward pass
    for name, param in network.named_parameters():
        print(f"After backward pass - {name} grad: {param.grad}")
    
    optimizer.step()
    
    # Print parameter values after step
    for name, param in network.named_parameters():
        print(f"After optimizer step - {name} data: {param.data}")
