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

def initialize_model(in_features, out_features, h1=64, h2=64):
    model = Model(in_features=in_features, h1=h1, h2=h2, out_features=out_features)
    return model


def train_with_loss(network, loss, optimizer):
    optimizer.zero_grad()
    
    print("Before backward pass - Loss value:", loss.item())
    print("Before backward pass - Loss grad:", loss.grad if loss.grad is not None else "None")

    loss.backward(retain_graph=True)
    
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

    #for name, param in network.named_parameters():
        #print(f"After backward pass - {name} grad: {param.grad}") 

    optimizer.step()
    
    #for name, param in network.named_parameters():
        #print(f"After optimizer step - {name} data: {param.data}") 
