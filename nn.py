import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the Model class
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=6, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
'''
# Load and prepare the dataset
#url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
#my_df = pd.read_csv(url)

# Ensure all species are correctly replaced

my_df['species'] = my_df['species'].replace({'setosa': 0, 'versicolor': 1, 'virginica': 2,
                                             'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

X = my_df.drop('species', axis=1)
y = my_df['species']
'''
def initialize_model(in_features, out_features, h1=8, h2=6):
    model = Model(in_features=in_features, h1=h1, h2=h2, out_features=out_features)
    return model

def train_model(model, X_train, y_train, epochs=100, lr=0.01):
    criterion = nn.MSELoss()  # Assuming regression task
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs): 
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return losses

def train_with_loss(network, loss, lr=0.01):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
# Plotting the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
'''