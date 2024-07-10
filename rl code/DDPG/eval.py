import torch
from model import Actor
from environment import reset, get_pressure, get_flow, new_state
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(state):
    return state.flatten()

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# Load the trained actor model
state_dim = len(flatten_state(reset()))
action_dim = 1
max_action = 13.2 #12.73

actor = Actor(state_dim, action_dim, max_action).to(device)
actor.load_state_dict(torch.load('DDPG_actor.pth'))
actor.eval()  # Set the model to evaluation mode

def select_action(state):
    flat_state = flatten_state(state)
    flat_state = normalize(flat_state)
    state = torch.FloatTensor(flat_state).reshape(1, -1).to(device)
    action = actor(state).cpu().data.numpy().flatten()
    return action.clip(-max_action, max_action)

# Example usage of the trained model
state = reset()
pressure = 0
episode_length = 2000
for t in range(episode_length):
    action = select_action(state)
    next_state, flow = new_state(action, pressure, state[2])
    state = next_state
    pressure = state[0]
    print(f"Step: {t}, Action: {action}, Flow: {flow}")

print("Inference finished!")
