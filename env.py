import numpy as np

mu, L, r = 1, 1, 1
desired = 5
noise = 0.1

# Environment
actions = np.array([i * desired / 50 for i in range(100)])

def get_pressure(prevPressure, setP):
    return prevPressure + (setP - prevPressure) * (1 - np.exp(-0.1))

def get_flow(pressure):
    return pressure * np.pi * r**4 / (8 * mu * L) * np.random.uniform(1 - noise, 1 + noise)

def get_reward(flow):
    return -(desired - flow) ** 2

def new_state(action, pressure, flow):
    return np.array([pressure, action, desired - flow])

reset = np.array([0, 0, desired])