import numpy as np

mu, L, r = 1, 1, 1
desired = 5
noise = 0.1

# Environment
actions = np.array([i * desired / 5 for i in range(10)])

def get_pressure(prevPressure, setP, deltaT):
    return prevPressure + (setP - prevPressure) * (1 - np.exp(-10 * deltaT))

def get_flow(pressure):
    return pressure * np.pi * r**4 / (8 * mu * L) * np.random.uniform(1 - noise, 1 + noise)

def get_reward(flow):
    return -(desired - flow) ** 2

def new_state(state, action, pressure, t):
    flow = get_flow(pressure)
    return np.array([flow, get_pressure(pressure, action, t), action, desired - flow])

reset = np.array([0, 0, 0, 0])