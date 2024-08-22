import gym
from gym import spaces
import numpy as np

class FlowControlEnv(gym.Env):
    def __init__(self):
        super(FlowControlEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([20]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Constants
        self.mu = 1
        self.L = 1
        self.r = 1
        self.desired = 5
        self.noise = 0.02
        
        # Initial state
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([0, 0, self.desired], dtype=np.float32)
        return self.state

    def step(self, action):
        setP = action[0]  # Extract the action value from the array
        
        # Get previous state values
        _, prevPressure, _ = self.state
        
        # Calculate new pressure
        pressure = self.get_pressure(prevPressure, setP)
        
        # Calculate flow
        flow = self.get_flow(pressure)
        
        # Calculate reward
        reward = self.get_reward(flow)
        
        # Update state
        self.state = self.new_state(setP, pressure, flow)
        
        # Ensure the new state is returned in the correct shape
        obs = self.state.reshape(1, -1)
        
        # Check if done (you can define your own condition, here it's always False)
        done = False
        
        return obs, reward, done, {}
    
    def get_pressure(self, prevPressure, setP):
        return prevPressure + (setP - prevPressure) * (1 - np.exp(-0.1))
    
    def get_flow(self, pressure):
        return pressure * np.pi * self.r**4 / (8 * self.mu * self.L) * np.random.normal(1, self.noise)
    
    def get_reward(self, flow):
        return -(self.desired - flow) ** 2
    
    def new_state(self, action, pressure, flow):
        return np.array([pressure, action, self.desired - flow], dtype=np.float32)
