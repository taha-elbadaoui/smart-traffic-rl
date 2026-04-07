import gymnasium as gym
from gymnasium import spaces
import numpy as np
import libsumo as traci # CRITICAL: High-speed C++ binding, bypassing sockets

class SingleIntersectionEnv(gym.Env):
    def __init__(self, sumocfg_file):
        super(SingleIntersectionEnv, self).__init__()
        self.sumocfg_file = sumocfg_file
        
        # ACTION SPACE: 2 buttons for the agent to press
        # Action 0: Green North/South
        # Action 1: Green East/West
        self.action_space = spaces.Discrete(2)
        
        # OBSERVATION SPACE: What the agent sees
        # Array of 2 values: [Queue_North, Queue_East]
        # Normalized between 0.0 and 1.0 to keep the neural network stable
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        """Executes the chosen phase, advances the simulation, and returns the new state."""
        # TODO: Tell TraCI to change the light to 'action'
        # TODO: Advance SUMO by 1 step
        # TODO: Read the new queue lengths from TraCI
        # TODO: Calculate the Reward: -(queue_N + queue_E)
        
        state = np.array([0.0, 0.0], dtype=np.float32) # Placeholder
        reward = 0.0 # Placeholder
        done = False # Did the simulation end?
        truncated = False 
        info = {}
        
        return state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Restarts the SUMO simulation for a new episode."""
        super().reset(seed=seed)
        
        # TODO: Start libsumo with the sumocfg_file
        
        initial_state = np.array([0.0, 0.0], dtype=np.float32) # Placeholder
        info = {}
        return initial_state, info

    def close(self):
        """Safely shuts down the SUMO instance."""
        try:
            traci.close()
        except:
            pass
