import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys

# Dynamic import based on GUI needs
import libsumo as headless_traci
import traci as gui_traci

class SingleIntersectionEnv(gym.Env):
    def __init__(self, sumocfg_file, use_gui=False):
        super(SingleIntersectionEnv, self).__init__()
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.traci = gui_traci if self.use_gui else headless_traci
        
        self.max_cars = 15.0 # Lowered from 50. 5 cars/min means queues rarely hit 50.
        self.green_time = 5  
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_state(self):
        q_n = self.traci.lane.getLastStepHaltingNumber("edge_N_0")
        q_e = self.traci.lane.getLastStepHaltingNumber("edge_E_0")
        
        norm_q_n = min(q_n / self.max_cars, 1.0)
        norm_q_e = min(q_e / self.max_cars, 1.0)
        
        return np.array([norm_q_n, norm_q_e], dtype=np.float32), q_n, q_e

    def step(self, action):
        target_phase = action * 2 
        self.traci.trafficlight.setPhase("TL_main", target_phase)
        
        for _ in range(self.green_time):
            self.traci.simulationStep()
            
        state, raw_q_n, raw_q_e = self._get_state()
        reward = -float(raw_q_n + raw_q_e)
        done = self.traci.simulation.getMinExpectedNumber() <= 0
        
        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            self.traci.close()
        except:
            pass
            
        binary = "sumo-gui" if self.use_gui else "sumo"
        self.traci.start([binary, "-c", self.sumocfg_file, "--no-warnings", "--start"])
        
        initial_state, _, _ = self._get_state()
        return initial_state, {}

    def close(self):
        try:
            self.traci.close()
        except:
            pass
