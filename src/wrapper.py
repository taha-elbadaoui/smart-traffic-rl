import gymnasium as gym
from gymnasium import spaces
import numpy as np
import libsumo as traci

class SingleIntersectionEnv(gym.Env):
    def __init__(self, sumocfg_file):
        super(SingleIntersectionEnv, self).__init__()
        self.sumocfg_file = sumocfg_file
        
        # Hyperparameters
        self.max_cars = 50.0 # Used to normalize the state
        self.yellow_time = 3 # Seconds for yellow light transition
        self.green_time = 5  # Minimum seconds for green light
        
        # ACTION SPACE: 0 = Green N/S, 1 = Green E/W
        self.action_space = spaces.Discrete(2)
        
        # OBSERVATION SPACE: [Queue_North, Queue_East] normalized (0.0 to 1.0)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_state(self):
        """Helper function to read and normalize queues."""
        # Added '_0' because TraCI needs the Lane ID, not the Edge ID
        q_n = traci.lane.getLastStepHaltingNumber("edge_N_0")
        q_e = traci.lane.getLastStepHaltingNumber("edge_E_0")
        
        # Normalize between 0 and 1
        norm_q_n = min(q_n / self.max_cars, 1.0)
        norm_q_e = min(q_e / self.max_cars, 1.0)
        
        return np.array([norm_q_n, norm_q_e], dtype=np.float32), q_n, q_e

    def step(self, action):
        # 1. Apply the action (Change the light)
        # Phase 0: North/South Green. Phase 2: East/West Green.
        # (Phases 1 and 3 are usually yellow lights in SUMO)
        target_phase = action * 2 
        traci.trafficlight.setPhase("TL_main", target_phase)
        
        # 2. Advance simulation by our green time
        for _ in range(self.green_time):
            traci.simulationStep()
            
        # 3. Observe new state
        state, raw_q_n, raw_q_e = self._get_state()
        
        # 4. Calculate Reward: Negative total waiting queue
        # The AI wants to maximize reward, so getting closer to 0 is better
        reward = -float(raw_q_n + raw_q_e)
        
        # 5. Check if simulation is over (no more cars coming)
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start or restart libsumo
        try:
            traci.close() # Close previous instance if it exists
        except:
            pass
            
        traci.start(["sumo", "-c", self.sumocfg_file, "--no-warnings"])
        
        # Get initial state
        initial_state, _, _ = self._get_state()
        return initial_state, {}

    def close(self):
        try:
            traci.close()
        except:
            pass
