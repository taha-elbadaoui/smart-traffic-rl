import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import uuid
import os
import time
import random

class CrossroadEnv(gym.Env):
    def __init__(self, sumocfg_file, use_gui=False):
        super(CrossroadEnv, self).__init__()
        self.sumocfg_file = os.path.abspath(sumocfg_file)
        self.use_gui = use_gui
        self.label = f"env_{uuid.uuid4().hex}"
        self.conn = None
        
        self.max_cars = 50.0 
        self.step_length = 5
        self.yellow_time = 3
        self.green_time = self.step_length - self.yellow_time
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.ts_id = "J4"
        self.edges = ["in_N", "int_E", "in_S", "in_W"]

    def _get_queue_lengths(self):
        return [self.conn.edge.getLastStepHaltingNumber(e) for e in self.edges]

    def _get_state(self):
        queues = self._get_queue_lengths()
        state = np.array([np.tanh(q / self.max_cars) for q in queues], dtype=np.float32)
        return state, sum(queues)

    def step(self, action):
        target_phase = int(action) * 2 
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        
        accumulated_reward = 0.0
        throughput = 0
        
        if current_phase != target_phase:
            self.conn.trafficlight.setPhase(self.ts_id, current_phase + 1)
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                accumulated_reward -= sum(self._get_queue_lengths())
                throughput += self.conn.simulation.getArrivedNumber()
                
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.green_time):
                self.conn.simulationStep()
                accumulated_reward -= sum(self._get_queue_lengths())
                throughput += self.conn.simulation.getArrivedNumber()
        else:
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.step_length):
                self.conn.simulationStep()
                accumulated_reward -= sum(self._get_queue_lengths())
                throughput += self.conn.simulation.getArrivedNumber()
            
        state, _ = self._get_state()
        reward = float(accumulated_reward + (throughput * 10.0))
        done = self.conn.simulation.getMinExpectedNumber() <= 0
        
        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Jitter: Stagger starts to prevent port collisions
        if self.conn is None:
            time.sleep(random.uniform(0.1, 0.6))
        
        # 2. Hard Reset: Close existing connection to clear memory/sockets
        if self.conn is not None:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([
            binary, "-c", self.sumocfg_file, 
            "--no-warnings", "--start", "--no-step-log", "--random"
        ], label=self.label)
        
        self.conn = traci.getConnection(self.label)
        state, _ = self._get_state()
        return state, {}

    def close(self):
        if self.conn:
            try:
                traci.switch(self.label)
                traci.close()
            except:
                pass
            self.conn = None