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
        # CRITICAL FIX: Increase step length. 5s was too short for a 3s yellow transition.
        self.step_length = 10 
        self.yellow_time = 3
        self.green_time = self.step_length - self.yellow_time

        self.action_space = spaces.Discrete(2)
        
        # CRITICAL FIX: Expanded to shape (6,) to include the current phase one-hot encoded vector
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self.ts_id = "J4"
        self.edges = ["in_N", "int_E", "in_S", "in_W"]

    def _get_queue_lengths(self):
        return [self.conn.edge.getLastStepHaltingNumber(e) for e in self.edges]

    def _get_state(self):
        queues = self._get_queue_lengths()
        norm_queues = [np.tanh(q / self.max_cars) for q in queues]
        
        # CRITICAL FIX: Provide the agent with the current light state context
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]
        
        state = np.array(norm_queues + phase_one_hot, dtype=np.float32)
        return state, sum(queues)

    def _compute_reward(self):
        queues = self._get_queue_lengths()

        # 1. Base: total congestion, normalized to [-1, 0]
        total_waiting = sum(queues)
        base_penalty = -(total_waiting / self.max_cars)

        # 2. Starvation: non-linear squared penalty on the worst lane
        max_queue = max(queues)
        starvation_penalty = -((max_queue / self.max_cars) ** 2)

        # 3. Imbalance: standard deviation across lanes, normalized
        imbalance_penalty = -(np.std(queues) / self.max_cars)

        reward = (
            1.0 * base_penalty +
            2.0 * starvation_penalty +
            0.5 * imbalance_penalty
        )
        return reward

    def step(self, action):
        target_phase = int(action) * 2
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)

        accumulated_reward = 0.0
        throughput = 0
        switching_penalty = 0.0

        if current_phase != target_phase:
            # CRITICAL FIX: Strong penalty for changing lights excessively
            switching_penalty = -5.0 
            
            # Yellow Phase Transition
            self.conn.trafficlight.setPhase(self.ts_id, current_phase + 1)
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()

            # Target Green Phase Execution
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.green_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
        else:
            # Maintain active Green Phase
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.step_length):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()

        state, _ = self._get_state()
        
        # Balance step accumulation metrics with throughput reward
        reward = float((accumulated_reward / self.step_length) + (throughput * 2.0) + switching_penalty)
        done = self.conn.simulation.getMinExpectedNumber() <= 0

        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        route_file = os.path.join(os.path.dirname(self.sumocfg_file), "dynamic.rou.xml")
        try:
            from traffic_generator import generate_dynamic_traffic
            generate_dynamic_traffic(route_file)
        except ImportError:
            print("Warning: traffic_generator not found. Using static routes.")

        if self.conn is None:
            time.sleep(random.uniform(0.1, 0.6))
        
        if self.conn is not None:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([
            binary, "-c", self.sumocfg_file, 
            "--no-warnings", "--start", "--no-step-log"
        ], label=self.label)
        
        self.conn = traci.getConnection(self.label)
        state, _ = self._get_state()
        return state, {}

    def close(self):
        if self.conn:
            try:
                traci.switch(self.label)
                traci.close()
            except Exception:
                pass
            self.conn = None