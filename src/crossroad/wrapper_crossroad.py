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

    def _compute_reward(self):
        """
        Decomposed reward function. Each term targets a specific failure mode.

        Old reward:  -(sum of queues) + throughput * 10
        Problem:     throughput bonus was large enough that maximizing one
                     direction's flow outweighed letting other lanes starve.

        New reward has three terms:

        1. BASE PENALTY — punishes total congestion across all lanes.
           Normalized by max_cars so it's always in [-1, 0].

        2. STARVATION PENALTY — punishes the WORST lane specifically.
           Squared so the penalty grows non-linearly: 5 cars = 0.01 penalty,
           10 cars = 0.04, 20 cars = 0.16, 40 cars = 0.64.
           This forces the agent to care about its worst lane, not just the average.
           Without this, phase-locking is rational (and what you observed).

        3. IMBALANCE PENALTY — punishes unequal distribution of cars.
           std([0,0,0,10]) >> std([3,2,3,2]). Nudges toward fairness.
           Lower weight (0.5) because it's a soft preference, not a hard constraint.

        Throughput bonus is kept but weight reduced from 10.0 to 2.0.
        One car exiting now cancels ~2 car-steps of waiting, not 10.
        """
        queues = self._get_queue_lengths()

        # 1. Base: total congestion, normalized to [-1, 0]
        total_waiting = sum(queues)
        base_penalty = -(total_waiting / self.max_cars)

        # 2. Starvation: squared penalty on the worst lane
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

        if current_phase != target_phase:
            # Transition to yellow before switching phases.
            # current_phase + 1 is always the yellow phase in a standard
            # 4-phase SUMO tlLogic (green, yellow, green, yellow).
            self.conn.trafficlight.setPhase(self.ts_id, current_phase + 1)
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()

            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.green_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
        else:
            # Already on the right phase, just run the full step
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.step_length):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()

        state, _ = self._get_state()

        # Throughput bonus: reduced from 10.0 to 2.0 so it can't overwhelm
        # the starvation penalty. One car exiting = ~2 car-steps of relief.
        reward = float(accumulated_reward + (throughput * 2.0))
        done = self.conn.simulation.getMinExpectedNumber() <= 0

        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- THE MASSIVE OVERHAUL INJECTION ---
        # Generate completely new traffic logic before starting the SUMO server
        route_file = os.path.join(os.path.dirname(self.sumocfg_file), "dynamic.rou.xml")
        try:
            from traffic_generator import generate_dynamic_traffic
            generate_dynamic_traffic(route_file)
        except ImportError:
            print("Warning: traffic_generator not found. Using static routes.")
        # --------------------------------------

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