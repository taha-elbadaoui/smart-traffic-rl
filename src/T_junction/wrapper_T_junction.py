import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import uuid
import os
import time
import random

# Fixed yellow phase map: green_phase -> yellow_phase
# TL_main has phases: 0=GGrr, 1=yyrr, 2=rrGG, 3=rryy
YELLOW_PHASE_MAP = {0: 1, 2: 3}


class TJunctionEnv(gym.Env):
    """
    Gymnasium environment for a T-Junction traffic signal controller.

    Observation (3,):  [queue_N_norm, queue_E_norm, phase_one_hot_0, phase_one_hot_1]
        - queue norms: tanh-normalized halting vehicle counts
        - phase one-hot: current active green phase encoded as [1,0] or [0,1]

    Action (Discrete 2):
        0 → Phase 0 green (East → straight + South)
        1 → Phase 2 green (North → straight + East)

    Reward: normalized congestion penalty + throughput bonus - switching penalty
    """

    MAX_EPISODE_STEPS = 3600  # Hard cap to prevent infinite episodes

    def __init__(self, sumocfg_file, use_gui=False):
        super().__init__()
        self.sumocfg_file = os.path.abspath(sumocfg_file)
        self.use_gui = use_gui
        self.label = f"env_{uuid.uuid4().hex}"
        self.conn = None

        self.max_cars = 15.0
        self.step_length = 10   # Increased from 5 — gives green phase enough time to clear cars
        self.yellow_time = 3
        self.green_time = self.step_length - self.yellow_time

        self.action_space = spaces.Discrete(2)
        # Extended to (4,): 2 queue values + 2 phase one-hot
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.ts_id = "TL_main"
        self.incoming_edges = ["edge_N", "edge_E"]

        self.current_step = 0

    def _get_queue_lengths(self):
        return [self.conn.edge.getLastStepHaltingNumber(e) for e in self.incoming_edges]

    def _get_state(self):
        queues = self._get_queue_lengths()
        norm_queues = [float(np.tanh(q / self.max_cars)) for q in queues]

        # One-hot encode current phase so agent knows what's active
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]

        state = np.array(norm_queues + phase_one_hot, dtype=np.float32)
        return state, sum(queues)

    def _compute_reward(self):
        queues = self._get_queue_lengths()
        total_waiting = sum(queues)

        # Base congestion penalty, normalized to [-1, 0]
        base_penalty = -(total_waiting / self.max_cars)

        # Non-linear starvation penalty on worst lane
        max_queue = max(queues)
        starvation_penalty = -((max_queue / self.max_cars) ** 2)

        # Imbalance penalty (std deviation across lanes)
        imbalance_penalty = -(np.std(queues) / self.max_cars)

        return (1.0 * base_penalty) + (2.0 * starvation_penalty) + (0.5 * imbalance_penalty)

    def step(self, action):
        target_phase = int(action) * 2  # Maps action 0→phase 0, action 1→phase 2
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)

        # Normalise: if SUMO has advanced to a yellow phase internally,
        # snap back to the nearest green so our map is always valid.
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = max(YELLOW_PHASE_MAP.keys(),
                                key=lambda p: 0 if p != current_phase - 1 else 1)

        accumulated_reward = 0.0
        throughput = 0
        switching_penalty = 0.0

        if current_phase != target_phase:
            # Scale switching cost relative to current total congestion
            current_queues = sum(self._get_queue_lengths())
            switching_penalty = -(max(5.0, current_queues * 0.5))

            # Yellow transition — use fixed map, never compute current_phase + 1
            yellow_phase = YELLOW_PHASE_MAP[current_phase]
            self.conn.trafficlight.setPhase(self.ts_id, yellow_phase)
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1

            # Target green phase
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.green_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1
        else:
            # Maintain current green phase
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.step_length):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1

        state, _ = self._get_state()

        # Normalise accumulated reward over the step window + throughput bonus
        reward = float(
            (accumulated_reward / self.step_length)
            + (throughput * 2.0)
            + switching_penalty
        )

        sim_done = self.conn.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.MAX_EPISODE_STEPS
        done = sim_done

        return state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self.conn is None:
            # Stagger startup to reduce port collision risk in SubprocVecEnv
            time.sleep(random.uniform(0.1, 0.6))

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start(
            [binary, "-c", self.sumocfg_file,
             "--no-warnings", "--start", "--no-step-log", "--random"],
            label=self.label
        )

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