import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import uuid
import os
import time
import random

# Fixed yellow phase map: green_phase -> yellow_phase
# J4 tlLogic has phases: 0=GGGg..., 1=yyyy..., 2=rrrrGGGg..., 3=rrrryyyyrrrr
YELLOW_PHASE_MAP = {0: 1, 2: 3}

# Base port — each worker gets BASE_PORT + rank so no two workers ever collide
BASE_PORT = 8813


class CrossroadEnv(gym.Env):
    """
    Gymnasium environment for a 4-way Crossroad traffic signal controller.

    Observation (6,):
        [queue_N_norm, queue_E_norm, queue_S_norm, queue_W_norm, phase_oh_0, phase_oh_1]
        - queue norms : tanh-normalised halting vehicle counts per incoming edge
        - phase one-hot: [1,0] = N/S green active, [0,1] = E/W green active

    Action (Discrete 2):
        0 -> Phase 0 green (N/S axes green)
        1 -> Phase 2 green (E/W axes green)

    Reward: weighted combination of congestion, starvation, imbalance penalties
            + throughput bonus - context-scaled switching penalty
    """

    MAX_EPISODE_STEPS = 3600  # Hard cap: prevents infinite hang on gridlock

    def __init__(self, sumocfg_file, use_gui=False, rank=0):
        super().__init__()
        self.sumocfg_file = os.path.abspath(sumocfg_file)
        self.use_gui = use_gui
        self.rank = rank
        self.port = BASE_PORT + rank  # Fixed unique port per worker — eliminates collisions
        self.label = f"env_{uuid.uuid4().hex}"
        self.conn = None

        self.max_cars = 50.0
        self.step_length = 10   # 10s steps: 3s yellow + 7s green minimum
        self.yellow_time = 3
        self.green_time = self.step_length - self.yellow_time

        self.action_space = spaces.Discrete(2)
        # 4 queue values + 2 phase one-hot
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self.ts_id = "J4"
        self.edges = ["in_N", "int_E", "in_S", "in_W"]

        self.current_step = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_queue_lengths(self):
        return [self.conn.edge.getLastStepHaltingNumber(e) for e in self.edges]

    def _get_state(self):
        queues = self._get_queue_lengths()
        norm_queues = [float(np.tanh(q / self.max_cars)) for q in queues]

        # Provide the agent with the current active green phase
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        # During yellow transitions current_phase will be 1 or 3 — map to nearest green
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]

        state = np.array(norm_queues + phase_one_hot, dtype=np.float32)
        return state, sum(queues)

    def _compute_reward(self):
        queues = self._get_queue_lengths()
        total_waiting = sum(queues)

        # 1. Base congestion penalty, normalised to [-1, 0]
        base_penalty = -(total_waiting / self.max_cars)

        # 2. Non-linear starvation penalty on the worst single lane
        max_queue = max(queues)
        starvation_penalty = -((max_queue / self.max_cars) ** 2)

        # 3. Imbalance penalty (std dev across lanes, normalised)
        imbalance_penalty = -(float(np.std(queues)) / self.max_cars)

        return (
            1.0 * base_penalty
            + 2.0 * starvation_penalty
            + 0.5 * imbalance_penalty
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(self, action):
        target_phase = int(action) * 2  # action 0 -> phase 0, action 1 -> phase 2

        # Read current phase; snap to nearest green if we're in a yellow
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2

        accumulated_reward = 0.0
        throughput = 0
        switching_penalty = 0.0

        if current_phase != target_phase:
            # Context-scaled switching cost proportional to current congestion
            current_total = sum(self._get_queue_lengths())
            switching_penalty = -(max(5.0, current_total * 0.5))

            # Yellow transition using fixed YELLOW_PHASE_MAP (never current_phase + 1)
            yellow_phase = YELLOW_PHASE_MAP[current_phase]
            self.conn.trafficlight.setPhase(self.ts_id, yellow_phase)
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1

            # Target green phase execution
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

        reward = float(
            (accumulated_reward / self.step_length)
            + (throughput * 2.0)
            + switching_penalty
        )

        sim_done = self.conn.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.MAX_EPISODE_STEPS
        # Must be done=True (not just truncated) for SB3 to log ep_rew_mean
        done = sim_done or truncated

        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Regenerate dynamic traffic for this episode
        route_file = os.path.join(os.path.dirname(self.sumocfg_file), "dynamic.rou.xml")
        try:
            from traffic_generator import generate_dynamic_traffic
            generate_dynamic_traffic(route_file)
        except ImportError:
            print("Warning: traffic_generator not found - using existing static routes.")

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        # Small stagger based on rank so workers don't all hit the OS simultaneously
        time.sleep(self.rank * 0.1)

        binary = "sumo-gui" if self.use_gui else "sumo"
        
        # FIX: Removed explicit "--remote-port" flags from the list.
        # TraCI manages the connection setup naturally via the port keyword argument.
        traci.start(
            [
                binary, "-c", self.sumocfg_file,
                "--no-warnings", "--start", "--no-step-log"
            ],
            label=self.label,
            port=self.port
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