import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import traci
import traci.constants as tc
import uuid
import os

LIBSUMO_AVAILABLE = False
try:
    import libsumo
    LIBSUMO_AVAILABLE = True
except ImportError:
    pass

YELLOW_PHASE_MAP = {0: 1, 2: 3}

class TJunctionEnv(gym.Env):
    MAX_EPISODE_STEPS = 3600

    def __init__(self, sumocfg_file, use_gui=False, env_rank=0):
        super().__init__()
        self.sumocfg_file = os.path.abspath(sumocfg_file)
        self.use_gui = use_gui
        self.label = f"env_{uuid.uuid4().hex}"
        self.conn = None
        self.env_rank = env_rank 

        self.max_cars = 15.0
        self.step_length = 5   
        self.yellow_time = 3
        self.green_time = self.step_length - self.yellow_time 

        self.min_green_time = 15
        self.max_green_time = 60
        self.current_phase_duration = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        self.ts_id = "TL_main"
        self.incoming_edges = ["edge_N", "edge_E"]
        self.current_step = 0

    def _get_queue_lengths(self):
        # One batched fetch for every subscribed edge instead of N individual FFI calls.
        results = self.conn.edge.getAllSubscriptionResults()
        if not results:
            # Subscriptions only populate after the first simulation step (e.g. at reset).
            return [self.conn.edge.getLastStepHaltingNumber(e) for e in self.incoming_edges]
        return [int(results[e][tc.LAST_STEP_VEHICLE_HALTING_NUMBER]) for e in self.incoming_edges]

    def _get_state(self):
        queues = self._get_queue_lengths()
        norm_queues = [math.tanh(q / self.max_cars) for q in queues]

        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]

        norm_duration = [min(1.0, self.current_phase_duration / self.max_green_time)]

        state = np.array(norm_queues + phase_one_hot + norm_duration, dtype=np.float32)
        return state, sum(queues)

    def _compute_reward(self):
        queues = self._get_queue_lengths()
        n = len(queues)
        total_waiting = sum(queues)
        max_queue = max(queues)

        # Manual std (small list) avoids numpy overhead in the per-sim-step hot loop.
        mean = total_waiting / n
        std = math.sqrt(sum((q - mean) ** 2 for q in queues) / n)

        base_penalty = -(total_waiting / self.max_cars)
        starvation_penalty = -(max_queue / self.max_cars)
        imbalance_penalty = -(std / self.max_cars)

        return (1.0 * base_penalty) + (2.0 * starvation_penalty) + (0.5 * imbalance_penalty)

    def step(self, action):
        target_phase = int(action) * 2
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2

        guardrail_penalty = 0.0

        # Guardrails: Forcing phase state based on constraints
        if target_phase != current_phase and self.current_phase_duration < self.min_green_time:
            target_phase = current_phase
            guardrail_penalty -= 5.0  

        if target_phase == current_phase and self.current_phase_duration >= self.max_green_time:
            target_phase = 2 if current_phase == 0 else 0
            guardrail_penalty -= 10.0 

        accumulated_reward = 0.0
        throughput = 0
        switching_penalty = 0.0

        if current_phase != target_phase:
            switching_penalty = -1.0 

            yellow_phase = YELLOW_PHASE_MAP[current_phase]
            self.conn.trafficlight.setPhase(self.ts_id, yellow_phase)
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1

            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.green_time):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1
            
            self.current_phase_duration = self.green_time
        else:
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            for _ in range(self.step_length):
                self.conn.simulationStep()
                accumulated_reward += self._compute_reward()
                throughput += self.conn.simulation.getArrivedNumber()
                self.current_step += 1
            
            self.current_phase_duration += self.step_length

        state, _ = self._get_state()

        reward = float(
            (accumulated_reward / self.step_length)
            + (throughput * 2.0)
            + switching_penalty
            + guardrail_penalty
        )

        sim_done = self.conn.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.MAX_EPISODE_STEPS
        done = sim_done or truncated

        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_phase_duration = 0

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        sumo_args = [
            "-c", self.sumocfg_file,
            "--no-warnings", "--start", "--no-step-log", "--random",
        ]

        # Decorative city polygons are loaded for the GUI only (kept out of the
        # headless training path so simulation throughput is unaffected).
        if self.use_gui:
            poly = os.path.join(os.path.dirname(self.sumocfg_file), "env.poly.xml")
            if os.path.exists(poly):
                sumo_args += ["--additional-files", poly]

        if self.use_gui or not LIBSUMO_AVAILABLE:
            # GUI or no libsumo: fall back to socket-based TraCI.
            binary = "sumo-gui" if self.use_gui else "sumo"
            traci.start([binary] + sumo_args, label=self.label)
            self.conn = traci.getConnection(self.label)
        else:
            # In-process libsumo: each SubprocVecEnv worker has its own instance,
            # so no ports/sockets are needed (much faster than TraCI).
            libsumo.start(["sumo"] + sumo_args)
            self.conn = libsumo

        # Subscribe once per episode so every step is a single batched read.
        for e in self.incoming_edges:
            self.conn.edge.subscribe(e, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])

        state, _ = self._get_state()
        return state, {}

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None