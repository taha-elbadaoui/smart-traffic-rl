import gymnasium as gym
from gymnasium import spaces
import numpy as np
import uuid
import os
import time

import traci
LIBSUMO_AVAILABLE = False
try:
    import libsumo
    LIBSUMO_AVAILABLE = True
except ImportError:
    pass

YELLOW_PHASE_MAP = {0: 1, 2: 3}

class CrossroadEnv(gym.Env):
    MAX_EPISODE_STEPS = 3600  

    def __init__(self, sumocfg_file, use_gui=False, rank=0):
        super().__init__()
        self.sumocfg_file = os.path.abspath(sumocfg_file)
        self.use_gui = use_gui
        self.rank = rank
        self.label = f"env_{uuid.uuid4().hex}"
        self.conn = None

        self.max_cars = 50.0
        self.step_length = 25   
        self.yellow_time = 3
        self.green_time = self.step_length - self.yellow_time
        
        self.min_green_time = 15
        self.max_green_time = 60
        self.current_phase_duration = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.ts_id = "J4"
        self.edges = ["in_N", "int_E", "in_S", "in_W"]
        self.current_step = 0
        
        # Completely isolate the generated XML file name for this specific worker rank
        self.config_dir = os.path.dirname(self.sumocfg_file)
        self.worker_route_filename = f"dynamic_worker_{self.rank}.rou.xml"
        self.route_file = os.path.join(self.config_dir, self.worker_route_filename)

    def _get_queue_lengths(self):
        return [self.conn.edge.getLastStepHaltingNumber(e) for e in self.edges]

    def _get_state(self):
        queues = self._get_queue_lengths()
        norm_queues = [float(np.tanh(q / self.max_cars)) for q in queues]

        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]
        
        norm_duration = [min(1.0, self.current_phase_duration / self.max_green_time)]

        state = np.array(norm_queues + phase_one_hot + norm_duration, dtype=np.float32)
        return state, sum(queues)

    def _compute_reward(self):
        queues = self._get_queue_lengths()
        total_waiting = sum(queues)

        base_penalty = -(total_waiting / self.max_cars)
        
        max_queue = max(queues)
        starvation_penalty = -(max_queue / self.max_cars)

        imbalance_penalty = -(float(np.std(queues)) / self.max_cars)

        return (1.0 * base_penalty) + (2.0 * starvation_penalty) + (0.5 * imbalance_penalty)

    def step(self, action):
        target_phase = int(action) * 2  

        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2
            
        guardrail_penalty = 0.0

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

        # Create isolated files cleanly
        try:
            from traffic_generator import generate_dynamic_traffic
            generate_dynamic_traffic(self.route_file)
        except ImportError:
            pass

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        # Command arguments: Explicitly instruct SUMO to override its config mapping
        # and parse the isolated specific worker route file tracking layout.
        sumo_args = [
            "-c", self.sumocfg_file,
            "--route-files", self.route_file,
            "--no-warnings",
            "--no-step-log",
            "--random"
        ]

        if self.use_gui or not LIBSUMO_AVAILABLE:
            binary = "sumo-gui" if self.use_gui else "sumo"
            traci.start([binary] + sumo_args, label=self.label)
            self.conn = traci.getConnection(self.label)
        else:
            # Parallel worker thread context isolation safety sleep
            time.sleep(self.rank * 0.15)
            libsumo.start(["sumo"] + sumo_args)
            self.conn = libsumo

        state, _ = self._get_state()
        return state, {}

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        
        # Cleanup isolated transient files to prevent directory congestion
        if os.path.exists(self.route_file):
            try:
                os.remove(self.route_file)
            except Exception:
                pass