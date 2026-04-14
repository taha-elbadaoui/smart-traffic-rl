import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import uuid

class TJunctionEnv(gym.Env):
    def __init__(self, sumocfg_file, use_gui=False):
        super(TJunctionEnv, self).__init__()
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        
        self.label = f"env_{uuid.uuid4().hex}"
        self.conn = None
        
        self.max_cars = 15.0 
        self.step_length = 5  
        self.yellow_time = 3 
        self.green_time = self.step_length - self.yellow_time
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.ts_id = "TL_main"
        self.incoming_edges = ["edge_N", "edge_E"]

    def _get_queue_lengths(self):
        return [self.conn.edge.getLastStepHaltingNumber(edge) for edge in self.incoming_edges]

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
            yellow_phase = current_phase + 1
            self.conn.trafficlight.setPhase(self.ts_id, yellow_phase)
            
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
        
        if self.conn is None:
            binary = "sumo-gui" if self.use_gui else "sumo"
            traci.start([binary, "-c", self.sumocfg_file, "--no-warnings", "--start", "--no-step-log"], label=self.label)
            self.conn = traci.getConnection(self.label)
        else:
            self.conn.load(["-c", self.sumocfg_file, "--no-warnings", "--start", "--no-step-log"])
        
        state, _ = self._get_state()
        return state, {}

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except traci.exceptions.FatalTraCIError:
                pass