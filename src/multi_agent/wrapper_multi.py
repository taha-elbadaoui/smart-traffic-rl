import os
import sys
import numpy as np

# Gestion des imports dynamiques de TraCI et Libsumo, Walid
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Veuillez déclarer la variable d'environnement 'SUMO_HOME'")

import traci

LIBSUMO_AVAILABLE = False
try:
    import libsumo
    LIBSUMO_AVAILABLE = True
except ImportError:
    pass

class MultiAgentTrafficEnv:
    def __init__(self, sumocfg_path, tls_ids=["1_0", "2_0"], gui=False):
        self.sumocfg_path = sumocfg_path
        self.tls_ids = tls_ids
        self.gui = gui
        
        # Mode de connexion : on force libsumo si l'interface graphique est coupée (mode entraînement)
        self.use_libsumo = (not self.gui) and LIBSUMO_AVAILABLE
        self.conn = None
        
    def reset(self):
        sumo_args = ["-c", self.sumocfg_path, "--no-warnings", "--no-step-log", "--random"]
        
        # Solution magique pour bypasser l'erreur de sécurité Windows :
        if self.use_libsumo:
            libsumo.start(["sumo"] + sumo_args)
            self.conn = libsumo
            print("--> Connexion établie via LIBSUMO (Bypass du blocage système Windows)")
        else:
            binary = "sumo-gui" if self.gui else "sumo"
            traci.start([binary] + sumo_args)
            self.conn = traci
            print("--> Connexion établie via TRACI")
        
        states = {}
        for tls_id in self.tls_ids:
            states[tls_id] = self._get_state(tls_id)
        return states

    def _get_state(self, tls_id):
        # Utilisation dynamique de l'API active (traci ou libsumo) via self.conn
        controlled_lanes = list(set(self.conn.trafficlight.getControlledLanes(tls_id)))
        queue_lengths = []
        for lane in controlled_lanes:
            halted_vehicles = self.conn.lane.getLastStepHaltingNumber(lane)
            queue_lengths.append(halted_vehicles)
        return np.array(queue_lengths, dtype=np.float32)

    def step(self, action_dict):
        for tls_id, action in action_dict.items():
            phase_index = int(action) * 2
            self.conn.trafficlight.setPhase(tls_id, phase_index)
            
        # Avancement de la simulation physique
        for _ in range(5):
            self.conn.simulationStep()
            
        next_states = {}
        rewards = {}
        terminateds = {"__all__": False}
        
        for tls_id in self.tls_ids:
            next_states[tls_id] = self._get_state(tls_id)
            rewards[tls_id] = self._compute_reward(tls_id)
            
        if self.conn.simulation.getMinExpectedNumber() <= 0:
            terminateds["__all__"] = True
            self.conn.close()
            
        return next_states, rewards, terminateds, {}

    def _compute_reward(self, tls_id):
        controlled_lanes = list(set(self.conn.trafficlight.getControlledLanes(tls_id)))
        total_waiting_time = 0
        for lane in controlled_lanes:
            total_waiting_time += self.conn.lane.getWaitingTime(lane)
        return -total_waiting_time

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
