import os
import sys
import numpy as np

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


YELLOW_PHASE_MAP = {0: 1, 2: 3}

class MultiAgentTrafficEnv:
    def __init__(self, sumocfg_path, tls_ids=["B0", "C0"], gui=False):
        self.sumocfg_path = sumocfg_path
        self.tls_ids = tls_ids
        self.gui = gui
        
        
        self.use_libsumo = (not self.gui) and LIBSUMO_AVAILABLE
        self.conn = None
        
        self.max_cars = 25.0
        self.yellow_time = 3
        self.step_length = 25
        self.green_time = self.step_length - self.yellow_time
        
        self.min_green_time = 15
        self.max_green_time = 60.0
        
        self.phase_durations = {tls_id: 0 for tls_id in self.tls_ids}
        
    def reset(self):
        sumo_args = ["-c", self.sumocfg_path, "--no-warnings", "--no-step-log", "--random"]
        
        if self.use_libsumo:
            libsumo.start(["sumo"] + sumo_args)
            self.conn = libsumo
        else:
            binary = "sumo-gui" if self.gui else "sumo"
            traci.start([binary] + sumo_args)
            self.conn = traci
        
        self.phase_durations = {tls_id: 0 for tls_id in self.tls_ids}
        
        states = {}
        for tls_id in self.tls_ids:
            states[tls_id] = self._get_state(tls_id)
        return states

    def _get_queue_lengths(self, tls_id):
        controlled_lanes = list(set(self.conn.trafficlight.getControlledLanes(tls_id)))
        return [self.conn.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]

    def _get_state(self, tls_id):
        queues = self._get_queue_lengths(tls_id)
        norm_queues = [float(np.tanh(q / self.max_cars)) for q in queues]
        
        current_phase = self.conn.trafficlight.getPhase(tls_id)
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]
        
        norm_duration = [min(1.0, self.phase_durations[tls_id] / self.max_green_time)]
        return np.array(norm_queues + phase_one_hot + norm_duration, dtype=np.float32)

    def step(self, action_dict):
        target_phases = {}
        current_phases = {}
        guardrail_penalties = {tls_id: 0.0 for tls_id in self.tls_ids}
        switching_penalties = {tls_id: 0.0 for tls_id in self.tls_ids}
        
        # Validation des actions et application des filets de sécurité mathématiques
        for tls_id in self.tls_ids:
            target = int(action_dict[tls_id]) * 2
            curr = self.conn.trafficlight.getPhase(tls_id)
            if curr not in YELLOW_PHASE_MAP:
                curr = 0 if curr == 1 else 2
            current_phases[tls_id] = curr

            # Empêcher le clignotement rapide
            if target != curr and self.phase_durations[tls_id] < self.min_green_time:
                target = curr
                guardrail_penalties[tls_id] -= 5.0

            # Forcer le changement si le feu est resté vert trop longtemps
            if target == curr and self.phase_durations[tls_id] >= self.max_green_time:
                target = 2 if curr == 0 else 0
                guardrail_penalties[tls_id] -= 10.0
                
            target_phases[tls_id] = target

        # --- Exécution Physique ---
        accumulated_rewards = {tls_id: 0.0 for tls_id in self.tls_ids}
        throughput = 0

        # Gestion de la transition orange
        needs_yellow = any(current_phases[t] != target_phases[t] for t in self.tls_ids)
        if needs_yellow:
            for tls_id in self.tls_ids:
                if current_phases[tls_id] != target_phases[tls_id]:
                    switching_penalties[tls_id] -= 1.0
                    self.conn.trafficlight.setPhase(tls_id, YELLOW_PHASE_MAP[current_phases[tls_id]])
                else:
                    self.conn.trafficlight.setPhase(tls_id, current_phases[tls_id])
                    
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
                throughput += self.conn.simulation.getArrivedNumber()
                for tls_id in self.tls_ids:
                    accumulated_rewards[tls_id] += self._compute_base_penalty(tls_id)

        # Application du feu vert cible
        for tls_id in self.tls_ids:
            self.conn.trafficlight.setPhase(tls_id, target_phases[tls_id])
            if current_phases[tls_id] != target_phases[tls_id]:
                self.phase_durations[tls_id] = self.green_time
            else:
                self.phase_durations[tls_id] += self.step_length if not needs_yellow else self.green_time

        run_time = self.green_time if needs_yellow else self.step_length
        for _ in range(run_time):
            self.conn.simulationStep()
            throughput += self.conn.simulation.getArrivedNumber()
            for tls_id in self.tls_ids:
                accumulated_rewards[tls_id] += self._compute_base_penalty(tls_id)

        # --- Synthèse des états et récompenses ---
        next_states = {}
        final_rewards = {}
        for tls_id in self.tls_ids:
            next_states[tls_id] = self._get_state(tls_id)
            # Récompense composite : Pénalité d'attente + Débit global + Garde-fous
            final_rewards[tls_id] = float(
                (accumulated_rewards[tls_id] / self.step_length)
                + (throughput * 1.5) # Le throughput profite aux deux agents collaboratifs
                + switching_penalties[tls_id]
                + guardrail_penalties[tls_id]
            )
            
        terminateds = {"__all__": self.conn.simulation.getMinExpectedNumber() <= 0}
        if terminateds["__all__"]:
            self.conn.close()
            
        return next_states, final_rewards, terminateds, {}

    def _compute_base_penalty(self, tls_id):
        queues = self._get_queue_lengths(tls_id)
        if not queues:
            return 0.0
            
        total_waiting = sum(queues)
        max_queue = max(queues)
        
        base_penalty = -(total_waiting / self.max_cars)
        starvation_penalty = -(max_queue / self.max_cars)
        imbalance_penalty = -(float(np.std(queues)) / self.max_cars)

        return (1.0 * base_penalty) + (2.0 * starvation_penalty) + (0.5 * imbalance_penalty)

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None