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

# Correspondance pour associer une phase verte à sa phase jaune de transition
YELLOW_PHASE_MAP = {0: 1, 2: 3}

class MultiAgentTrafficEnv:
    def __init__(self, sumocfg_path, tls_ids=["B0", "C0"], gui=False):
        self.sumocfg_path = sumocfg_path
        self.tls_ids = tls_ids
        self.gui = gui
        
        self.use_libsumo = (not self.gui) and LIBSUMO_AVAILABLE
        self.conn = None
        
        # Paramètres temporels alignés sur vos baselines monocibles
        self.yellow_time = 3
        self.step_length = 25
        self.green_time = self.step_length - self.yellow_time
        self.max_green_time = 60.0
        
        # Suivi temporel interne par intersection
        self.phase_durations = {tls_id: 0 for tls_id in self.tls_ids}
        
    def reset(self):
        sumo_args = ["-c", self.sumocfg_path, "--no-warnings", "--no-step-log", "--random"]
        
        if self.use_libsumo:
            libsumo.start(["sumo"] + sumo_args)
            self.conn = libsumo
            print("--> Connexion établie via LIBSUMO")
        else:
            binary = "sumo-gui" if self.gui else "sumo"
            traci.start([binary] + sumo_args)
            self.conn = traci
            print("--> Connexion établie via TRACI")
        
        # Réinitialisation des compteurs temporels
        self.phase_durations = {tls_id: 0 for tls_id in self.tls_ids}
        
        states = {}
        for tls_id in self.tls_ids:
            states[tls_id] = self._get_state(tls_id)
        return states

    def _get_state(self, tls_id):
        controlled_lanes = list(set(self.conn.trafficlight.getControlledLanes(tls_id)))
        
        # 1. Collecte et normalisation des files d'attente (max_cars arbitré à 25)
        max_cars = 25.0
        queues = [self.conn.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
        norm_queues = [float(np.tanh(q / max_cars)) for q in queues]
        
        # 2. Récupération et encodage de la phase actuelle (One-Hot)
        current_phase = self.conn.trafficlight.getPhase(tls_id)
        if current_phase not in YELLOW_PHASE_MAP:
            current_phase = 0 if current_phase == 1 else 2
        phase_one_hot = [1.0, 0.0] if current_phase == 0 else [0.0, 1.0]
        
        # 3. Normalisation de la durée écoulée dans cette phase
        norm_duration = [min(1.0, self.phase_durations[tls_id] / self.max_green_time)]
        
        # Combinaison : taille totale de l'état ajustable dynamiquement
        return np.array(norm_queues + phase_one_hot + norm_duration, dtype=np.float32)

    def step(self, action_dict):
        target_phases = {}
        current_phases = {}
        
        # Extraction des phases cibles demandées par le réseau de neurones
        for tls_id in self.tls_ids:
            target_phases[tls_id] = int(action_dict[tls_id]) * 2
            curr = self.conn.trafficlight.getPhase(tls_id)
            if curr not in YELLOW_PHASE_MAP:
                curr = 0 if curr == 1 else 2
            current_phases[tls_id] = curr

        # --- ÉTAPE 1 : Intercalation de la phase jaune pour la sécurité (3 secondes) ---
        for tls_id in self.tls_ids:
            if current_phases[tls_id] != target_phases[tls_id]:
                # Changement d'action détecté -> On active le feu orange associé
                yellow_phase = YELLOW_PHASE_MAP[current_phases[tls_id]]
                self.conn.trafficlight.setPhase(tls_id, yellow_phase)
            else:
                # Maintien de l'état -> On reste sur la phase verte en cours
                self.conn.trafficlight.setPhase(tls_id, current_phases[tls_id])
                
        # Avancement physique de la transition jaune dans SUMO
        for _ in range(self.yellow_time):
            self.conn.simulationStep()
            
        # --- ÉTAPE 2 : Application des phases vertes cibles (22 secondes) ---
        for tls_id in self.tls_ids:
            self.conn.trafficlight.setPhase(tls_id, target_phases[tls_id])
            if current_phases[tls_id] != target_phases[tls_id]:
                # Le feu a basculé : on réinitialise son compteur à la durée de base
                self.phase_durations[tls_id] = self.green_time
            else:
                # Le feu est resté stable : on incrémente la durée cumulée
                self.phase_durations[tls_id] += self.step_length

        # Avancement du reste du bloc de temps macroscopique
        for _ in range(self.green_time):
            self.conn.simulationStep()
            
        # --- ÉTAPE 3 : Collecte des nouveaux états et des récompenses ---
        next_states = {}
        rewards = {}
        for tls_id in self.tls_ids:
            next_states[tls_id] = self._get_state(tls_id)
            rewards[tls_id] = self._compute_reward(tls_id)
            
        terminateds = {"__all__": False}
        if self.conn.simulation.getMinExpectedNumber() <= 0:
            terminateds["__all__"] = True
            self.conn.close()
            
        return next_states, rewards, terminateds, {}

    def _compute_reward(self, tls_id):
        controlled_lanes = list(set(self.conn.trafficlight.getControlledLanes(tls_id)))
        max_cars = 25.0
        
        # Substitution de la fonction getWaitingTime brute par le calcul homogène des files d'attente
        queues = [self.conn.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
        total_waiting = sum(queues)
        
        # Pénalité de base basée sur l'engorgement des voies contrôlées
        base_penalty = -(total_waiting / max_cars)
        return float(base_penalty)

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
