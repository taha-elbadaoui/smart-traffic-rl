import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Walid, l'objectif ici est d'utiliser uniquement 'traci' pour contourner le bug de DLL.
# On supprime l'import dynamique de libsumo.
import traci

class SingleIntersectionEnv(gym.Env):
    def __init__(self, sumocfg_file, use_gui=False):
        super(SingleIntersectionEnv, self).__init__()
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        
        # Walid, on assigne traci directement pour tous les modes (avec ou sans GUI)
        self.traci = traci
        
        # Walid, l'objectif est d'augmenter la capacité max à 30 
        # puisqu'il y a maintenant 2 voies par axe pouvant stocker des voitures.
        self.max_cars = 30.0 
        self.green_time = 5  
        
        # L'espace d'action reste 2 : 
        # Action 0 = Phase 0 (Vert N/S), Action 1 = Phase 2 (Vert E/W)
        self.action_space = spaces.Discrete(2)
        
        # L'objectif est d'avoir 4 capteurs virtuels (un pour chaque axe principal)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def _get_state(self):
        # L'objectif de traci.edge est de récupérer la somme des voitures à l'arrêt 
        # sur TOUTES les voies de l'axe spécifié.
        q_n = self.traci.edge.getLastStepHaltingNumber("in_N")
        q_e = self.traci.edge.getLastStepHaltingNumber("int_E")
        q_s = self.traci.edge.getLastStepHaltingNumber("in_S")
        q_w = self.traci.edge.getLastStepHaltingNumber("in_W")
        
        norm_q_n = min(q_n / self.max_cars, 1.0)
        norm_q_e = min(q_e / self.max_cars, 1.0)
        norm_q_s = min(q_s / self.max_cars, 1.0)
        norm_q_w = min(q_w / self.max_cars, 1.0)
        
        return np.array([norm_q_n, norm_q_e, norm_q_s, norm_q_w], dtype=np.float32), q_n, q_e, q_s, q_w

    def step(self, action):
        target_phase = int(action) * 2 
        
        # Walid, l'interprétation ici est cruciale : on cible le nouvel ID du feu, "J4"
        self.traci.trafficlight.setPhase("J4", target_phase)
        
        for _ in range(self.green_time):
            self.traci.simulationStep()
            
        state, q_n, q_e, q_s, q_w = self._get_state()
        
        # L'objectif de la récompense est de punir la congestion totale des 8 voies d'entrée
        reward = -float(q_n + q_e + q_s + q_w)
        done = self.traci.simulation.getMinExpectedNumber() <= 0
        
        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            self.traci.close()
        except:
            pass
            
        binary = "sumo-gui" if self.use_gui else "sumo"
        self.traci.start([binary, "-c", self.sumocfg_file, "--no-warnings", "--start"])
        
        initial_state, _, _, _, _ = self._get_state()
        return initial_state, {}

    def close(self):
        try:
            self.traci.close()
        except:
            pass