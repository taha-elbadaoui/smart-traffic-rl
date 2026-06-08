import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration des chemins pour importer votre wrapper multi-agent, Walid
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
sys.path.append(SCRIPT_DIR)

from wrapper_multi import MultiAgentTrafficEnv

CONFIG_PATH = os.path.join(ROOT_DIR, "envs", "boulevard_coordonne", "env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class CentralizedTrafficEnvWrapper(gym.Env):
    """
    Objectif : Pont d'encapsulation pour rendre l'environnement multi-agent 
    compatible à 100% avec Stable-Baselines3 sans double-connexion.
    """
    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env
        self.tls_ids = base_env.tls_ids
        
        print(f"Walid, initialisation du pont Gymnasium pour les feux : {self.tls_ids}")
        
        # Reset d'essai pour récupérer dynamiquement la taille des états de SUMO
        initial_states = self.base_env.reset()
        total_obs_dim = sum([len(state) for state in initial_states.values()])
        
        # 🔥 CRUCIAL : On ferme immédiatement la connexion d'essai pour laisser 
        # le champ libre au reset d'entraînement de Stable-Baselines3 !
        self.base_env.close()
        
        # Espace d'observation et d'actions combiné
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * len(self.tls_ids))
        
    def _flatten_obs(self, states_dict):
        return np.concatenate([states_dict[tls_id] for tls_id in self.tls_ids])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        states_dict = self.base_env.reset()
        return self._flatten_obs(states_dict), {}

    def step(self, action_array):
        # Traduction du vecteur d'actions vers le dictionnaire d'agents
        action_dict = {self.tls_ids[i]: action_array[i] for i in range(len(self.tls_ids))}
        next_states_dict, rewards_dict, terminateds, info = self.base_env.step(action_dict)
        
        obs = self._flatten_obs(next_states_dict)
        total_reward = float(sum(rewards_dict.values())) # Récompense collaborative
        terminated = terminateds["__all__"]
        truncated = False
        
        return obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()


if __name__ == "__main__":
    print("--- Running MARL Training Optimized Context on: CPU ---")

    # 1. Instanciation de l'environnement de base SUMO ciblant B0 et C0
    base_marl_env = MultiAgentTrafficEnv(
        sumocfg_path=CONFIG_PATH,
        tls_ids=["B0", "C0"],  
        gui=False
    )

    # 2. Emballage protecteur
    env = CentralizedTrafficEnvWrapper(base_marl_env)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_marl_vague_verte_ckpt",
        verbose=1,
    )

    # 3. Modèle PPO unifié pour la synchronisation
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,      
        batch_size=64,    
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=LOG_DIR,
        device="cpu",  
    )

    print(f"🚀 Lancement de l'apprentissage collaboratif...")
    model.learn(
        total_timesteps=100_000,
        progress_bar=True,
        tb_log_name="ppo_marl_vague_verte",
        callback=checkpoint_cb,
    )

    final_model_path = os.path.join(MODEL_DIR, "ppo_marl_vague_verte_final")
    model.save(final_model_path)
    print(f"✅ Modèle sauvegardé avec succès : {final_model_path}.zip")
    
    env.close()
