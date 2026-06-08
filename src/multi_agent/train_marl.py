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
    Objectif : Convertir l'environnement Multi-Agent dictionnaire en un environnement 
    Gymnasium standard compatible à 100% avec Stable-Baselines3.
    """
    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env
        self.tls_ids = base_env.tls_ids
        
        print(f"Walid, initialisation du pont Gymnasium pour les feux : {self.tls_ids}")
        
        # On lance un reset d'essai pour détecter dynamiquement la taille des états de SUMO
        initial_states = self.base_env.reset()
        
        # Calcul de la dimension totale en combinant les capteurs de chaque carrefour
        total_obs_dim = sum([len(state) for state in initial_states.values()])
        
        # Espace d'observation unifié
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # Actions MultiDiscrete : [2, 2] car 2 feux ayant chacun 2 actions possibles (0 ou 1)
        self.action_space = spaces.MultiDiscrete([2] * len(self.tls_ids))
        
    def _flatten_obs(self, states_dict):
        # Interprétation : On fusionne les vecteurs de files d'attente de J1 et J2 en un seul vecteur plat
        return np.concatenate([states_dict[tls_id] for tls_id in self.tls_ids])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        states_dict = self.base_env.reset()
        return self._flatten_obs(states_dict), {}

    def step(self, action_array):
        # On convertit le tableau d'actions de l'IA [a1, a2] en dictionnaire {"1_0": a1, "2_0": a2}
        action_dict = {self.tls_ids[i]: action_array[i] for i in range(len(self.tls_ids))}
        
        # Envoi des actions simultanées à SUMO
        next_states_dict, rewards_dict, terminateds, info = self.base_env.step(action_dict)
        
        # Aplatissement du nouvel état combiné
        obs = self._flatten_obs(next_states_dict)
        
        # Récompense collective : Somme des récompenses pour forcer la coopération pour la vague verte !
        total_reward = float(sum(rewards_dict.values()))
        
        terminated = terminateds["__all__"]
        truncated = False
        
        return obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()


if __name__ == "__main__":
    print("--- Running MARL Training Optimized Context on: CPU ---")

    # 1. Instanciation de l'environnement de base SUMO
    base_marl_env = MultiAgentTrafficEnv(
        sumocfg_path=CONFIG_PATH,
        tls_ids=["1_0", "2_0"],  # ID par défaut générés par netgenerate
        gui=False
    )

    # 2. Emballage dans le wrapper centralisé pour Stable-Baselines3
    env = CentralizedTrafficEnvWrapper(base_marl_env)

    # 3. Configuration des sauvegardes automatiques pendant l'entraînement, Walid
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_marl_vague_verte_ckpt",
        verbose=1,
    )

    # 4. Initialisation du modèle PPO avec tes hyperparamètres favoris
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,      
        batch_size=64,    
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        tensorboard_log=LOG_DIR,
        device="cpu",  
    )

    # 5. Lancement de l'apprentissage sur 200 000 pas de temps
    print(f"🚀 Walid, le modèle PPO commence à apprendre la synchronisation des feux...")
    model.learn(
        total_timesteps=200_000,
        progress_bar=True,
        tb_log_name="ppo_marl_vague_verte",
        callback=checkpoint_cb,
    )

    # 6. Sauvegarde finale du modèle entraîné
    final_model_path = os.path.join(MODEL_DIR, "ppo_marl_vague_verte_final")
    model.save(final_model_path)
    print(f"✅ Modèle d'IA collaboratif sauvegardé avec succès : {final_model_path}.zip")
    
    env.close()
