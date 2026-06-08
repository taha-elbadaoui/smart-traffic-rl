import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Configuration des chemins
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
    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env
        self.tls_ids = base_env.tls_ids
        
        # 14 features (4 queues + 2 phase + 1 duration) * 2 feux
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * len(self.tls_ids))
        
    def _flatten_obs(self, states_dict):
        return np.concatenate([states_dict[tls_id] for tls_id in self.tls_ids])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        states_dict = self.base_env.reset()
        return self._flatten_obs(states_dict), {}

    def step(self, action_array):
        action_dict = {self.tls_ids[i]: action_array[i] for i in range(len(self.tls_ids))}
        next_states_dict, rewards_dict, terminateds, info = self.base_env.step(action_dict)
        
        obs = self._flatten_obs(next_states_dict)
        total_reward = float(sum(rewards_dict.values()))
        terminated = terminateds["__all__"]
        truncated = False
        
        return obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

def make_env(rank):
    def _init():
        env = MultiAgentTrafficEnv(sumocfg_path=CONFIG_PATH, tls_ids=["B0", "C0"], gui=False)
        # Note: Si ton wrapper supporte le rank, garde-le. Sinon, le port est géré par la config SUMO
        env.rank = rank 
        return CentralizedTrafficEnvWrapper(env)
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel CTCE Training for Boulevard")
    parser.add_argument("--num_cpu", type=int, default=4, help="Nombre de processus parallèles")
    args = parser.parse_args()

    print(f"--- Running CTCE Training across {args.num_cpu} CPUs ---")
    
    # 1. Parallélisation
    raw_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
    
    # 2. Normalisation
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)

    # 3. Callback sauvegarde (ajusté pour le nombre de CPU)
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // args.num_cpu, 
        save_path=MODEL_DIR, 
        name_prefix="ppo_centralized_boulevard_ckpt"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=LOG_DIR,
        device="cpu"
    )

    print("[INFO] Lancement de l'apprentissage parallèle...")
    model.learn(
        total_timesteps=2_000_000,
        progress_bar=True,
        tb_log_name="ppo_centralized_boulevard_parallel",
        callback=checkpoint_cb,
    )

    final_model_path = os.path.join(MODEL_DIR, "ppo_centralized_final_parallel")
    model.save(final_model_path)
    env.save(os.path.join(MODEL_DIR, "ppo_centralized_vecnorm_parallel.pkl"))
    
    print("[SUCCESS] Modèle et normaliseur sauvegardés.")
    env.close()