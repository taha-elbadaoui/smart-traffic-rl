import os
import argparse
import torch
from stable_baselines3 import PPO  # CRITICAL: Swapped out DQN for PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper_crossroad import CrossroadEnv 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hardware Acceleration Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on: {device.upper()} ---")

def make_env(rank, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        env = CrossroadEnv(CONFIG_PATH, use_gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel PPO Training for Crossroad Traffic Management")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    # Highly Optimized for an 8-core CPU setup
    parser.add_argument("--num_cpu", type=int, default=8, help="Number of parallel environments")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"🚀 Lancement de l'entraînement PPO sur {args.num_cpu} environnements...")
        
        # Gathering synchronous rollouts across 8 instances safely
        env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
        
        # PPO Hyperparameter Configuration for Complex Controls
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4,     # Standard stable LR for PPO
            n_steps=1024,           # Step budget collected per worker per update (8 * 1024 = 8192 total batch steps)
            batch_size=64,          # Mini-batch size for surrogate loss gradient descent
            n_epochs=10,            # Number of optimization passes over the collected experience batch
            gamma=0.99,             # MDP discount factor for long-term congestion horizon
            gae_lambda=0.95,        # Factor for trade-off of bias vs variance for GAE
            clip_range=0.2,         # Bound to limit policy churn step updates
            ent_coef=0.01,          # Entropy coefficient to maintain slight exploratory actions
            tensorboard_log=LOG_DIR,
            device="cpu"
        )
        
        # Training loop execution
        model.learn(total_timesteps=1000000, progress_bar=True, tb_log_name="ppo_crossroad")
        model_name = "ppo_crossroad_parallel"
    else:
        print("🎲 Création d'un modèle aléatoire...")
        env = CrossroadEnv(CONFIG_PATH, use_gui=False)
        model = PPO("MlpPolicy", env, verbose=1, device=device)
        model_name = "ppo_crossroad_random"

    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"✅ Modèle final enregistré : {model_path}")

    env.close()