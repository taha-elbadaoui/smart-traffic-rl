import os
import argparse
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper_T_junction import TJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/t_junction/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Hardware Detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on: {device.upper()} ---")

def make_env(rank, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        # Using absolute path is safer for subprocesses
        env = TJunctionEnv(CONFIG_PATH, use_gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Training for T-Junction Management")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    parser.add_argument("--num_cpu", type=int, default=8, help="Number of parallel threads")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"🚀 Lancement de l'entraînement DQN sur {args.num_cpu} environnements...")
        
        # Parallelize the 200,000 steps across multiple cores
        env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
        
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=1e-3, 
            buffer_size=50000, 
            exploration_fraction=0.5,
            device=device
        )
        
        model.learn(total_timesteps=200000, progress_bar=True)
        model_name = "dqn_t_junction_parallel"
    else:
        print("🎲 Creating an UNTRAINED random model...")
        env = TJunctionEnv(CONFIG_PATH, use_gui=False)
        model = DQN("MlpPolicy", env, verbose=1, device=device)
        model_name = "dqn_t_junction_random"

    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"✅ Training Complete. Final model saved at: {model_path}")

    env.close()