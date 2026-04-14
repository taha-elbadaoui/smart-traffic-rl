import os
import argparse
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper_crossroad import CrossroadEnv 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Hardware Detection for your G14
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on: {device.upper()} ---")

def make_env(rank, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        # Using absolute path is safer for parallel subprocesses
        env = CrossroadEnv(CONFIG_PATH, use_gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Training for Crossroad Traffic Management")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    # Recommended: 8 logical cores for your G14
    parser.add_argument("--num_cpu", type=int, default=8, help="Number of parallel environments")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"🚀 Lancement de l'entraînement DQN sur {args.num_cpu} environnements...")
        
        # Parallelize to increase your it/s
        env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
        
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=1e-3, 
            buffer_size=50000, 
            # REDUCED: Force the agent to start using its brain after 10% of training
            exploration_fraction=0.1, 
            device=device
        )
        
        # INCREASED: 1,000,000 steps to allow the agent to learn the benefit of switching
        model.learn(total_timesteps=1000000, progress_bar=True)
        model_name = "dqn_crossroad_parallel"
    else:
        print("🎲 Création d'un modèle aléatoire...")
        env = CrossroadEnv(CONFIG_PATH, use_gui=False)
        model = DQN("MlpPolicy", env, verbose=1, device=device)
        model_name = "dqn_crossroad_random"

    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"✅ Modèle final enregistré : {model_path}")

    env.close()