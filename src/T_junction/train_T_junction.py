import os
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from wrapper_T_junction import TJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/t_junction/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Training script for T-Junction Traffic Management")
parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
args = parser.parse_args()

print("Initializing T-Junction Environment...")
env = TJunctionEnv(CONFIG_PATH, use_gui=False)

check_env(env)

if args.mode == "train":
    print("🚀 Starting DQN Training (200,000 steps, final save only)...")
    
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3, 
        buffer_size=50000, 
        exploration_fraction=0.5
    )
    
    model.learn(total_timesteps=200000, progress_bar=True)
    model_name = "dqn_t_junction_final"
else:
    print("🎲 Creating an UNTRAINED random model...")
    model = DQN("MlpPolicy", env, verbose=1)
    model_name = "dqn_t_junction_random"

model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)
print(f"✅ Training Complete. Final model saved at: {model_path}")

env.close()