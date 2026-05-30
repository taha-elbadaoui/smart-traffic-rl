"""
Generates and saves an untrained PPO baseline for the T-Junction environment.
Run from the project root or any directory — paths are resolved automatically.
"""
import os
from stable_baselines3 import PPO
from wrapper_T_junction import TJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/T_junction/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Initialising T-Junction environment...")
env = TJunctionEnv(CONFIG_PATH, use_gui=False)

print("Creating untrained (random) PPO model...")
model = PPO("MlpPolicy", env, verbose=0)

model_path = os.path.join(MODEL_DIR, "ppo_t_junction_random")
model.save(model_path)
print(f"✅ Untrained baseline saved to: {model_path}.zip")

env.close()