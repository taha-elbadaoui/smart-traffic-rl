import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from wrapper import SingleIntersectionEnv

# Define paths
CONFIG_PATH = "../envs/1x1_minimal/env.sumocfg"
MODEL_DIR = "../models"

print("Initializing Environment...")
env = SingleIntersectionEnv(CONFIG_PATH, use_gui=False)

# Validate the custom Gym environment before training
check_env(env)

print("Starting DQN Training (Headless Mode)...")
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.5)

# Train for 20,000 steps to start
model.learn(total_timesteps=200000, progress_bar=True)

# Save the trained brain
model_path = os.path.join(MODEL_DIR, "dqn_1x1_baseline")
model.save(model_path)
print(f"Training Complete. Model saved to {model_path}")

env.close()
