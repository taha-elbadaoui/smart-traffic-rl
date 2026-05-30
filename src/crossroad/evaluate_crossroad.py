import time
import argparse
import os
from stable_baselines3 import PPO                                   
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

parser = argparse.ArgumentParser(description="Crossroad PPO Evaluation (GUI Mode)")
# Upgraded: Added explicit choices to select between your 500k and 2M checkpoints cleanly
parser.add_argument(
    "--mode",
    type=str,
    choices=["500k", "2M", "random"],
    default="2M",
    help="Select which model type to evaluate in the visual GUI simulation",
)
args = parser.parse_args()

# Dynamically resolve file names depending on selected model scale
if args.mode == "500k":
    model_name = "ppo_crossroad_500k_final"
    norm_name = "ppo_crossroad_500k_vecnorm.pkl"
elif args.mode == "2M":
    model_name = "ppo_crossroad_2M_final"
    norm_name = "ppo_crossroad_2M_vecnorm.pkl"
else:
    model_name = "ppo_crossroad_random"
    norm_name = None

model_path = os.path.join(MODEL_DIR, model_name)
norm_path  = os.path.join(MODEL_DIR, norm_name) if norm_name else None

print(f"Loading evaluation target model configuration: {model_name}")

raw_env = DummyVecEnv([lambda: CrossroadEnv(CONFIG_PATH, use_gui=True)])

if args.mode != "random" and norm_path and os.path.exists(norm_path):
    env = VecNormalize.load(norm_path, raw_env)
    env.training = False      
    env.norm_reward = False   
    print(f"VecNormalize stats loaded successfully from {norm_path}")
else:
    env = raw_env
    if args.mode != "random":
        print("Warning: normalizer file not found — running without obs normalisation.")

try:
    model = PPO.load(model_path, env=env)   
    print(f"✅ Model loaded from {model_path}.zip")
except Exception as e:
    print(f"❌ Could not load model at {os.path.abspath(model_path)}.zip")
    print(f"   Error: {e}")
    env.close()
    exit(1)

obs = env.reset()
done = False
total_reward = 0.0
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += float(reward[0])
    step_count += 1

    phase_label = "N/S Green" if int(action[0]) == 0 else "E/W Green"
    print(f"Step {step_count:4d} | Phase: {phase_label:<10} | "
          f"Step reward: {float(reward[0]):8.2f} | Total: {total_reward:10.2f}")

    time.sleep(0.1)

print(f"\nEpisode finished — {step_count} steps, total reward: {total_reward:.2f}")
env.close()