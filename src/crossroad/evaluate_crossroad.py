import time
import argparse
import os
from stable_baselines3 import DQN
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

parser = argparse.ArgumentParser(description="Crossroad DQN Evaluation")
parser.add_argument(
    "--mode",
    type=str,
    # "trained" = your actual DQN agent
    # "random"  = untrained baseline (acts randomly, for comparison)
    choices=["trained", "random"],
    default="trained"
)
args = parser.parse_args()

model_name = f"dqn_crossroad_{args.mode}"
model_path = os.path.join(MODEL_DIR, model_name)

print(f"Loading model: {model_name}")
env = CrossroadEnv(CONFIG_PATH, use_gui=True)

try:
    model = DQN.load(model_path, env=env)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Could not load model at {os.path.abspath(model_path)}")
    print(f"Error: {e}")
    print("Did you run train_crossroad.py --mode train first?")
    env.close()
    exit()

obs, info = env.reset()
done = False
total_reward = 0
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    phase_label = "N/S Green" if action == 0 else "E/W Green"
    print(f"Step {step_count:4d} | Phase: {phase_label} | Step reward: {reward:8.2f} | Total: {total_reward:10.2f}")

    time.sleep(0.1)

print(f"\nEpisode finished — {step_count} steps, total reward: {total_reward:.2f}")
env.close()