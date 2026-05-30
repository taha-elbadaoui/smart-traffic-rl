import time
import argparse
import os
from stable_baselines3 import PPO                                   # Fixed: was DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

parser = argparse.ArgumentParser(description="Crossroad PPO Evaluation (GUI Mode)")
parser.add_argument(
    "--mode",
    type=str,
    choices=["final", "random"],
    default="final",
    help="'final' = trained PPO agent; 'random' = untrained baseline for comparison",
)
args = parser.parse_args()

# Fixed: model prefix is now ppo_crossroad_*, not dqn_crossroad_*
model_name = f"ppo_crossroad_{args.mode}"
model_path = os.path.join(MODEL_DIR, model_name)
norm_path  = os.path.join(MODEL_DIR, "ppo_crossroad_vecnorm.pkl")

print(f"Loading model: {model_name}")

# Wrap in DummyVecEnv so VecNormalize works for single-env evaluation
raw_env = DummyVecEnv([lambda: CrossroadEnv(CONFIG_PATH, use_gui=True)])

if args.mode == "final" and os.path.exists(norm_path):
    # Restore saved normalisation stats to match the training distribution
    env = VecNormalize.load(norm_path, raw_env)
    env.training = False      # Freeze stats — do not update during evaluation
    env.norm_reward = False   # Show real (un-normalised) rewards in terminal output
    print(f"VecNormalize stats loaded from {norm_path}")
else:
    env = raw_env
    if args.mode == "final":
        print("Warning: normalizer file not found — running without obs normalisation.")
        print("         Evaluation rewards may differ from training-time values.")

try:
    model = PPO.load(model_path, env=env)   # Fixed: was DQN.load
    print(f"✅ Model loaded from {model_path}.zip")
except Exception as e:
    print(f"❌ Could not load model at {os.path.abspath(model_path)}.zip")
    print(f"   Error: {e}")
    print("   Did you run train_crossroad.py --mode train first?")
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