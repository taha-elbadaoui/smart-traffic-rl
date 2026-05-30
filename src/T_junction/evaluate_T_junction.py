import time
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wrapper_T_junction import TJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/T_junction/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

parser = argparse.ArgumentParser(description="T-Junction PPO Evaluation (GUI Mode)")
parser.add_argument(
    "--mode",
    type=str,
    choices=["final", "random"],
    default="final",
    help="'final' loads the trained model; 'random' loads the untrained baseline",
)
args = parser.parse_args()

model_name = f"ppo_t_junction_{args.mode}"
model_path = os.path.join(MODEL_DIR, model_name)
norm_path  = os.path.join(MODEL_DIR, "ppo_t_junction_vecnorm.pkl")

print(f"Initialising T-Junction environment (GUI enabled)...")

# Wrap in DummyVecEnv so VecNormalize can be applied for single-env evaluation
raw_env = DummyVecEnv([lambda: TJunctionEnv(CONFIG_PATH, use_gui=True)])

if args.mode == "final" and os.path.exists(norm_path):
    # Load saved normalisation stats so reward/obs scaling matches training
    env = VecNormalize.load(norm_path, raw_env)
    env.training = False        # Freeze normalizer — do not update stats during eval
    env.norm_reward = False     # Show real reward values in the output
    print(f"VecNormalize stats loaded from {norm_path}")
else:
    env = raw_env
    if args.mode == "final":
        print("Warning: normalizer file not found — running without obs normalisation.")

try:
    model = PPO.load(model_path, env=env)
    print(f"✅ Model '{model_name}' loaded from {model_path}.zip")
except Exception as e:
    print(f"❌ Could not load model at {os.path.abspath(model_path)}.zip")
    print(f"   Error: {e}")
    print("   Did you run train_T_junction.py --mode train first?")
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

    phase_label = "East→straight Green" if int(action[0]) == 0 else "North→straight Green"
    print(f"Step {step_count:4d} | Phase: {phase_label:<24} | "
          f"Step reward: {float(reward[0]):8.2f} | Total: {total_reward:10.2f}")

    time.sleep(0.1)

print(f"\nEpisode finished — {step_count} steps, total reward: {total_reward:.2f}")
env.close()