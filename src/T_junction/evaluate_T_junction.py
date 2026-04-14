import time
import argparse
import os
from stable_baselines3 import DQN
from wrapper_T_junction import TJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/t_junction/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

parser = argparse.ArgumentParser(description="Évaluation du trafic T-Junction")
parser.add_argument("--mode", type=str, choices=["final", "random"], default="final")
args = parser.parse_args()

model_name = f"dqn_t_junction_{args.mode}"
model_path = os.path.join(MODEL_DIR, model_name)

print(f"Initialisation de l'environnement T-Junction (GUI Activé)...")
env = TJunctionEnv(CONFIG_PATH, use_gui=True)

try:
    model = DQN.load(model_path, env=env)
    print(f"🚀 Modèle '{model_name}' chargé avec succès depuis {model_path} !")
except Exception as e:
    print(f"❌ Erreur : Impossible de charger le modèle à {os.path.abspath(model_path)}")
    env.close()
    exit()

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    direction = "West Green" if action == 0 else "North Green"
    print(f"[{args.mode.upper()}] Phase: {direction} | Reward: {reward}")
    
    time.sleep(0.1)

print(f"Évaluation terminée. Score total : {total_reward}")
env.close()