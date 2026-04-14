import time
import argparse
import os
from stable_baselines3 import DQN
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

parser = argparse.ArgumentParser(description="Évaluation DQN pour l'intersection simple")
parser.add_argument("--mode", type=str, choices=["final", "random", "parallel"], default="parallel")
args = parser.parse_args()

model_name = f"dqn_crossroad_{args.mode}"
model_path = os.path.join(MODEL_DIR, model_name)

print(f"Initialisation de l'environnement Crossroad (GUI Activé)...")
env = CrossroadEnv(CONFIG_PATH, use_gui=True)

try:
    model = DQN.load(model_path, env=env)
    print(f"🚀 Modèle '{model_name}' chargé avec succès depuis {model_path} !")
except Exception as e:
    print(f"❌ Impossible de charger le modèle à : {os.path.abspath(model_path)}")
    print(f"Détail de l'erreur : {e}")
    env.close()
    exit()

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    direction = "N/S Vert" if action == 0 else "E/W Vert"
    print(f"[Evaluation] Phase: {direction} | Step Reward: {reward}")
    
    time.sleep(0.1)

print(f"Évaluation terminée. Score total de l'épisode : {total_reward}")
env.close()