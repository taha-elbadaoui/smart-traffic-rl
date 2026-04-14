import time
import argparse
import os
from stable_baselines3 import DQN
from wrapper_4x4 import SingleIntersectionEnv

# NOUVEAU CODE POUR WALID
CONFIG_PATH = "envs/4x4_intersections/env.sumocfg"
MODEL_DIR = "models"

parser = argparse.ArgumentParser(description="Évaluation 4x4 pour Walid")
parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
args = parser.parse_args()

model_name = "dqn_4x4_smart_traffic" if args.mode == "train" else "dqn_4x4_random"
model_path = os.path.join(MODEL_DIR, model_name)

# Walid, activation du GUI pour voir le résultat du Smart Traffic
env = SingleIntersectionEnv(CONFIG_PATH, use_gui=True)

try:
    model = DQN.load(model_path, env=env)
    print(f"🚀 Walid, modèle {model_name} chargé avec succès !")
except Exception as e:
    print(f"❌ Walid, impossible de trouver le modèle. Entraîne-le d'abord.")
    env.close()
    exit()

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Walid, affichage des logs pour suivre la réduction de la congestion
    direction = "N/S Vert" if action == 0 else "E/W Vert"
    print(f"[Walid - 4x4] Phase: {direction} | Reward: {reward}")
    
    time.sleep(0.1)

print(f"Walid, évaluation terminée. Score total : {total_reward}")
env.close()
