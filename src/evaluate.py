import time
import argparse
import os
from stable_baselines3 import DQN
from wrapper import SingleIntersectionEnv

# Configuration des chemins
CONFIG_PATH = "../envs/1x1_minimal/env.sumocfg"
MODEL_DIR = "../models/"

# 1. Ajout du sélecteur de mode
parser = argparse.ArgumentParser(description="Évaluation du trafic (Entraîné vs Aléatoire)")
parser.add_argument("--mode", type=str, choices=["train", "random"], default="train",
                    help="Choisir 'train' pour l'IA entraînée ou 'random' pour le modèle BS")
args = parser.parse_args()

# 2. Détermination du modèle à charger
if args.mode == "train":
    model_name = "dqn_1x1_baseline"
    print("🚀 Chargement du modèle ENTRAÎNÉ...")
else:
    model_name = "dqn_untrained_random"
    print("🎲 Chargement du modèle ALÉATOIRE (Mode BS)...")

model_path = os.path.join(MODEL_DIR, model_name)

# 3. Initialisation de l'environnement avec GUI
env = SingleIntersectionEnv(CONFIG_PATH, use_gui=True)

try:
    model = DQN.load(model_path, env=env)
except Exception as e:
    print(f"❌ Erreur : Impossible de trouver '{model_name}'. Lance d'abord 'python train.py --mode {args.mode}'.")
    env.close()
    exit()

# 4. Boucle d'exécution
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # L'agent prédit l'action (en mode déterministe pour l'évaluation)
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"[{args.mode.upper()}] Action: {'N/S Green' if action == 0 else 'E/W Green'} | Reward: {reward}")
    
    # Pause pour permettre l'observation visuelle dans sumo-gui
    time.sleep(0.1)

print(f"Évaluation terminée. Récompense totale accumulée : {total_reward}")
env.close()
