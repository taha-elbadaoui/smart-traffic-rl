import os
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from wrapper import SingleIntersectionEnv

# Configuration des chemins
CONFIG_PATH = "../envs/1x1_minimal/env.sumocfg"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Configuration de l'argument de choix
parser = argparse.ArgumentParser(description="Entraînement ou Initialisation du modèle RL")
parser.add_argument("--mode", type=str, choices=["train", "random"], default="train",
                    help="Choisir 'train' pour l'entraînement complet ou 'random' pour un modèle vierge")
args = parser.parse_args()

print("Initializing Environment...")
env = SingleIntersectionEnv(CONFIG_PATH, use_gui=False)

# Validation de l'environnement
check_env(env)

if args.mode == "train":
    # --- MODE ENTRAÎNEMENT ---
    print("🚀 Starting DQN Training (200,000 steps)...")
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.5)
    model.learn(total_timesteps=200000, progress_bar=True)
    model_name = "dqn_1x1_baseline"
else:
    # --- MODE ALÉATOIRE ---
    print("🎲 Creating an UNTRAINED random model...")
    model = DQN("MlpPolicy", env, verbose=1)
    # On saute model.learn() pour garder les poids aléatoires
    model_name = "dqn_untrained_random"

# Sauvegarde du modèle choisi
model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)
print(f"✅ Terminé. Modèle enregistré sous : {model_path}")

env.close()
