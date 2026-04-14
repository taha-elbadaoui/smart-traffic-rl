import os
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from wrapper_4x4 import SingleIntersectionEnv 

CONFIG_PATH = "envs/4x4_intersections/env.sumocfg"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Entraînement 4x4 pour Walid")
parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
args = parser.parse_args()

print(f"Walid, initialisation de l'environnement 4x4...")
env = SingleIntersectionEnv(CONFIG_PATH, use_gui=False)

# Vérification de la compatibilité Gymnasium
check_env(env)

if args.mode == "train":
    print("🚀 Lancement de l'entraînement DQN sur 4x4")
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.5)
    model.learn(total_timesteps=300000, progress_bar=True)
    model_name = "dqn_4x4_smart_traffic"
else:
    print("🎲 Création d'un modèle aléatoire pour test...")
    model = DQN("MlpPolicy", env, verbose=1)
    model_name = "dqn_4x4_random"

model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)
print(f"✅ Modèle enregistré : {model_path}")

env.close()



