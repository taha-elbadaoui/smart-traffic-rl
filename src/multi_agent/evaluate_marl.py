import os
import sys
import time
import argparse

# Configuration des chemins pour importer vos modules locaux, Walid
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
sys.path.append(SCRIPT_DIR)

from stable_baselines3 import PPO
from wrapper_multi import MultiAgentTrafficEnv
from train_marl import CentralizedTrafficEnvWrapper

CONFIG_PATH = os.path.join(ROOT_DIR, "envs", "boulevard_coordonne", "env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def main():
    # 1. Ajout d'arguments pour pouvoir couper l'interface graphique si besoin
    parser = argparse.ArgumentParser(description="MARL PPO Evaluation (GUI Mode)")
    parser.add_argument("--no_gui", action="store_true", help="Désactiver l'interface graphique sumo-gui")
    args = parser.parse_args()

    use_gui = not args.no_gui
    model_path = os.path.join(MODEL_DIR, "ppo_marl_vague_verte_final")

    print(f"=== Walid, initialisation du script d'évaluation visuelle ===")
    print(f"Mode Graphique (sumo-gui) : {'ACTIVÉ' if use_gui else 'DÉSACTIVÉ'}")

    # 2. Instanciation de l'environnement de base ciblant vos deux feux synchronisés
    base_marl_env = MultiAgentTrafficEnv(
        sumocfg_path=CONFIG_PATH,
        tls_ids=["B0", "C0"],  # Alignés sur la nomenclature netgenerate
        gui=use_gui
    )

    # 3. Emballage dans le même wrapper centralisé qu'à l'entraînement
    env = CentralizedTrafficEnvWrapper(base_marl_env)

    print(f"Chargement du cerveau de l'IA depuis : {model_path}.zip")
    try:
        # Chargement du modèle PPO entraîné
        model = PPO.load(model_path, env=env)
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Impossible de charger le modèle. Vérifie qu'il a bien été entraîné. Erreur : {e}")
        env.close()
        return

    # 4. Lancement de la simulation de test
    obs, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    print("\n🚥 Lancement du trafic sur le boulevard... Regarde la synchronisation !")
    while not done:
        # Détermination de l'action de manière déterministe (Pas d'exploration aléatoire ici, Walid)
        action, _ = model.predict(obs, deterministic=True)

        # Exécution du pas de temps dans l'environnement unifié
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Affichage dynamique des décisions en direct pour vous aider à analyser
        phase_b0 = "VERT Principal" if action[0] == 0 else "VERT Secondaire"
        phase_c0 = "VERT Principal" if action[1] == 0 else "VERT Secondaire"
        
        print(f"Pas {step_count:4d} | Phase B0: {phase_b0:<15} | Phase C0: {phase_c0:<15} | Score: {reward:8.2f}")

        # Ralentisseur pour vous laisser le temps d'observer le déplacement des voitures sur l'écran
        if use_gui:
            time.sleep(0.05)

        done = terminated or truncated

    print(f"\n=== Fin de la simulation d'évaluation, Walid ! ===")
    print(f"Durée totale : {step_count} secondes de simulation.")
    print(f"Pénalité totale d'attente (Collaborative) : {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()

