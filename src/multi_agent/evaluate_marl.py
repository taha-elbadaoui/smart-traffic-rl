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
    # 1. Ajout d'arguments pour pouvoir couper l'interface graphique si besoin, Walid
    parser = argparse.ArgumentParser(description="MARL PPO Evaluation (GUI Mode)")
    parser.add_argument("--no_gui", action="store_true", help="Desactiver l'interface graphique sumo-gui")
    args = parser.parse_args()

    use_gui = not args.no_gui
    model_path = os.path.join(MODEL_DIR, "ppo_marl_vague_verte_final")

    print(f"=== Walid, initialisation du script d'evaluation visuelle ===")
    print(f"Mode Graphic (sumo-gui) : {'ACTIVE' if use_gui else 'DESACTIVE'}")

    # 2. Instanciation de l'environnement de base ciblant vos deux feux synchronises
    base_marl_env = MultiAgentTrafficEnv(
        sumocfg_path=CONFIG_PATH,
        tls_ids=["B0", "C0"],  # Alignes sur la nomenclature netgenerate
        gui=use_gui
    )

    # 3. Emballage dans le même wrapper centralise qu'a l'entrainement
    env = CentralizedTrafficEnvWrapper(base_marl_env)

    print(f"Chargement du cerveau de l'IA depuis : {model_path}.zip")
    try:
        # Chargement du modele PPO entraine
        model = PPO.load(model_path, env=env)
        print("[SUCCESS] Modele charge avec succes !")
    except Exception as e:
        print(f"[ERROR] Impossible de charger le modele. Verifie qu'il a bien ete entraine. Erreur : {e}")
        env.close()
        return

    # 4. Lancement de la simulation de test
    obs, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    print("\n[INFO] Lancement du trafic sur le boulevard... Regarde la synchronisation !")
    while not done:
        # Determination de l'action de maniere deterministe (Pas d'exploration aleatoire ici, Walid)
        action, _ = model.predict(obs, deterministic=True)

        # Execution du pas de temps dans l'environnement unifie
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Affichage dynamique des decisions en direct pour vous aider a analyser, Walid
        phase_b0 = "VERT Principal" if action[0] == 0 else "VERT Secondaire"
        phase_c0 = "VERT Principal" if action[1] == 0 else "VERT Secondaire"
        
        print(f"Pas {step_count:4d} | Phase B0: {phase_b0:<15} | Phase C0: {phase_c0:<15} | Score: {reward:8.2f}")

        # Ralentisseur pour vous laisser le temps d'observer le deplacement des voitures sur l'ecran
        if use_gui:
            time.sleep(0.05)

        done = terminated or truncated

    print(f"\n=== Fin de la simulation d'evaluation, Walid ! ===")
    print(f"Duree totale : {step_count} secondes de simulation.")
    print(f"Penalite totale d'attente (Collaborative) : {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
