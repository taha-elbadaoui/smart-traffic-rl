import os
import sys
import time
import argparse
import warnings

# Couper les avertissements TensorFlow/SB3 pour une console propre
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
sys.path.append(SCRIPT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wrapper_multi import MultiAgentTrafficEnv
from train_marl import CentralizedTrafficEnvWrapper

CONFIG_PATH = os.path.join(ROOT_DIR, "envs", "boulevard_coordonne", "env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def main():
    parser = argparse.ArgumentParser(description="Centralized PPO Evaluation (Boulevard)")
    parser.add_argument("--no_gui", action="store_true", help="Désactiver l'interface graphique sumo-gui")
    args = parser.parse_args()

    use_gui = not args.no_gui
    model_path = os.path.join(MODEL_DIR, "ppo_boulevard_centralized_2M_final")
    norm_path = os.path.join(MODEL_DIR, "ppo_boulevard_centralized_2M_vecnorm.pkl")
    
    print(f"=== Évaluation Boulevard Centralisé (Mode Fluide / Vague Verte) ===")

    def make_env():
        return CentralizedTrafficEnvWrapper(
            MultiAgentTrafficEnv(sumocfg_path=CONFIG_PATH, tls_ids=["B0", "C0"], gui=use_gui)
        )

    raw_env = DummyVecEnv([make_env])
    
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, raw_env)
        env.training = False
        env.norm_reward = False
        print("[INFO] Fichier de normalisation chargé.")
    else:
        env = raw_env
        print("[WARNING] Normaliseur introuvable. L'évaluation risque d'être faussée.")

    try:
        model = PPO.load(model_path, env=env, device='cpu')
        print("[SUCCESS] Cerveau IA chargé.")
    except Exception as e:
        print(f"[ERROR] Impossible de charger le modèle : {e}")
        env.close()
        return

    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    print("\n[INFO] Démarrage... Appuyez sur Play dans SUMO-GUI pour observer la synchronisation.")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])
        step_count += 1

        # Affichage console épuré
        phase_b0 = "Vert Axe Principal" if action[0][0] == 0 else "Vert Axe Secondaire"
        phase_c0 = "Vert Axe Principal" if action[0][1] == 0 else "Vert Axe Secondaire"
        
        print(f"Pas {step_count:3d} | Feux (B0: {phase_b0} | C0: {phase_c0}) | Score: {float(reward[0]):8.2f}")

        if use_gui:
            time.sleep(0.15)

    print(f"\n🏁 Évaluation terminée. Récompense globale : {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()