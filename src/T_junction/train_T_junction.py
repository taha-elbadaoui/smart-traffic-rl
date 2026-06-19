import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wrapper_T_junction import TJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/T_junction/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Force le vidage immédiat des logs sur le disque pour TensorBoard
os.environ["TENSORBOARD_BINARY_FLUSH_SECONDS"] = "5"

device = "cpu"
# Tiny MLP: one torch thread avoids the learner fighting the SUMO env workers
# for CPU, which improves overall SubprocVecEnv throughput.
torch.set_num_threads(1)
print(f"--- Running on: {device.upper()} ---")

def make_env(rank, seed=0, use_gui=False):
    def _init():
        env = TJunctionEnv(CONFIG_PATH, use_gui=use_gui, env_rank=rank)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for T-Junction")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    parser.add_argument("--num_cpu", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "train":
        print(f"🚀 Starting PPO training across {args.num_cpu} environments...")

        vec_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])

        env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=50.0,
        )

        # 1. Callback classique pour sauvegarder des checkpoints réguliers
        checkpoint_cb = CheckpointCallback(
            save_freq=100_000 // args.num_cpu,
            save_path=MODEL_DIR,
            name_prefix="ppo_t_junction_ckpt",
            verbose=1,
        )

        # 2. Callback d'ÉVALUATION FORCÉE pour générer le dossier 'eval/' dans TensorBoard
        # On utilise un seul CPU isolé en arrière-plan sans GUI pour calculer les performances réelles
        print("📊 Configuring isolated Evaluation Environment for TensorBoard 'eval/' tracking...")
        eval_vec_env = SubprocVecEnv([make_env(rank=999, seed=42, use_gui=False)])
        
        eval_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=False,  # Important : pas de normalisation de récompense à l'évaluation pour garder les vrais scores
            clip_obs=10.0,
        )
        # Share the live training normalizer (mutated in place) and freeze the eval
        # env's own updates, so eval/ observations are normalized with trained stats.
        eval_env.obs_rms = env.obs_rms
        eval_env.training = False

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
            log_path=LOG_DIR,
            eval_freq=max(1000, 10_000 // args.num_cpu),  # Évalue toutes les 10 000 étapes cumulées d'entraînement
            deterministic=True,
            render=False,
            warn=False
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,           
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03, 
            tensorboard_log=LOG_DIR,
            device=device,
        )

        # Entraînement en passant la liste des deux callbacks conjugués
        model.learn(
            total_timesteps=500_000,
            progress_bar=True,
            tb_log_name="ppo_t_junction_eval_forced",
            callback=[checkpoint_cb, eval_cb],
        )

        model_name = "ppo_t_junction_final"
        model_path = os.path.join(MODEL_DIR, model_name)
        model.save(model_path)

        norm_path = os.path.join(MODEL_DIR, "ppo_t_junction_vecnorm.pkl")
        env.save(norm_path)
        
        # Nettoyage des environnements
        eval_env.close()
        env.close()
        
        print(f"✅ Model saved:      {model_path}.zip")
        print(f"✅ Normalizer saved: {norm_path}")

    else:
        print("🎲 Creating an UNTRAINED random baseline model...")
        env = TJunctionEnv(CONFIG_PATH, use_gui=False)
        model = PPO("MlpPolicy", env, verbose=1, device=device)
        model.save(os.path.join(MODEL_DIR, "ppo_t_junction_random"))
        env.close()