import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(rank, seed=0):
    def _init():
        env = CrossroadEnv(CONFIG_PATH, use_gui=False, rank=rank)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel PPO Training for Crossroad")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    parser.add_argument("--num_cpu", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "train":
        print(f"Starting PPO training across {args.num_cpu} environments...")

        vec_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])

        # Widened clip_reward
        env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=50.0,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=100_000 // args.num_cpu,
            save_path=MODEL_DIR,
            name_prefix="ppo_crossroad_ckpt",
            verbose=1,
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
            ent_coef=0.03, # Bumped for exploration
            tensorboard_log=LOG_DIR,
            device="cpu",  
        )

        model.learn(
            total_timesteps=1_000_000,
            progress_bar=True,
            tb_log_name="ppo_crossroad",
            callback=checkpoint_cb,
        )

        model_name = "ppo_crossroad_final"
        model_path = os.path.join(MODEL_DIR, model_name)
        model.save(model_path)

        norm_path = os.path.join(MODEL_DIR, "ppo_crossroad_vecnorm.pkl")
        env.save(norm_path)
        print(f"Model saved:      {model_path}.zip")
        print(f"Normalizer saved: {norm_path}")

    else:
        print("Creating untrained random baseline model...")
        env = CrossroadEnv(CONFIG_PATH, use_gui=False, rank=0)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        model.save(os.path.join(MODEL_DIR, "ppo_crossroad_random"))
        env.close()