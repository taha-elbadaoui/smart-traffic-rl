import os
import argparse
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on: {device.upper()} ---")


def make_env(rank, seed=0):
    """Factory function for SubprocVecEnv. Each subprocess needs its own env instance."""
    def _init():
        env = CrossroadEnv(CONFIG_PATH, use_gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossroad DQN Training")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    parser.add_argument("--num_cpu", type=int, default=4,
                        help="Parallel environments. Note: DQN is off-policy so more "
                             "envs = more diverse experience, but SB3's DQN wasn't "
                             "designed for VecEnv — 4 is a safe ceiling.")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"Launching DQN training across {args.num_cpu} environments...")

        env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])

        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,

            # --- Learning rate ---
            # 1e-3 is often too aggressive for traffic envs with sparse rewards.
            # 5e-4 is more stable; the agent makes smaller updates and doesn't
            # overshoot when it finally sees a good/bad episode.
            learning_rate=5e-4,

            # --- Replay buffer ---
            # With 4 parallel envs you fill the buffer 4x faster.
            # 100k gives the agent a diverse mix of experiences to sample from.
            # Too small = it keeps relearning from the same recent episodes.
            buffer_size=100_000,

            # --- Learning starts ---
            # Don't update the network until 5000 steps are in the buffer.
            # Prevents learning from an almost-empty buffer (noisy, unstable).
            learning_starts=5000,

            # --- Batch size ---
            # How many transitions are sampled from the buffer per update.
            # 128 is more stable than the default 32 for environments with
            # continuous numerical states like queue lengths.
            batch_size=128,

            # --- Exploration ---
            # The agent picks random actions for the first 30% of training (300k steps).
            # This forces it to experience BOTH phases enough times to learn
            # that switching is sometimes the right move.
            # Previously 0.1 = only 100k steps of exploration — too little.
            exploration_fraction=0.3,

            # --- Final epsilon ---
            # After exploration ends, the agent still picks random actions 5% of
            # the time. Prevents it from getting stuck in a deterministic rut.
            exploration_final_eps=0.05,

            # --- Target network update ---
            # The target Q-network (used for stable bootstrapping) is updated
            # every 1000 steps. Too frequent = unstable training.
            target_update_interval=1000,

            # --- Tensorboard ---
            # Run `tensorboard --logdir tensorboard_logs` to visualize training.
            tensorboard_log=LOG_DIR,

            device=device,
        )

        model.learn(
            total_timesteps=1_000_000,
            progress_bar=True,
            tb_log_name="dqn_crossroad",  # folder name inside tensorboard_logs/
        )

        model_name = "dqn_crossroad_trained"

    else:
        print("Creating untrained baseline model (random policy)...")
        # Single env is enough — we're not actually training, just saving
        # the initialized weights so evaluate_crossroad.py --mode random
        # loads a model that acts purely by chance. This is your baseline.
        env = CrossroadEnv(CONFIG_PATH, use_gui=False)
        model = DQN("MlpPolicy", env, verbose=1, device=device)
        model_name = "dqn_crossroad_random"

    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Model saved: {model_path}")

    env.close()