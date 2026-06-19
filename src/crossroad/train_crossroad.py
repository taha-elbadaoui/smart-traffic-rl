import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wrapper_crossroad import CrossroadEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CONFIG_PATH = os.path.join(ROOT_DIR, "envs/crossroad/env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Force immediate log flushes onto the disk for active TensorBoard streams
os.environ["TENSORBOARD_BINARY_FLUSH_SECONDS"] = "5"

print("--- Running PPO Policy execution optimized context on: CPU ---")

def make_env(rank, seed=0, use_gui=False):
    def _init():
        # Match wrapper initialization parameters
        env = CrossroadEnv(CONFIG_PATH, use_gui=use_gui, rank=rank)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel PPO Training for Crossroad")
    parser.add_argument("--mode", type=str, choices=["train", "random"], default="train")
    # G14 Tuning: Defaulting to 6 workers leaves 2 physical cores free for Windows and 
    # lets the remaining 6 cores sustain maximum single-core Turbo clocks without overheating.
    parser.add_argument("--num_cpu", type=int, default=6)
    args = parser.parse_args()

    if args.mode == "train":
        print(f"Starting PPO training across {args.num_cpu} parallel isolated environments...")

        vec_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])

        env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=50.0,
        )

        # 1. Standard Checkpoint Callback
        checkpoint_cb = CheckpointCallback(
            save_freq=100_000 // args.num_cpu,
            save_path=MODEL_DIR,
            name_prefix="ppo_crossroad_ckpt",
            verbose=1,
        )

        # 2. Forced Evaluation Callback for Real-Time TensorBoard Scaling Records
        print("📊 Configuring isolated Evaluation Environment for Crossroad 'eval/' folder...")
        eval_vec_env = SubprocVecEnv([make_env(rank=999, seed=1337, use_gui=False)])
        
        eval_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=False, # Maintain real evaluation step metrics intact
            clip_obs=10.0,
        )
        # Share the live training normalizer (mutated in place) and freeze the eval
        # env's own updates, so eval/ observations are normalized with trained stats.
        eval_env.obs_rms = env.obs_rms
        eval_env.training = False

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, "best_crossroad_model"),
            log_path=LOG_DIR,
            eval_freq=max(1000, 10_000 // args.num_cpu), # Evaluates every 10k total global steps across all workers
            deterministic=True,
            render=False,
            warn=False
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            # G14 Tuning: Increased n_steps and batch_size dramatically to let the CPU 
            # run pure simulation steps uninterrupted without constantly pausing for policy updates.
            n_steps=4096,      
            batch_size=128,    
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03,
            tensorboard_log=LOG_DIR,
            device="cpu",  
        )

        # Ensure the learn block uses a separate tensorboard graph name
        model.learn(
            total_timesteps=2_000_000, 
            progress_bar=True,
            tb_log_name="ppo_crossroad_eval_forced", 
            callback=[checkpoint_cb, eval_cb],
        )

        # Update these strings to isolate the 2M outputs
        model_name = "ppo_crossroad_2M_final"
        model_path = os.path.join(MODEL_DIR, model_name)
        model.save(model_path)

        norm_path = os.path.join(MODEL_DIR, "ppo_crossroad_2M_vecnorm.pkl")
        env.save(norm_path)
        
        eval_env.close()
        env.close()
        print(f"✅ Model saved:      {model_path}.zip")
        print(f"✅ Normalizer saved: {norm_path}")

    else:
        print("Creating untrained random baseline model...")
        env = CrossroadEnv(CONFIG_PATH, use_gui=False, rank=0)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        model.save(os.path.join(MODEL_DIR, "ppo_crossroad_random"))
        env.close()