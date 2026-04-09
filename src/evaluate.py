import time
from stable_baselines3 import DQN
from wrapper import SingleIntersectionEnv

CONFIG_PATH = "../envs/1x1_minimal/env.sumocfg"
MODEL_PATH = "../models/dqn_1x1_baseline"

print("Loading Trained Model and Launching GUI...")
# Set use_gui=True to force sumo-gui to open
env = SingleIntersectionEnv(CONFIG_PATH, use_gui=True)
model = DQN.load(MODEL_PATH, env=env)

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # The agent predicts the best action based on the queues
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"Action Taken: {'North/South Green' if action == 0 else 'East/West Green'} | Reward: {reward}")
    
    # Slow down the loop so your eyes can follow the cars
    time.sleep(0.1)

print(f"Evaluation finished. Total accumulated reward: {total_reward}")
env.close()
