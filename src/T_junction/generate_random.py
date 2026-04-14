from stable_baselines3 import DQN
from wrapper_T_junction import TJunctionEnv

# Initialisation simple sans .learn()
env = TJunctionEnv("../envs/T_junction/env.sumocfg")
model = DQN("MlpPolicy", env)

# Sauvegarde immédiate
model.save("../models/dqn_untrained_random")
print("Modèle aléatoire sauvegardé.")
