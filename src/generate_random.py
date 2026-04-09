from stable_baselines3 import DQN
from wrapper import SingleIntersectionEnv

# Initialisation simple sans .learn()
env = SingleIntersectionEnv("../envs/1x1_minimal/env.sumocfg")
model = DQN("MlpPolicy", env)

# Sauvegarde immédiate
model.save("../models/dqn_untrained_random")
print("Modèle aléatoire sauvegardé.")
