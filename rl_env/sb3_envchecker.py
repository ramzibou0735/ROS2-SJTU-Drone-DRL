from drone_env import SecurityDroneEnv
from stable_baselines3.common.env_checker import check_env

env = SecurityDroneEnv()
check_env(env)
print("Environment passed all checks")
