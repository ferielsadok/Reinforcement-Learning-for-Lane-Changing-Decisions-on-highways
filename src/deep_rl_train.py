import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env_continuous import SumoContinuousEnv

LOG_DIR = "logs/ppo_lane_change"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env():
    env = SumoContinuousEnv(
        sumo_cfg="data/obstacles.sumocfg",
        max_steps=200
    )
    env = Monitor(env)   # REQUIRED for episode rewards
    return env


env = DummyVecEnv([make_env])

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=100_000)

model.save(f"{MODEL_DIR}/ppo_lane_change")

env.close()
print("Training finished and model saved.")

