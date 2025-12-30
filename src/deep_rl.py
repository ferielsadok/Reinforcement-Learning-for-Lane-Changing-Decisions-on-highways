import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from env_continuous import SumoContinuousEnv

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Create Environment ---
def make_env():
    return SumoContinuousEnv(sumo_cfg="data/obstacles.sumocfg", max_steps=200)

env = DummyVecEnv([make_env])

# --- Evaluation callback ---
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=5000,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    verbose=1
)

# --- Create DQN Agent ---
model = DQN(
    "MlpPolicy",          # Fully connected network
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# --- Train ---
TIMESTEPS = 50000
model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)

# --- Save final model ---
model_path = os.path.join(MODEL_DIR, "dqn_sumo")
model.save(model_path)
print(f"Model saved to {model_path}")

# --- Close environment ---
env.close()
