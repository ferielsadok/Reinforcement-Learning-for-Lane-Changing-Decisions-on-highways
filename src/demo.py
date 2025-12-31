import csv
import numpy as np
from stable_baselines3 import PPO
from env_continuous import SumoContinuousEnv

MODEL_PATH = "models/ppo_lane_change"
CSV_PATH = "results/evaluation_results.csv"

N_EPISODES = 50

env = SumoContinuousEnv(
    sumo_cfg="data/obstacles.sumocfg",
    max_steps=200
)

model = PPO.load(MODEL_PATH)

results = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False

    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    results.append([ep, total_reward, steps])
    print(f"Episode {ep} | Reward: {total_reward:.2f} | Steps: {steps}")

env.close()

# Save CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward", "episode_length"])
    writer.writerows(results)

print(f"Evaluation saved to {CSV_PATH}")

