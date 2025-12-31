import pandas as pd
import matplotlib.pyplot as plt

# Load the evaluation results
df = pd.read_csv("results/evaluation_results.csv")

# Plot 1: Total reward per episode
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['total_reward'], marker='o', linestyle='-', color='blue')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

# Plot 2: Episode length per episode
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['episode_length'], marker='o', linestyle='-', color='green')
plt.title("Episode Length per Episode")
plt.xlabel("Episode")
plt.ylabel("Episode Length (timesteps)")
plt.grid(True)
plt.show()

# Plot 3: Reward vs Episode Length (scatter to see correlation)
plt.figure(figsize=(8,6))
plt.scatter(df['episode_length'], df['total_reward'], color='purple')
plt.title("Total Reward vs Episode Length")
plt.xlabel("Episode Length (timesteps)")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()




