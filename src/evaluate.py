import numpy as np
import matplotlib.pyplot as plt

history = np.load("src/rewards_history.npy")
plt.plot(history, label="Reward")
plt.title("Historique d'apprentissage")
plt.legend()
plt.show()