import random
import numpy as np
import matplotlib.pyplot as plt
from env import SumoEnv
import time
import inspect
import os  # Nécessaire pour vérifier si le fichier existe

def state_to_index(state):
    lane, d1, d2 = state
    return int(lane * 9 + d1 * 3 + d2)

# --- CONFIGURATION ---
alpha = 0.1 
gamma = 0.95
epsilon = 1.0 
epsilon_decay = 0.998
min_epsilon = 0.05
nbr_episode = 500  # Augmenté pour un vrai apprentissage

n_state = 18
n_actions = 3

# --- INITIALISATION OU CHARGEMENT DE LA Q-TABLE ---
if os.path.exists("q_table_highway.npy"):
    q_table = np.load("q_table_highway.npy")
    # Si on charge une table apprise, on peut réduire epsilon pour moins de hasard
    epsilon = 0.3 
    print("--- Q-Table chargée avec succès ! Reprise de l'apprentissage... ---")
else:
    q_table = np.zeros((n_state, n_actions))
    print("--- Nouvelle Q-Table créée ---")

# --- POUR LES GRAPHIQUES ---
rewards_history = []
epsilons_history = []

print("Arguments acceptés par SumoEnv:", inspect.signature(SumoEnv.__init__))

# ENTRAÎNEMENT SANS GUI
env = SumoEnv(use_gui=False)

print("--- Début de l'entraînement (Mode Rapide) ---")

for episode in range(nbr_episode):
    state_raw = env.reset()
    state = state_to_index(state_raw)
    done = False
    total_reward = 0

    for step in range(env.max_steps):
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(q_table[state, :])

        next_state_raw, reward, done = env.step(action)
        next_state_int = state_to_index(next_state_raw)

        # Mise à jour
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state_int, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state_int
        total_reward += reward
        if done: break

    rewards_history.append(total_reward)
    epsilons_history.append(epsilon)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if episode % 10 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

env.close()

# --- SAUVEGARDE DE LA Q-TABLE ---
np.save("q_table_highway.npy", q_table)
print("--- Q-Table sauvegardée sous le nom 'q_table_highway.npy' ---")

# --- AFFICHAGE DES COURBES ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_history, color='blue', alpha=0.3, label='Reward Brut')
if len(rewards_history) > 10:
    rolling_mean = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
    plt.plot(rolling_mean, color='red', label='Moyenne Mobile (10)')
plt.title('Évolution du Reward')
plt.xlabel('Épisodes')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epsilons_history, color='green')
plt.title('Décroissance de l\'Epsilon')
plt.xlabel('Épisodes')
plt.tight_layout()
plt.show()

# --- PHASE DE TEST VISUEL ---
print("\n--- Début des tests (Visualisation) ---")
env = SumoEnv(use_gui=True) 
for ep in range(5):
    state_raw = env.reset()
    state = state_to_index(state_raw) 
    done = False
    while not done:
        action = np.argmax(q_table[state, :])
        next_state_raw, reward, done = env.step(action)
        state = state_to_index(next_state_raw) 
    print(f"Test Episode {ep} terminé.")
env.close()