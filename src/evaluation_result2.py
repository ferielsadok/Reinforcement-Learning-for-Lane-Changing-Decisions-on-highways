import numpy as np
import pandas as pd
import os

# --- 1. CONVERTIR LA Q-TABLE ---
q_table_path = "src/q_table_highway.npy"

if os.path.exists(q_table_path):
    q_table = np.load(q_table_path)
    
    # On crée des noms de colonnes pour les actions
    # 0: Stay, 1: Left, 2: Right (selon ta logique)
    df_q = pd.DataFrame(q_table, columns=['Action_Stay', 'Action_Left', 'Action_Right'])
    
    # On ajoute une colonne pour l'index de l'état
    df_q.index.name = 'State_Index'
    
    # Sauvegarde en CSV
    df_q.to_csv("q_table_results.csv")
    print("✅ Q-Table convertie : q_table_results.csv")
else:
    print("❌ Fichier q_table_highway.npy introuvable.")

# --- 2. CONVERTIR L'HISTORIQUE DES REWARDS ---
history_path = "src/rewards_history.npy"

if os.path.exists(history_path):
    rewards = np.load(history_path)
    
    # On crée un DataFrame avec une colonne Episode et une colonne Reward
    df_rewards = pd.DataFrame({
        'Episode': np.arange(len(rewards)),
        'Total_Reward': rewards
    })
    
    # Sauvegarde en CSV
    df_rewards.to_csv("rewards_history.csv", index=False)
    print("✅ Historique converti : rewards_history.csv")
else:
    print("❌ Fichier rewards_history.npy introuvable.")