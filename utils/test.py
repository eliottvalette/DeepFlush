import json 
from tqdm import tqdm
import os
import numpy as np

# Charger les données
with open('./viz_json/episodes_states.json', 'r') as f:
    episode_states = json.load(f)
with open('./viz_json/metrics_history.json', 'r') as f:
    metrics_history = json.load(f)

# Test 1: Vérifier les récompenses par joueur
rewards = {f"Player_{i}": [] for i in range(6)}
for episode_idx, metrics_list in metrics_history.items():
    for i in range(6):
        rewards[f"Player_{i}"].append(metrics_list[i]['reward'])

print("\nTest 1: Moyenne des récompenses par joueur:")
for player, player_rewards in rewards.items():
    print(f"{player}: {np.mean(player_rewards):.2f}")

# Test 2: Vérifier les changements de stack par joueur
stack_changes = {f"Player_{i}": [] for i in range(6)}
for episode in episode_states.values():
    for state in episode:
        if state["phase"] == "showdown":
            for player in stack_changes.keys():
                if player in state["stack_changes"]:
                    stack_changes[player].append(state["stack_changes"][player])

print("\nTest 2: Moyenne des changements de stack par joueur:")
for player, changes in stack_changes.items():
    print(f"{player}: {np.mean(changes):.2f}")

# Test 3: Vérifier le nombre d'actions par joueur
actions = {f"Player_{i}": {"fold": 0, "check": 0, "call": 0, "raise": 0, "all-in": 0} for i in range(6)}
for episode in episode_states.values():
    for state in episode:
        if state["action"] and state["player"] in actions:
            action = state["action"].lower()
            if action in actions[state["player"]]:
                actions[state["player"]][action] += 1

print("\nTest 3: Nombre d'actions par joueur:")
for player, action_counts in actions.items():
    print(f"\n{player}:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")

# Test 4: Vérifier la cohérence des données de showdown
showdown_inconsistencies = 0
for episode in episode_states.values():
    showdown_states = [s for s in episode if s["phase"] == "showdown"]
    if showdown_states:
        last_showdown = showdown_states[-1]
        # Vérifier que tous les joueurs sont présents dans stack_changes
        if len(last_showdown["stack_changes"]) != 6:
            showdown_inconsistencies += 1
            print(f"\nIncohérence trouvée dans stack_changes:")
            print(f"Joueurs présents: {list(last_showdown['stack_changes'].keys())}")

print(f"\nTest 4: Nombre d'incohérences dans les données de showdown: {showdown_inconsistencies}")

# Test 5: Vérifier la distribution des gains/pertes
print("\nTest 5: Distribution des gains/pertes par joueur:")
for player, changes in stack_changes.items():
    if changes:
        print(f"\n{player}:")
        print(f"  Min: {min(changes):.2f}")
        print(f"  Max: {max(changes):.2f}")
        print(f"  Médiane: {np.median(changes):.2f}")
        print(f"  Écart-type: {np.std(changes):.2f}")

# Ajoutons un test supplémentaire pour voir la corrélation entre stack changes et rewards
print("\nTest 6: Corrélation stack changes vs rewards par épisode:")
for episode_num in list(episode_states.keys())[:10]:  # Regardons les 10 premiers épisodes
    print(f"\nÉpisode {episode_num}:")
    # Trouver le dernier état showdown
    showdown_states = [s for s in episode_states[episode_num] if s["phase"] == "showdown"]
    if showdown_states:
        last_showdown = showdown_states[-1]
        stack_changes = last_showdown["stack_changes"]
        rewards_for_episode = metrics_history[episode_num]
        
        for i in range(6):
            player = f"Player_{i}"
            stack_change = stack_changes[player]
            reward = rewards_for_episode[i]["reward"]
            print(f"{player}: Stack Change = {stack_change:.2f}, Reward = {reward:.2f}")
