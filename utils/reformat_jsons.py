import json
import os
from pathlib import Path

# Lecture des fichiers JSON
with open('./viz_json/metrics_history.json', 'r') as f:
    data_metrics = json.load(f)

with open('./viz_json/episodes_states.json', 'r') as f:
    data_episodes = json.load(f)

# Sauvegarde des fichiers reformatés
with open('./viz_json/metrics_history_reformatted.json', 'w') as f:
    json.dump(data_metrics, f, indent=1)

with open('./viz_json/episodes_states_reformatted.json', 'w') as f:
    json.dump(data_episodes, f, indent=1)

print("Fichiers JSON reformatés avec succès!")

