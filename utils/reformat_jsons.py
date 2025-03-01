import json

with open('viz_json/metrics_history.json', 'r') as f:
    data_metrics = json.load(f)
with open('viz_json/episodes_states.json', 'w') as f:
    data_episodes = json.load(f)

# save data in a new json file
with open('viz_json/metrics_history_reformatted.json', 'w') as f:
    json.dump(data_metrics, f, indent=4)

with open('viz_json/episodes_states_reformatted.json', 'w') as f:
    json.dump(data_episodes, f, indent=4)


