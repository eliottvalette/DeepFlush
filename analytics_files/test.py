import json

with open("./viz_json/episodes_states.json", "r") as f:
    data = json.load(f)

print(data.keys())