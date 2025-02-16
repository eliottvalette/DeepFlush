import json 
from tqdm import tqdm
import os

with open('./viz_json/episodes_states.json', 'r') as f:
    episode_states = json.load(f)
with open('./viz_json/metrics_history.json', 'r') as f:
    metrics_history = json.load(f)

rewards = [0] * 6
for episode_idx, metrics_list in tqdm(metrics_history.items()):
    for i in range(6):
        rewards[i] += metrics_list[i]['reward']

# [341.66090590229936, 400.4821081346454, -288.28692246341296, -8.357756385473605, -281.46495861305715, -979.2834231968868]
print(rewards)