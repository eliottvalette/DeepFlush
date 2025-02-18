import json 
from tqdm import tqdm
import os
import numpy as np

# Charger les données
with open('./viz_json/episodes_states.json', 'r') as f:
    episode_states = json.load(f)
with open('./viz_json/metrics_history.json', 'r') as f:
    metrics_history = json.load(f)

rewards = {f"Player_{i}": [] for i in range(6)}
stack_change_history = {f"Player_{i}": [] for i in range(6)}
stack_final_history = {f"Player_{i}": [] for i in range(6)}

# Handle rewards
for episode_idx, metrics_list in metrics_history.items():
    for i in range(6):
        rewards[f"Player_{i}"].append(metrics_list[i]['reward'])

# Handle stack changes and final stacks
for episode_idx, episode_states in episode_states.items():
    for state in episode_states:
        for i in range(6):
            player = state['player']
            total_bets = sum(state['state_vector']['current_bets'])
            total_stack = sum(state['state_vector']['player_stacks'])
            all_money_in_game = total_bets + total_stack
            if all_money_in_game > 600.2 or all_money_in_game < 599.8:
                print('state["state_vector"] :', json.dumps(state, indent=4))
                print('episode_idx :', episode_idx)
                print('all_money_in_game :', all_money_in_game)
                print('total_bets :', total_bets)
                print('total_stack :', total_stack)
                print('phase :', state['phase'])
                print('player :', player)
                print('--------------------------------')
                raise ValueError('Error')
        for i in range(6):
            player = f"Player_{i}"
            stack_change_history[player].append(state['stack_changes'][player])
            stack_final_history[player].append(state['final_stacks'][player])
            if state['stack_changes'][player] > 500.2:
                print('state[stack_changes][player] :', state['stack_changes'][player])
                print('episode_idx :', episode_idx)
                print('--------------------------------')
            if state['final_stacks'][player] > 600.2:
                print('state[final_stacks][player] :', state['final_stacks'][player])
                print('episode_idx :', episode_idx)
                print('--------------------------------')

# Print statistics
print("\nRewards Statistics:")
for player in rewards:
    print(f"{player}: mean={np.mean(rewards[player]):.2f}, std={np.std(rewards[player]):.2f}, min={np.min(rewards[player]):.2f}, max={np.max(rewards[player]):.2f}")

print("\nStack Change Statistics:")
for player in stack_change_history:
    print(f"{player}: mean={np.mean(stack_change_history[player]):.2f}, std={np.std(stack_change_history[player]):.2f}, min={np.min(stack_change_history[player]):.2f}, max={np.max(stack_change_history[player]):.2f}")

print("\nFinal Stack Statistics:")
for player in stack_final_history:
    print(f"{player}: mean={np.mean(stack_final_history[player]):.2f}, std={np.std(stack_final_history[player]):.2f}, min={np.min(stack_final_history[player]):.2f}, max={np.max(stack_final_history[player]):.2f}")
