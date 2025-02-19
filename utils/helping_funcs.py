import os
import torch
import json

def save_models(players, episode, models_dir="saved_models"):
    """
    Save the trained models for each player that has one.
    
    Args:
        players: List of players
        episode: Current episode number
        models_dir: Directory where models should be saved
    """
    print("\nSaving models...")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for player in players:
        # Only save agents that have a model (not bots)
        if hasattr(player.agent, 'model'):
            torch.save(
                player.agent.model.state_dict(),
                f"{models_dir}/poker_agent_{player.name}_epoch_{episode+1}.pth"
            )
    print("Models saved successfully!")

def save_metrics(metrics_history, output_dir):
    """
    Save training metrics history to a JSON file.
    
    Args:
        metrics_history: Dictionary containing training metrics
        output_dir: Directory where metrics should be saved
    """
    metrics_path = os.path.join(output_dir, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nTraining completed and metrics saved to '{metrics_path}'")
