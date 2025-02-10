# visualization.py
import matplotlib
matplotlib.use('Agg')  # Use Agg backend that doesn't require GUI
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from poker_game import HandRank
import seaborn as sns
import os
import json
class DataCollector:
    def __init__(self, save_interval, plot_interval, output_dir="viz_json"):
        """
        Initialise le collecteur de données.
        
        Args:
            save_interval (int): Nombre d'épisodes à regrouper avant de sauvegarder
            plot_interval (int): Intervalle pour le tracé (actuellement non utilisé)
            output_dir (str): Répertoire pour enregistrer les fichiers JSON
        """
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.output_dir = output_dir
        self.current_episode_states = []
        self.batch_episode_states = [] # Contient un liste de current_episode_states qui seront ajouté toutes les save_interval dans le fichier json
        
        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def add_state(self, state_info):
        """
        Ajoute un état à l'épisode courant avec le vecteur d'état subdivisé.
        """
        state_vector = state_info["state_vector"]
        
        # Subdiviser le vecteur d'état (basé sur la structure dans poker_game.py)
        subdivided_state = {
            "player_cards": {
                "card1": {
                    "values": state_vector[0:13],
                    "suits": state_vector[13:17]
                },
                "card2": {
                    "values": state_vector[17:30],
                    "suits": state_vector[30:34]
                }
            },
            "community_cards": [
                {
                    "values": state_vector[34+i*17:47+i*17],
                    "suits": state_vector[47+i*17:51+i*17]
                } for i in range(5)
            ],
            "hand_rank": state_vector[119],
            "game_phase": state_vector[120:125],
            "current_max_bet": state_vector[125],
            "player_stacks": state_vector[126:132],
            "current_bets": state_vector[132:138],
            "player_activity": state_vector[138:144],
            "fold_status": state_vector[144:150],
            "relative_positions": state_vector[150:156],
            "available_actions": state_vector[156:161],
            "previous_actions": [
                state_vector[161+i*6:167+i*6] for i in range(6)
            ],
            "win_probability": state_vector[197],
            "pot_odds": state_vector[198],
            "equity": state_vector[199],
            "aggression_factor": state_vector[200]
        }

        # Mettre à jour state_info avec le vecteur d'état subdivisé
        state_info["state_vector"] = subdivided_state
        self.current_episode_states.append(state_info)
    
    def save_episode(self, episode_num):
        """
        Sauvegarde les états de l'épisode courant dans un fichier JSON.
        
        Args:
            episode_num (int): Numéro de l'épisode
        """
        # Add current episode to batch
        self.batch_episode_states.append(self.current_episode_states)
        
        # Check if we've reached the save interval
        if len(self.batch_episode_states) >= self.save_interval:
            filename = os.path.join(self.output_dir, "episodes_states.json")
            
            # Charger les données existantes ou créer un nouveau dictionnaire
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    all_episodes = json.load(f)
            else:
                all_episodes = {}
            
            # Ajouter tous les épisodes batchés aux données
            for i, episode_states in enumerate(self.batch_episode_states):
                episode_idx = str(episode_num - len(self.batch_episode_states) + i + 1)
                all_episodes[episode_idx] = episode_states
            
            # Sauvegarder toutes les données
            with open(filename, 'w') as f:
                json.dump(all_episodes, f, indent=2)
            
            # Réinitialiser le batch
            self.batch_episode_states = []
        
        # Réinitialiser les états de l'épisode courant
        self.current_episode_states = []