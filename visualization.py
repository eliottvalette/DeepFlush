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
import pandas as pd

class DataCollector:
    def __init__(self, save_interval, plot_interval, start_epsilon, epsilon_decay, output_dir="viz_json"):
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
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.current_episode_states = []
        self.batch_episode_states = [] # Contient un liste de current_episode_states qui seront ajouté toutes les save_interval dans le fichier json
        self.batch_episode_metrics = [] # Contient les métriques d'entraînement pour chaque épisode
        
        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Supprimer les fichiers JSON existants dans le répertoire
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

        self.visualizer = Visualizer(output_dir=output_dir, start_epsilon=start_epsilon, epsilon_decay=epsilon_decay)
    
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
    
    def add_metrics(self, episode_metrics):
        """
        Ajoute les métriques d'un épisode au batch.
        
        Args:
            episode_metrics (list): Liste des métriques pour chaque agent dans l'épisode
        """
        self.batch_episode_metrics.append(episode_metrics)
    
    def save_episode(self, episode_num):
        """
        Sauvegarde les états et métriques de l'épisode courant dans des fichiers JSON.
        
        Args:
            episode_num (int): Numéro de l'épisode
        """
        # Add current episode to batch
        self.batch_episode_states.append(self.current_episode_states)
        
        # Check if we've reached the save interval
        if len(self.batch_episode_states) >= self.save_interval:
            # Sauvegarder les états
            states_filename = os.path.join(self.output_dir, "episodes_states.json")
            if os.path.exists(states_filename):
                with open(states_filename, 'r') as f:
                    all_episodes = json.load(f)
            else:
                all_episodes = {}
            
            # Ajouter tous les épisodes batchés aux données
            for i, episode_states in enumerate(self.batch_episode_states):
                episode_idx = episode_num - len(self.batch_episode_states) + i + 1
                all_episodes[str(episode_idx)] = episode_states  # Convertir l'index en string
            
            # Sauvegarder les états
            with open(states_filename, 'w') as f:
                json.dump(all_episodes, f, indent=2)
            
            # Sauvegarder les métriques
            metrics_filename = os.path.join(self.output_dir, "metrics_history.json")
            if os.path.exists(metrics_filename):
                with open(metrics_filename, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}
            
            # Ajouter toutes les métriques batchées aux données
            for i, episode_metrics in enumerate(self.batch_episode_metrics):
                episode_idx = episode_num - len(self.batch_episode_metrics) + i + 1
                all_metrics[str(episode_idx)] = episode_metrics  # Convertir l'index en string
            
            # Sauvegarder les métriques
            with open(metrics_filename, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            # Réinitialiser les batchs
            self.batch_episode_states = []
            self.batch_episode_metrics = []
        
        # Réinitialiser les états de l'épisode courant
        self.current_episode_states = []

        if episode_num % self.plot_interval == self.plot_interval - 1:
            self.visualizer.plot_metrics()
    
    def plot_metrics(self):
        self.visualizer.plot_metrics()

class Visualizer:
    """
    Visualise les données collectées dans le répertoire viz_json
    """
    def __init__(self, start_epsilon, epsilon_decay, output_dir="viz_json"):
        self.output_dir = output_dir
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Définition des couleurs pour chaque action
        self.action_colors = {
            'fold': '#780000',     # Rouge Sang
            'check': '#C1121F',    # Rouge
            'call': '#FDF0D5',     # Beige
            'raise': '#669BBC',    # Bleu
            'all-in': '#003049'    # Bleu Nuit
        }

    def plot_metrics(self):
        """
        Génère les visualisations à partir des données JSON enregistrées
        """
        # Charger les données
        states_path = os.path.join(self.output_dir, "episodes_states.json")
        metrics_path = os.path.join(self.output_dir, "metrics_history.json")
        
        with open(states_path, 'r') as f:
            states_data = json.load(f)
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        # Créer une figure avec 4 sous-graphiques
        fig = plt.figure(figsize=(20, 15))
        
        # Définir une palette de couleurs pastel
        pastel_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7']
        
        # 1. Average mbb/hand par agent
        ax1 = plt.subplot(2, 2, 1)
        window = 50
        agents = set()
        for episode in states_data.values():
            for state in episode:
                agents.add(state["player"])
        
        mbb_data = {agent: [] for agent in agents}
        for episode_num, episode in states_data.items():
            episode_results = {agent: 0 for agent in agents}
            for state in episode:
                player = state["player"]
                if state["phase"] == "showdown":
                    stack_change = state["state_vector"]["player_stacks"][int(player.split("_")[1]) - 1]
                    episode_results[player] = stack_change * 1000  # Conversion en mbb
            
            for agent in agents:
                mbb_data[agent].append(episode_results[agent])
        
        for i, (agent, data) in enumerate(mbb_data.items()):
            rolling_avg = pd.Series(data).rolling(window=window).mean()
            ax1.plot(rolling_avg, label=agent, color=pastel_colors[i % len(pastel_colors)], linewidth=2)
        
        ax1.set_title('Average mbb/hand par agent')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('mbb/hand')
        ax1.legend()
        
        ax1.set_facecolor('#F0F0F0')  # Fond légèrement gris
        ax1.grid(True, alpha=0.3)
        
        # 2. Fréquence des actions par agent
        ax3 = plt.subplot(2, 2, 2)
        action_freq = {agent: {
            'fold': 0, 
            'check': 0, 
            'call': 0, 
            'raise': 0, 
            'all-in': 0
        } for agent in agents}
        
        for episode in states_data.values():
            for state in episode:
                if state["action"]:
                    action = state["action"].lower()
                    action_freq[state["player"]][action] += 1
        
        x = np.arange(len(agents))
        width = 0.15
        actions = ['fold', 'check', 'call', 'raise', 'all-in']
        
        for i, action in enumerate(actions):
            values = [action_freq[agent][action] for agent in agents]
            total_actions = [sum(action_freq[agent].values()) for agent in agents]
            frequencies = [v/t if t > 0 else 0 for v, t in zip(values, total_actions)]
            bars = ax3.bar(x + i*width, frequencies, width, 
                          label=action, 
                          color=self.action_colors[action])
            
            # Ajouter les pourcentages au-dessus des barres
            for bar, freq in zip(bars, frequencies):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{freq*100:.1f}%',
                        ha='center', va='bottom', rotation=0,
                        fontsize=8)
        
        ax3.set_title('Fréquence des actions par agent')
        ax3.set_xticks(x + width*2)
        ax3.set_xticklabels(agents)
        ax3.legend()
        ax3.set_ylim(0, 1)  # Fixer la limite y à 1
        
        ax3.set_facecolor('#F0F0F0')
        ax3.grid(True, alpha=0.3)

        # 3. Fréquence des actions par agent et par phase de jeu
        ax2 = plt.subplot(2, 2, 3)
        phase_action_freq = {agent: {
            phase: {
                'fold': 0, 
                'check': 0, 
                'call': 0, 
                'raise': 0, 
                'all-in': 0
            } for phase in ['preflop', 'flop', 'turn', 'river', 'showdown']
        } for agent in agents}

        # Compter les actions par phase pour chaque agent
        for episode in states_data.values():
            for state in episode:
                if state["action"] and state["phase"].lower() != 'showdown':
                    action = state["action"].lower()
                    phase = state["phase"].lower()
                    phase_action_freq[state["player"]][phase][action] += 1

        # Créer le graphique empilé pour chaque phase
        phases = ['preflop', 'flop', 'turn', 'river']
        actions = ['fold', 'check', 'call', 'raise', 'all-in']
        x = np.arange(len(agents))
        width = 0.2  # Largeur des barres

        for p_idx, phase in enumerate(phases):
            bottom = np.zeros(len(agents))
            for a_idx, action in enumerate(actions):
                values = []
                for agent in agents:
                    total_actions = sum(phase_action_freq[agent][phase].values())
                    freq = phase_action_freq[agent][phase][action] / total_actions if total_actions > 0 else 0
                    values.append(freq)
                
                bars = ax2.bar(x + p_idx * width - width * 1.5, values, width, 
                        bottom=bottom, 
                        label=f'{action} ({phase})' if p_idx == 0 else "",
                        color=self.action_colors[action],
                        alpha=0.7)
                
                # Ajouter les pourcentages pour les barres > 5%
                for idx, (val, b) in enumerate(zip(values, bars)):
                    if val > 0.05:  # Seulement pour les valeurs > 5%
                        ax2.text(
                            b.get_x() + b.get_width()/2.,
                            bottom[idx] + val/2,
                            f'{val*100:.0f}%',
                            ha='center',
                            va='center',
                            fontsize=6,
                            color='black',
                            rotation=90
                        )
                bottom += np.array(values)  # Mettre à jour bottom avec un numpy array

        ax2.set_title('Fréquence des actions par agent et par phase de jeu')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, 1)

        ax2.set_facecolor('#F0F0F0')
        ax2.grid(True, alpha=0.3)

        # 4. Evolution de epsilon avec nouvelle couleur
        ax4 = plt.subplot(2, 2, 4)
        episodes = sorted([int(k) for k in metrics_data.keys()])
        epsilon_values = [self.start_epsilon * (self.epsilon_decay ** episode) for episode in episodes]
        ax4.plot(episodes, epsilon_values, color='#2E86AB', linewidth=2)  # Bleu foncé pour epsilon
        ax4.set_title('Evolution de Epsilon')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_ylim(0, 1)
        
        ax4.set_facecolor('#F0F0F0')
        ax4.grid(True, alpha=0.3)
        
        # Style global mis à jour
        plt.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#F8F9FA',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        plt.tight_layout()
        plt.savefig('viz_json/Poker_progress.jpg', dpi=750, bbox_inches='tight')
        plt.close()
    
if __name__ == "__main__":
    visualizer = Visualizer(start_epsilon=1, epsilon_decay=0.999)
    visualizer.plot_metrics()