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

PLAYERS = ['Player_0', 'Player_1', 'Player_2', 'Player_3', 'Player_4', 'Player_5']

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

        self.visualizer = Visualizer(output_dir=output_dir, start_epsilon=start_epsilon, epsilon_decay=epsilon_decay, plot_interval=plot_interval, save_interval=save_interval)
    
    def add_state(self, state_info):
        """
        Ajoute un état à l'épisode courant avec le vecteur d'état subdivisé.
        """
        state_vector = state_info["state_vector"]
        
        # Extraire le hand rank du vecteur d'état (indice 35-44 pour le one-hot encoding du rang)
        hand_rank_vector = state_vector[35:45]
        hand_rank = hand_rank_vector.index(1) if 1 in hand_rank_vector else 0
        
        # Subdiviser le vecteur d'état (basé sur la nouvelle structure)
        subdivided_full_state = {
            "player_cards": [
                {
                    "value": state_vector[0],  # Valeur normalisée
                    "suit": state_vector[1:5]  # One-hot encoding de la couleur
                },
                {
                    "value": state_vector[5],  # Valeur normalisée
                    "suit": state_vector[6:10]  # One-hot encoding de la couleur
                }
            ],
            "community_cards": [
                {
                    "value": state_vector[10 + i*5],  # Valeur normalisée
                    "suit": state_vector[11 + i*5:15 + i*5]  # One-hot encoding de la couleur
                } for i in range(5)
            ],
            "hand_info": {
                "rank": hand_rank,
                "kicker": state_vector[45],  # Valeur du kicker normalisée
                "normalized_rank": state_vector[46]  # Rang normalisé
            },
            "game_phase": state_vector[47:52],  # One-hot encoding de la phase
            "current_max_bet": state_vector[52],  # Mise maximale normalisée
            "player_stacks": state_vector[53:59],  # Stacks normalisés (Attention, c'est stack sont peut-etre réinitialisés si consultés au showdown, donc pas fiable)
            "current_bets": state_vector[59:65],  # Mises actuelles normalisées
            "player_activity": state_vector[65:71],  # État des joueurs (actif/inactif)
            "relative_positions": state_vector[71:77],  # Position relative one-hot
            "available_actions": state_vector[77:82],  # Actions disponibles
            "previous_actions": [
                state_vector[82+i*5:87+i*5] for i in range(6)  # One-hot encoding des actions précédentes
            ],
            "strategic_info": {
                "preflop_win_prob": state_vector[112],  # Probabilité de victoire préflop
                "pot_odds": state_vector[113]  # Cotes du pot
            },
            "hand_draw_potential": {
                "straight_draw": state_vector[114],  # Potentiel de quinte
                "flush_draw": state_vector[115]  # Potentiel de couleur
            }
        }

        subdivided_simple_state = {
            "player_cards": [
                {
                    "value": state_vector[0],  # Valeur normalisée
                    "suit": state_vector[1:5]  # One-hot encoding de la couleur
                },
                {
                    "value": state_vector[5],  # Valeur normalisée
                    "suit": state_vector[6:10]  # One-hot encoding de la couleur
                }
            ],
            "player_stacks": state_vector[53:59],  # Stacks normalisés
            "relative_positions": state_vector[71:77],  # Position relative one-hot
        }

        # Mettre à jour state_info avec le vecteur d'état subdivisé
        state_info["state_vector"] = subdivided_simple_state
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
            self.visualizer.plot_progress(dpi = 250)
        if episode_num % (self.plot_interval * 4) == (self.plot_interval * 4) - 1:
            self.visualizer.plot_metrics()
            self.visualizer.plot_analytics()
            self.visualizer.plot_heatmaps()
    
    def force_visualization(self):
        self.visualizer.plot_progress()
        self.visualizer.plot_metrics()
        self.visualizer.plot_analytics()
        self.visualizer.plot_heatmaps()

class Visualizer:
    """
    Visualise les données collectées dans le répertoire viz_json
    """
    def __init__(self, start_epsilon, epsilon_decay, plot_interval, save_interval, output_dir="viz_json"):
        self.output_dir = output_dir
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.plot_interval = plot_interval
        self.save_interval = save_interval
        
        # Définition des couleurs pour chaque action
        self.action_colors = {
            'fold': '#780000',     # Rouge Sang
            'check': '#C1121F',    # Rouge
            'call': '#FDF0D5',     # Beige
            'raise': '#669BBC',    # Bleu
            'all-in': '#003049'    # Bleu Nuit
        }

    def plot_progress(self, dpi=500):
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

        # Créer une figure avec 6 sous-graphiques (2x3)
        fig = plt.figure(figsize=(25, 20))
        
        # Définir une palette de couleurs pastel
        pastel_colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
        
        # 1. Average mbb/hand par agent
        ax1 = plt.subplot(2, 3, 1)
        window = self.plot_interval * 3 
        agents = PLAYERS
        mbb_data = {agent: [] for agent in agents}
        for episode_num, episode in states_data.items():
            for state in episode:
                if state["phase"] == "showdown":            
                    # Pour chaque agent, ajouter son stack change en mbb
                    for agent in agents:
                        if agent in state["stack_changes"]:
                            stack_change = state["stack_changes"][agent]
                            mbb_data[agent].append(stack_change * 1000)  # Conversion en mbb
                        else:
                            raise Exception(f"Agent {agent} n'a pas de showdown dans les metrics même vide")

        # Tracer les moyennes mobiles pour chaque agent
        for i, (agent, data) in enumerate(mbb_data.items()):
            rolling_avg = pd.Series(data).rolling(window=window, min_periods=1).mean()
            ax1.plot(rolling_avg, label=agent, color=pastel_colors[i], linewidth=3)
        
        ax1.set_title('Average mbb/hand par agent')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('mbb/hand')
        ax1.legend()
        
        ax1.set_facecolor('#F0F0F0')  # Fond légèrement gris
        ax1.grid(True, alpha=0.3)
        
        # 2. Fréquence des actions par agent
        ax3 = plt.subplot(2, 3, 2)
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
        ax2 = plt.subplot(2, 3, 3)
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
        ax4 = plt.subplot(2, 3, 4)
        episodes = sorted([int(k) for k in metrics_data.keys()])
        epsilon_values = [np.clip(self.start_epsilon * self.epsilon_decay ** episode, 0.05, self.start_epsilon) for episode in episodes]
        ax4.plot(episodes, epsilon_values, color='#2E86AB', linewidth=2)  # Bleu foncé pour epsilon
        ax4.set_title('Evolution de Epsilon')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_ylim(0, 1)
        
        ax4.set_facecolor('#F0F0F0')
        ax4.grid(True, alpha=0.3)

        # 5. Rewards par agent
        ax5 = plt.subplot(2, 3, 5)  # Position in bottom right
        

        # Tracer les récompenses pour chaque agent
        rewards_window = self.plot_interval * 3
        for i, agent in enumerate(agents):
            episodes = []
            rewards = []
            
            for episode_num, episode_metrics in metrics_data.items():
                if 'reward' in episode_metrics[i]:
                    episodes.append(int(episode_num))
                    rewards.append(float(episode_metrics[i]['reward']))
            
            if rewards:
                rolling_avg = pd.Series(rewards).rolling(window=rewards_window, min_periods=1).mean()
                ax5.plot(episodes, rolling_avg, 
                        label=f"{agent} Reward", 
                        color=pastel_colors[i],
                        linewidth=2)
        
        ax5.set_title('Evolution des Récompenses par Agent')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Reward')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_facecolor('#F0F0F0')

        # 6. Winrate par agent
        ax6 = plt.subplot(2, 3, 6)

        # Collecter et trier les agents
        window = self.plot_interval * 3

        # Préparer les données pour chaque agent
        agent_results = {agent: [] for agent in agents}

        for episode_num, episode in states_data.items():
            # Trouver le dernier état showdown qui contient les stack_changes finaux
            showdown_states = [s for s in episode if s["phase"].lower() == "showdown"]
                
            last_showdown_state = showdown_states[-1]
            
            # Déterminer les gagnants basés sur les stack_changes
            stack_changes = last_showdown_state["stack_changes"]
                
            # Trouver le gain maximum
            max_gain = max(stack_changes.values())
            winners = [player for player, gain in stack_changes.items() if gain == max_gain]
            players = PLAYERS
            
            # Distribuer les résultats (1 pour victoire, 0 pour défaite)
            win_share = 1.0 / len(winners) if winners else 0
            for agent in agents:
                if agent in players:
                    result = win_share if agent in winners else 0
                    agent_results[agent].append(result)

        # Tracer les courbes de winrate pour chaque agent
        for i, (agent, results) in enumerate(agent_results.items()):
            # Convertir en pandas Series pour gérer les valeurs manquantes
            series = pd.Series(results)
            # Calculer la moyenne mobile en ignorant les valeurs manquantes
            rolling_winrate = series.rolling(window=window, min_periods=1).mean()
            
            ax6.plot(rolling_winrate, 
                     label=f"{agent}", 
                     color=pastel_colors[i],
                     linewidth=2)

        ax6.set_title('Winrate par agent (moyenne mobile sur 1000 épisodes)')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Winrate')
        ax6.legend()
        ax6.set_ylim(0, 1)

        ax6.set_facecolor('#F0F0F0')
        ax6.grid(True, alpha=0.3)

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
        plt.savefig('viz_json/Poker_progress.jpg', dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_metrics(self):
        """
        Génère des visualisations des métriques d'entraînement à partir du fichier metrics_history.json
        """
        # Charger les données
        metrics_path = os.path.join(self.output_dir, "metrics_history.json")
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        # Créer une figure avec 4 sous-graphiques
        fig = plt.figure(figsize=(20, 15))
        
        # Définir une palette de couleurs pastel
        pastel_colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
        
        # Extraire les agents uniques (maintenant basé sur l'index dans la liste des métriques)
        agents = PLAYERS

        # Métriques spécifiques à tracer
        metrics_to_plot = ['entropy_loss', 'value_loss', 'loss', 'std']

        # Créer un subplot pour chaque métrique
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = plt.subplot(2, 2, idx + 1)
            
            # Préparer les données pour chaque agent
            for agent_idx, agent in enumerate(agents):
                episodes = []
                values = []
                
                for episode_num, episode_metrics in metrics_data.items():
                    if metric_name in episode_metrics[agent_idx]:  # Utiliser l'index de l'agent
                        episodes.append(int(episode_num))
                        values.append(float(episode_metrics[agent_idx][metric_name]))
                
                # Calculer la moyenne mobile
                window = self.plot_interval * 3
                if len(values) > 0:
                    rolling_avg = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    ax.plot(episodes, rolling_avg, 
                           label=agent, 
                           color=pastel_colors[agent_idx % len(pastel_colors)],
                           linewidth=2)

            # Personnaliser les titres selon la métrique
            title_mapping = {
                'entropy_loss': 'Perte d\'entropie',
                'value_loss': 'Perte de valeur',
                'loss': 'Perte totale',
                'std': 'Écart-type'
            }
            
            ax.set_title(f'Evolution de {title_mapping.get(metric_name, metric_name)}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_name)
            ax.legend()
            
            # Style du subplot
            ax.set_facecolor('#F0F0F0')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Style global
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
        plt.savefig('viz_json/Poker_metrics.jpg', dpi=500, bbox_inches='tight')
        plt.close()

    def plot_analytics(self):
        """
        Génère des visualisations analytiques avancées du jeu de poker
        """
        # Charger les données
        states_path = os.path.join(self.output_dir, "episodes_states.json")
        with open(states_path, 'r') as f:
            states_data = json.load(f)

        # Créer une figure avec 8 sous-graphiques (2x4)
        fig = plt.figure(figsize=(25, 20))
        # Définir une palette de couleurs pastel cohérente
        position_colors = {
            "SB": '#003049',
            "BB": '#006DAA',
            "UTG": '#D62828',
            "MP": '#F77F00',
            "CO": '#FCBF49',
            "BTN": '#EAE2B7'
        }

        # 1. Win Rate par position et par agent (Bar Plot)
        ax1 = plt.subplot(1, 1, 1)
        position_wins = defaultdict(lambda: defaultdict(int))
        position_games = defaultdict(lambda: defaultdict(int))
        
        # Collect union of all players that were active (i.e. had a showdown state) across all episodes
        players = PLAYERS
        for episode in states_data.values():            
            # Get the last showdown state to access stack changes
            showdown_states = [s for s in episode if s["phase"].lower() == "showdown"]
            if not showdown_states:
                continue
                
            last_showdown_state = showdown_states[-1]  # Get the last showdown state
            winner = max(last_showdown_state["stack_changes"], 
                        key=last_showdown_state["stack_changes"].get) if last_showdown_state["stack_changes"] else None

            # Count positions only for active players in this episode
            for state in episode:
                player = state["player"]
                if player not in players:
                    continue  # Skip inactive players for this episode
                positions = state["state_vector"]["relative_positions"]
                position_idx = positions.index(1.0)
                position_name = ["SB", "BB", "UTG", "MP", "CO", "BTN"][position_idx]
                
                position_games[player][position_name] += 1
                if player == winner:
                    position_wins[player][position_name] += 1
        
        # Préparer les données pour le bar plot using active players
        positions = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        bar_width = 0.15
        index = np.arange(len(players))

        # Tracer un groupe de barres pour chaque position
        for i, position in enumerate(positions):
            win_rates = []
            for player in players:
                games = position_games[player][position]
                wins = position_wins[player][position]
                win_rates.append(wins / games if games > 0 else 0)
            
            bars = ax1.bar(index + i * bar_width, win_rates, bar_width, 
                         label=position, alpha=0.8, color=position_colors[position])
            
            # Ajouter les pourcentages au-dessus des barres
            for j, v in enumerate(win_rates):
                ax1.text(index[j] + i * bar_width, v + 0.01, 
                        f'{v:.0%}', ha='center', va='bottom', 
                        fontsize=8, rotation=90)

        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate par Agent et par Position')
        ax1.set_xticks(index + bar_width * (len(positions)-1) / 2)
        ax1.set_xticklabels(players)
        ax1.legend(title='Position')
        ax1.set_ylim(0,1)
        
        plt.tight_layout()
        plt.savefig('viz_json/Poker_analytics.jpg', dpi=500, bbox_inches='tight')
        plt.close()

    def plot_heatmaps(self):
        # Charger les données
        states_path = os.path.join(self.output_dir, "episodes_states.json")
        with open(states_path, 'r') as f:
            states_data = json.load(f)

        # Créer une figure plus grande avec plus d'espace entre les subplots
        fig = plt.figure(figsize=(30, 20))  # Increased figure size
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # More space between subplots
        
        players = PLAYERS
        player_colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']

        # 1-6. Range win rate heat maps pour chaque agent
        for i, player in enumerate(players):
            ax = plt.subplot(2, 3, i + 1)
            
            # Initialiser la matrice 13x13 pour les combinaisons de cartes
            hand_matrix = np.zeros((13, 13))
            hand_counts = np.zeros((13, 13))            
            # Process only episodes where this player was active
            for episode in states_data.values():                                
                # Determine winner using showdown states
                winners = [s["player"] for s in episode if s["phase"].lower() == "showdown" and s["stack_changes"][s["player"]] > 0]
                if not winners:
                    continue
                winner = winners[0]
                
                # Get player's cards from their first showdown state
                player_states = [s for s in episode if s["player"] == player and s["phase"].lower() == "showdown"]
                if not player_states:
                    continue
                
                first_state = player_states[0]
                state_vector = first_state["state_vector"]
                player_cards = state_vector["player_cards"]
                
                # Récupérer la première et deuxième carte
                card1 = player_cards[0]
                card1_value = int(round(card1["value"] * 14 + 2))  # Denormalise la valeur de la première carte
                card1_suit = card1["suit"].index(1) if 1 in card1["suit"] else -1
                card2 = player_cards[1]
                card2_value = int(round(card2["value"] * 14 + 2))  # Denormalise la valeur de la deuxième carte
                card2_suit = card2["suit"].index(1) if 1 in card2["suit"] else -1
                
                if card1_value < 2 or card2_value < 2:
                    continue
                    
                try:
                    # Convertir en indices 0-12 (2->0, A->12)
                    card1_idx = card1_value - 2
                    card2_idx = card2_value - 2
                    
                    if card1_idx != -1 and card2_idx != -1:
                        # Utiliser le winner déterminé précédemment
                        won = (player == winner)
                        
                        is_suited = (card1_suit == card2_suit)
                        
                        if card1_idx != card2_idx:
                            if is_suited:
                                row = min(card1_idx, card2_idx)
                                col = max(card1_idx, card2_idx)
                            else:
                                row = max(card1_idx, card2_idx)
                                col = min(card1_idx, card2_idx)
                        else:
                            row = card1_idx
                            col = card2_idx
                        
                        hand_matrix[row][col] += 1 if won else 0
                        hand_counts[row][col] += 1
                except (ValueError, IndexError):
                    continue

            # Calculer les win rates en évitant la division par zéro
            win_rates = np.zeros((13, 13))
            mask = hand_counts > 0
            win_rates[mask] = hand_matrix[mask] / hand_counts[mask]
            
            empty_mask = hand_counts == 0
            card_labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']  # Ordre inverse
            
            # Flip matrices pour correspondre aux étiquettes
            win_rates = np.flip(win_rates, axis=(0, 1))
            empty_mask = np.flip(empty_mask, axis=(0, 1))
            
            sns.heatmap(win_rates,
                       mask=empty_mask,  # masquer les cellules sans données
                       annot=True,  # afficher les valeurs
                       fmt='.2f',
                       cmap='RdYlBu_r',
                       xticklabels=card_labels,
                       yticklabels=card_labels,
                       ax=ax,
                       cbar_kws={'label': 'Win Rate'},
                       vmin=0,
                       vmax=1)
            
            # Ajouter des annotations textuelles pour les mains
            for y in range(13):
                for x in range(13):
                    if not empty_mask[y, x]:
                        first_card = card_labels[y]
                        second_card = card_labels[x]
                        if x == y:
                            hand_text = f"{second_card}{first_card}"
                        elif x > y:  # Suited
                            hand_text = f"{first_card}{second_card}s"
                        else:  # Offsuit
                            hand_text = f"{second_card}{first_card}o"
                            
                        ax.text(x + 0.5, y + 0.2, hand_text,
                              ha='center', va='center', 
                              color='white', alpha=0.7, 
                              fontsize=8, fontweight='bold')

            ax.set_xticklabels(card_labels, fontsize=10, rotation=0)
            ax.set_yticklabels(card_labels, fontsize=10, rotation=0)
            ax.set_title(f'Win Rate Matrix - {player}',
                         color=player_colors[i],
                         pad=20,
                         fontsize=12)

        plt.suptitle('Hand Win Rates by Player\n(s: suited, o: offsuit)', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('viz_json/Poker_heatmaps.jpg', dpi=500, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualizer = Visualizer(start_epsilon=0.8, epsilon_decay=0.9999, plot_interval=500, save_interval=250)
    visualizer.plot_progress()
    visualizer.plot_metrics()
    visualizer.plot_analytics()
    visualizer.plot_heatmaps()