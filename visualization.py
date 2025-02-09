# visualization.py
import matplotlib
matplotlib.use('Agg')  # Use Agg backend that doesn't require GUI
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from poker_game import HandRank
import seaborn as sns

class TrainingVisualizer:
    def __init__(self, save_interval: int = 1000):
        # Create subplots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        
        # Create new figure for metrics
        self.metrics_fig, self.metrics_axs = plt.subplots(3, 2, figsize=(15, 12))
        self.metrics_axs = self.metrics_axs.flatten()
        
        # Configure plots
        self.ax1.set_title('Average Reward per Agent')
        self.ax2.set_title('Win Rate per Agent')
        self.ax3.set_title('Action Distribution per Agent')
        self.ax4.set_title('Hand Strength vs Win Rate Correlation')
        self.ax5.set_title('Epsilon Decay')
        
        # Configure new epsilon decay plot
        self.ax5.set_xlabel('Episodes')
        self.ax5.set_ylabel('Epsilon Value')
        self.ax5.grid(True)
        
        # Keep ax6 empty for now or hide it
        self.ax6.set_visible(False)
        
        # Initialize data
        self.window_size = 50
        self.rewards_data = {f"Agent {i+1}": [] for i in range(6)}
        self.wins_data = {f"Agent {i+1}": [] for i in range(6)}
        self.action_data = {f"Agent {i+1}": {
            'FOLD': [], 'CHECK': [], 'CALL': [], 'RAISE': [], 'ALL_IN': []
        } for i in range(6)}
        self.hand_strength_data = {f"Agent {i+1}": {'strength': [], 'win': []} for i in range(6)}
        self.episodes = []
        
        # Colors for agents and actions
        self.colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

        self.action_colors = {
            'FOLD': '#FF9999',    # Light red
            'CHECK': '#99FF99',   # Light green
            'CALL': '#9999FF',    # Light blue
            'RAISE': '#FFFF99',   # Yellow
            'ALL_IN': '#FF99FF'   # Pink
        }
        
        # Configure axes
        self.ax1.set_xlabel('Episodes')
        self.ax1.set_ylabel('Average Reward')
        self.ax2.set_xlabel('Episodes')
        self.ax2.set_ylabel('Win Rate (%)')
        self.ax3.set_xlabel('Episodes')
        self.ax3.set_ylabel('Action Distribution (%)')
        self.ax4.set_xlabel('Episodes')
        self.ax4.set_ylabel('Win Rate Correlation')
        
        # Add grids
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.grid(True)
        
        # Initialize lines for rewards and wins
        self.reward_lines = {}
        self.win_lines = {}
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.reward_lines[agent_name], = self.ax1.plot([], [], 
                                                         label=agent_name, 
                                                         color=self.colors[i])
            self.win_lines[agent_name], = self.ax2.plot([], [], 
                                                      label=agent_name, 
                                                      color=self.colors[i])
        
        self.ax1.legend()
        self.ax2.legend()
        
        # Initialize metrics tracking
        self.metrics_history = defaultdict(lambda: defaultdict(list))
        self.episodes_history = []
        
        # Configure metrics axes
        metric_titles = ['Approx KL', 'Entropy Loss', 'Value Loss', 
                        'Advantage STD', 'Learning Rate', 'Total Loss']
        for ax, title in zip(self.metrics_axs, metric_titles):
            ax.set_title(title)
            ax.grid(True)
            ax.set_xlabel('Episodes')
        
        # Add counter for file saves
        self.save_counter = 0
        self.save_interval = save_interval // 10
        
        # Initialize epsilon data
        self.epsilon_data = []
        self.epsilon_line, = self.ax5.plot([], [], 'k-', label='Epsilon')
        self.ax5.legend()

        # -----------------------------------------------------------------
        # Ajout d'un suivi des variations de stack (en milli big blind par main)
        # Utiliser ax6 (précédemment masqué) pour le graphique des variations de stack
        self.stack_variations = {f"Agent {i+1}": [] for i in range(6)}
        self.ax6.set_visible(True)
        self.ax6.set_title('Variation de Stack (mBB/h)')
        self.ax6.set_xlabel('Mains (x100)')
        self.ax6.set_ylabel('Variation (mBB/h)')
        self.stack_lines = {}
        for i, agent_name in enumerate(self.stack_variations.keys()):
            self.stack_lines[agent_name], = self.ax6.plot([], [], label=agent_name, color=self.colors[i])
        self.ax6.legend()
        # -----------------------------------------------------------------

        # Ajouter un dictionnaire pour suivre les win rates par hand rank
        self.hand_rank_stats = {
            f"Agent {i+1}": {rank: {'wins': 0, 'total': 0} for rank in HandRank} 
            for i in range(6)
        }

    def update_action_distribution(self, agent_name, action):
        """Update action distribution for an agent"""
        action_name = action.name if hasattr(action, 'name') else action
        for act in self.action_data[agent_name].keys():
            self.action_data[agent_name][act].append(1 if act == action_name else 0)

    def update_hand_strength_data(self, agent_name, strength, win_status):
        """Update hand strength and win data"""
        self.hand_strength_data[agent_name]['strength'].append(strength)
        self.hand_strength_data[agent_name]['win'].append(1 if win_status else 0)

    def update_hand_rank_stats(self, agent_name: str, hand_rank: HandRank, won: bool):
        """
        Met à jour les statistiques de win rate par hand rank pour un agent.
        
        Args:
            agent_name (str): Nom de l'agent
            hand_rank (HandRank): Le rang de la main finale ou au moment du fold
            won (bool): Si l'agent a gagné ou non
        """
        self.hand_rank_stats[agent_name][hand_rank]['total'] += 1
        if won:
            self.hand_rank_stats[agent_name][hand_rank]['wins'] += 1

    def update_hand_rank_data(self, final_hand_ranks):
        """
        Met à jour les statistiques de win rate par hand rank pour chaque agent.
        
        Args:
            final_hand_ranks (list): Liste de tuples (HandRank, bool) pour chaque agent.
        """
        for i, (rank, won) in enumerate(final_hand_ranks):
            agent_name = f"Agent {i+1}"
            self.update_hand_rank_stats(agent_name, rank, won)

    def plot_action_distribution(self):
        """Plot action distribution for each agent"""
        self.ax3.clear()
        self.ax3.set_title('Action Distribution per Agent')
        self.ax3.grid(True)
        
        bar_width = 0.15
        agent_positions = np.arange(len(self.rewards_data))
        total_width = bar_width * len(self.action_data['Agent 1'].keys())
        offset = total_width / 2 - bar_width / 2
        
        for action_idx, action_name in enumerate(self.action_data['Agent 1'].keys()):
            action_percentages = []
            for agent_name in self.rewards_data.keys():
                action_counts = self.action_data[agent_name][action_name]
                if action_counts:
                    percentage = sum(action_counts[-self.window_size:]) / min(len(action_counts), self.window_size) * 100
                else:
                    percentage = 0
                action_percentages.append(percentage)
            
            self.ax3.bar(agent_positions + action_idx * bar_width - offset,
                        action_percentages,
                        bar_width,
                        label=action_name,
                        color=self.action_colors[action_name])
        
        self.ax3.set_xticks(agent_positions)
        self.ax3.set_xticklabels(self.rewards_data.keys())
        self.ax3.legend()
        self.ax3.set_ylim(0, 100)

    def plot_hand_strength_analysis(self):
        """Plot hand strength correlation with wins"""
        self.ax4.clear()
        self.ax4.set_title('Hand Strength vs Win Rate Correlation')
        
        for i, (agent_name, color) in enumerate(zip(self.hand_strength_data.keys(), self.colors)):
            strengths = self.hand_strength_data[agent_name]['strength']
            wins = self.hand_strength_data[agent_name]['win']
            
            if len(strengths) > 0 and len(wins) > 0:
                correlations = []
                for j in range(len(strengths)):
                    start = max(0, j - self.window_size)
                    end = j + 1
                    
                    window_strengths = strengths[start:end]
                    window_wins = wins[start:end]
                    
                    # Check if we have enough variance in both arrays
                    if (len(window_strengths) > 1 and 
                        len(set(window_strengths)) > 1 and 
                        len(set(window_wins)) > 1):
                        try:
                            # Suppress numpy warnings during correlation calculation
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr = np.corrcoef(window_strengths, window_wins)[0,1]
                            # Check if correlation is valid
                            if corr is not None and not np.isnan(corr):
                                correlations.append(corr)
                            else:
                                correlations.append(0)
                        except Exception:
                            correlations.append(0)
                    else:
                        correlations.append(0)
                
                if correlations:
                    self.ax4.plot(self.episodes[-len(correlations):], correlations, 
                                label=agent_name, color=color)
        
        self.ax4.legend()
        self.ax4.grid(True)
        self.ax4.set_ylim(-1, 1)
        self.ax4.axhline(0, color='gray', linestyle='--')

    def update_metrics(self, episode, metrics_list):
        """Update training metrics history"""
        self.episodes_history.append(episode)
        
        for i, metrics in enumerate(metrics_list):
            if metrics is not None:
                agent_name = f"Agent {i+1}"
                for metric_name, value in metrics.items():
                    self.metrics_history[agent_name][metric_name].append(value)

    def plot_metrics(self):
        """Plot training metrics"""
        metric_keys = ['approx_kl', 'entropy_loss', 'value_loss', 
                      'std', 'learning_rate', 'loss']
        
        for ax, metric_key in zip(self.metrics_axs, metric_keys):
            ax.clear()
            ax.set_title(metric_key.replace('_', ' ').title())
            ax.grid(True)
            
            for agent_name, metrics in self.metrics_history.items():
                if metric_key in metrics:
                    values = metrics[metric_key]
                    if len(values) > 0:
                        # Calculate moving average
                        window = min(50, len(values))
                        smoothed_values = np.convolve(values, 
                                                    np.ones(window)/window, 
                                                    mode='valid')
                        episodes = self.episodes_history[-len(smoothed_values):]
                        
                        color = self.colors[int(agent_name[-1])-1]
                        ax.plot(episodes, smoothed_values, 
                               label=agent_name, color=color)
            
            ax.legend()
            ax.set_xlabel('Episodes')
        
        plt.tight_layout()
        self.metrics_fig.savefig('viz_pdf/training_metrics.jpg')

    def plot_hand_rank_win_rates(self):
        """
        Crée un bar plot des win rates par hand rank pour chaque agent dans une seule figure.
        Utilise toutes les données sans fenêtrage.
        """
        # Créer une figure avec 6 subplots (pour 6 agents)
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        axs = axs.flatten()
        
        for idx, (agent_name, stats) in enumerate(self.hand_rank_stats.items()):
            ax = axs[idx]
            # Préparer les données pour toutes les ranks
            ranks = [rank.name for rank in HandRank]
            win_rates = []
            total_hands = []
            win_counts = []  # Nouveau: stocker (nombre gagné, nombre perdu)
            for rank in HandRank:
                total = stats[rank]['total']
                wins = stats[rank]['wins']
                losses = total - wins
                # Utilisation de toutes les données pour calculer le win rate
                win_rate = (wins / total * 100) if total > 0 else 0
                win_rates.append(win_rate)
                total_hands.append(total)
                win_counts.append((wins, losses))
            
            # Créer le bar plot sur l'axe correspondant
            bars = ax.bar(ranks, win_rates, color='skyblue')
            ax.set_title(f'{agent_name} Win Rates by Hand Rank')
            ax.set_xlabel('Hand Rank')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)
            
            # Rotation des labels pour une meilleure lisibilité
            ax.set_xticks(np.arange(len(ranks)))
            ax.set_xticklabels(ranks, rotation=45, ha='right')
            
            # Ajouter les valeurs sur les barres avec le win rate et le tuple (gagné, perdu)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ylim = ax.get_ylim()[1]
                if height > 0.95 * ylim:
                    va = 'top'
                    text_color = 'white'
                    offset = -2
                else:
                    va = 'bottom'
                    text_color = 'black'
                    offset = 2
                ax.text(
                    bar.get_x() + bar.get_width()/2., height + offset,
                    f'{win_rates[i]:.1f}%\n({win_counts[i][0]}, {win_counts[i][1]})',
                    ha='center', va=va, color=text_color, fontweight='bold', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                )
            ax.grid(True)
        
        plt.tight_layout()
        fig.savefig('viz_pdf/hand_rank_win_rates_all_agents.jpg')
        plt.close(fig)

    def update_plots(self, episode, rewards, wins, actions_dict, hand_strengths, metrics_list=None, epsilon=None, hand_variations=None):
        """Update all plots with new data"""
        self.episodes.append(episode)
        
        # Update rewards and wins
        for i, agent_name in enumerate(self.rewards_data.keys()):
            self.rewards_data[agent_name].append(rewards[i])
            self.wins_data[agent_name].append(wins[i])
            
            # Update action distribution if actions are provided
            if actions_dict and agent_name in actions_dict and actions_dict[agent_name]:
                for action in actions_dict[agent_name]:
                    self.update_action_distribution(agent_name, action)
            
            # Update hand strength data with win status
            if hand_strengths and i < len(hand_strengths):
                self.update_hand_strength_data(agent_name, hand_strengths[i], wins[i])
            
            # Calculate moving average of rewards
            if len(self.rewards_data[agent_name]) >= self.window_size:
                moving_avg = []
                for j in range(1, len(self.rewards_data[agent_name])+1):
                    # Get window of data
                    start_idx = max(0, j-self.window_size)
                    window_data = self.rewards_data[agent_name][start_idx:j]
                    
                    # Remove extreme values (0.5% from each end)
                    if len(window_data) > 4:  # Only trim if we have enough data points
                        window_data = np.array(window_data)
                        lower_percentile = np.percentile(window_data, 0.5)
                        upper_percentile = np.percentile(window_data, 99.5)
                        trimmed_data = window_data[(window_data >= lower_percentile) & 
                                                 (window_data <= upper_percentile)]
                        avg = np.mean(trimmed_data) if len(trimmed_data) > 0 else np.mean(window_data)
                    else:
                        avg = np.mean(window_data)
                    
                    moving_avg.append(avg)
                
                self.reward_lines[agent_name].set_data(self.episodes, moving_avg)
            
            # Calculate cumulative win rate
            if len(self.wins_data[agent_name]) > 0:
                win_rates = [
                    sum(self.wins_data[agent_name][:j+1])/(j+1) * 100
                    for j in range(len(self.wins_data[agent_name]))
                ]
                self.win_lines[agent_name].set_data(self.episodes, win_rates)
        
        # Update additional plots
        self.plot_action_distribution()
        self.plot_hand_strength_analysis()
        
        # Adjust axis limits for rewards and wins
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()
        self.ax2.set_ylim([-5, 105])
        
        # Update and plot metrics if provided
        if metrics_list is not None:
            self.update_metrics(episode, metrics_list)
        
        # Update epsilon plot if epsilon is provided
        if epsilon is not None:
            self.epsilon_data.append(epsilon)
            self.epsilon_line.set_data(self.episodes, self.epsilon_data)
            self.ax5.relim()
            self.ax5.autoscale_view()
            self.ax5.set_ylim([-0.05, 1.05])  # Set y-axis limits for epsilon

        # -----------------------------------------------------------------
        # Mise à jour du graphique des variations de stack (mBB/h)
        # hand_variations : liste (pour chaque agent) de la variation de stack en BB pour la main actuelle
        if hand_variations is not None:
            block_size = 100  # Lissage sur 100 mains
            for i, agent_name in enumerate(self.stack_variations.keys()):
                # Convertir la variation en milli big blinds et stocker
                self.stack_variations[agent_name].append(hand_variations[i] * 1000)
                data = self.stack_variations[agent_name]
                num_complete_blocks = len(data) // block_size
                if num_complete_blocks > 0:
                    x_vals = list(range(1, num_complete_blocks + 1))
                    block_avgs = [np.mean(data[j*block_size:(j+1)*block_size]) for j in range(num_complete_blocks)]
                    self.stack_lines[agent_name].set_data(x_vals, block_avgs)
            self.ax6.relim()
            self.ax6.autoscale_view()
        # -----------------------------------------------------------------
        
        # Sauvegarder les graphiques à intervalles réguliers
        self.save_counter += 1
        if self.save_counter >= self.save_interval:
            plt.tight_layout()
            self.fig.savefig('viz_pdf/training_progress.jpg')
            self.plot_metrics()
            self.plot_hand_rank_win_rates()
            self.save_counter = 0

def plot_winning_stats(winning_history: dict, window_size: int = 50, save_path: str = "viz_pdf/poker_wins.jpg"):
    """
    Plot the total number of wins for each agent as bars.
    """
    plt.figure(figsize=(12, 6))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    agent_names = list(winning_history.keys())
    total_wins = []
    
    # Calculate total wins for each agent
    for agent_name in agent_names:
        wins = winning_history[agent_name]
        total_wins.append(sum(wins))
    
    # Create bar positions
    positions = np.arange(len(agent_names))
    
    # Create bars
    plt.bar(positions, total_wins, color=colors)
    
    # Customize the plot
    plt.title('Total Wins per Agent')
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    
    # Set x-axis labels to agent names
    plt.xticks(positions, agent_names)
    
    # Add value labels on top of each bar
    for i, v in enumerate(total_wins):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    plt.savefig(save_path)
    plt.close()

def update_rewards_history(rewards_history: dict, reward_list: list, agent_list: list) -> dict:
    for i, agent in enumerate(agent_list):
        if agent.name not in rewards_history:
            rewards_history[agent.name] = []
        rewards_history[agent.name].append(reward_list[i])
    return rewards_history

def update_winning_history(winning_history: dict, winning_list: list, agent_list: list) -> dict:
    for i, agent in enumerate(agent_list):
        if agent.name not in winning_history:
            winning_history[agent.name] = []
        winning_history[agent.name].append(winning_list[i])
    return winning_history