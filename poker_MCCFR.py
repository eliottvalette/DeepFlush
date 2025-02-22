# poker_MCCFR.py
import numpy as np
import random as rd
import torch
from typing import List, Dict, Tuple
from poker_game import PlayerAction, PokerGame
from poker_game_optimized import PokerGameOptimized
from collections import defaultdict

class MCCFRTrainer:
    """
    Implémente l'algorithme Monte Carlo Counterfactual Regret Minimization (MCCFR)
    pour l'apprentissage de stratégies GTO au poker.
    """
    
    def __init__(self, env: PokerGame, num_simulations: int = 100):
        self.game = env
        self.num_simulations = num_simulations

    def compute_expected_payoffs_and_target_vector(self, valid_actions: List[PlayerAction]) -> Tuple[np.ndarray, Dict[PlayerAction, float]]:
        """
        Simule le futur d'une partie en parcourant les trajectoires des actions valides.
        """
        self.payoff_per_trajectory_action = defaultdict(float)
        self.replicated_game = PokerGameOptimized(self.game)

        for _ in range(self.num_simulations):
            for trajectory_action in valid_actions:
                payoff = self.replicated_game.play_trajectory(trajectory_action)
                self.payoff_per_trajectory_action[trajectory_action] += payoff / self.num_simulations # Moyenne des payoffs

        target_vector = self.compute_target_probability_vector(self.payoff_per_trajectory_action)
        
        return target_vector, self.payoff_per_trajectory_action

    def get_self_strategy(self, self_agent, self_state, valid_actions: List[PlayerAction]) -> np.ndarray:
        """
        Retourne la stratégie du joueur pour l'état donné.
        """
        _, _, action_probs = self_agent.get_action(self_state, valid_actions)
        return action_probs

    def get_opponent_strategy(self, opponent_agent, opponent_state, valid_actions: List[PlayerAction]) -> np.ndarray:
        """
        Retourne la stratégie de l'adversaire pour l'état donné.
        """
        _, _, action_probs = opponent_agent.get_action(opponent_state, valid_actions)
        action_probs_noised = self.add_noise_to_policy(action_probs)
        return action_probs_noised
                              
    def add_noise_to_policy(self, base_probs: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Applique du bruit sur la distribution de probabilités d'un adversaire pour éviter une prédiction déterministe.

        Args:
            base_probs (np.ndarray): Distribution de base
            noise_level (float): Niveau de bruit appliqué
        
        Returns:
            np.ndarray: Nouvelle distribution bruitée
        """
        noise = np.random.dirichlet(np.ones_like(base_probs)) * noise_level
        new_probs = base_probs * (1 - noise_level) + noise
        return new_probs / new_probs.sum()  # Normalisation
    
    import numpy as np

    def compute_target_probability_vector(self, payoffs: Dict[PlayerAction, float]) -> np.ndarray:
        """
        Calcule le vecteur de probabilité cible basé sur les regrets estimés.
        """
        max_payoff = max(payoffs.values())
        positive_regrets = np.array([max(0, max_payoff - payoffs[action]) for action in PlayerAction])

        if np.sum(positive_regrets) > 0:
            return positive_regrets / np.sum(positive_regrets)
        else:
            return np.ones(len(PlayerAction)) / len(PlayerAction)  # Uniforme si regrets nuls

