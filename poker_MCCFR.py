import numpy as np
import random as rd
import copy
import gc
import json
from typing import List, Dict, Tuple
from poker_game import PlayerAction, PokerGame
from poker_game_optimized import PokerGameOptimized
from collections import defaultdict

class MCCFRTrainer:
    """
    Implémente l'algorithme Monte Carlo Counterfactual Regret Minimization (MCCFR)
    pour l'apprentissage de stratégies GTO au poker.
    """
    
    def __init__(self, num_simulations: int):
        self.num_simulations = num_simulations

    def compute_expected_payoffs_and_target_vector(self, valid_actions: List[PlayerAction], simple_game_state) -> Tuple[np.ndarray, Dict[PlayerAction, float]]:
        """
        Simule le futur d'une partie en parcourant les trajectoires des actions valides.
        """
        # Initialiser a 0 pour toutes les actions de PlayerAction
        self.payoff_per_trajectory_action = {action: 0 for action in PlayerAction}
        
        for simulation_index in range(self.num_simulations):
            print(f"\nSimulation [{simulation_index + 1}/{self.num_simulations}]")
            print(f"Hero name: {simple_game_state['hero_name']}")
            rd_opponents_cards, rd_missing_community_cards = self.get_opponent_hands_and_community_cards(simple_game_state)
            for trajectory_action in valid_actions:
                # Créer une nouvelle instance pour chaque trajectoire
                replicated_game = PokerGameOptimized(copy.deepcopy(simple_game_state))
                payoff = replicated_game.play_trajectory(trajectory_action, rd_opponents_cards, rd_missing_community_cards, valid_actions)
                self.payoff_per_trajectory_action[trajectory_action] += payoff / self.num_simulations
                del replicated_game


        target_vector = self.compute_target_probability_vector(self.payoff_per_trajectory_action)
        
        print('----------------------------------')
        print('valid_actions :', valid_actions)
        print('target_vector :', target_vector)
        print('payoff_per_trajectory_action : ')
        for action in [PlayerAction.FOLD, PlayerAction.CHECK, PlayerAction.CALL, PlayerAction.RAISE, PlayerAction.ALL_IN]:
            print(f'{action} : {self.payoff_per_trajectory_action[action]}')
        print('----------------------------------')
        
        return target_vector, self.payoff_per_trajectory_action
    
    def get_remaining_deck(self, known_cards: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """
        Retourne la liste des cartes restantes dans le deck.
        
        Args:
            player_cards: Liste des cartes du joueur
            community_cards: Liste des cartes communes
            
        Returns:
            List[Tuple[int, str]]: Liste des cartes restantes dans le deck
        """
        # Create full deck
        suits = ["♠", "♥", "♦", "♣"]
        values = range(2, 15)  # 2 to 14 (Ace)
        deck = [(value, suit) for value in values for suit in suits]
        
        # Remove cards that are already in play
        remaining_deck = [card for card in deck if card not in known_cards]
        
        return remaining_deck
    
    def get_opponent_hands_and_community_cards(self, state_info: Dict):
        """
        Génère des mains aléatoires pour les adversaires et complète les cartes communes restantes.
        """
        known_cards = state_info['hero_cards'] + state_info['community_cards']
        remaining_deck = self.get_remaining_deck(known_cards)

        num_opponents = state_info['num_active_players'] - 1
        nb_missing_community_cards = 5 - len(state_info["community_cards"])

        # Échantillonnage des cartes restantes
        drawn_cards = rd.sample(remaining_deck, nb_missing_community_cards + 2 * num_opponents)
        
        missing_community_cards = drawn_cards[:nb_missing_community_cards]
        opponent_hands = [drawn_cards[nb_missing_community_cards + i*2: nb_missing_community_cards + i*2+2] for i in range(num_opponents)]

        return opponent_hands, missing_community_cards

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
        if max_payoff > 0:
            positive_regrets = np.array([max(0, payoffs[action]) for action in PlayerAction])
            return positive_regrets / np.sum(positive_regrets)

        elif max_payoff <= 0:
            # Si on a en payoff [-100, -10, -50] on veut que le target_action_prob soit [0, 1, 0]
            # On prend donc l'action qui minimise la perte
            min_loss_action = min(payoffs.items(), key=lambda x: abs(x[1]))[0]
            target_probs = np.zeros(len(PlayerAction))
            action_list = list(PlayerAction)
            min_loss_idx = action_list.index(min_loss_action)
            target_probs[min_loss_idx] = 1.0
            return target_probs

