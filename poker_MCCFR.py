# poker_MCCFR.py
import numpy as np
import random as rd
import torch
from utils.config import DEBUG
from typing import List, Dict, Tuple
from poker_game import PlayerAction, PokerGame
from poker_game_optimized import PokerGameOptimized
import time
class MCCFRTrainer:
    """
    Implémente l'algorithme Monte Carlo Counterfactual Regret Minimization (MCCFR)
    pour l'apprentissage de stratégies GTO au poker.
    """
    
    def __init__(self, num_simulations: int):
        self.num_simulations = num_simulations

    def compute_expected_payoffs_and_target_vector(self, valid_actions: List[PlayerAction], flat_state_and_count: Tuple[List, int]) -> Tuple[np.ndarray, Dict[PlayerAction, float]]:
        """
        Simule le futur d'une partie en parcourant les trajectoires des actions valides.
        """
        # Initialiser a None pour toutes les actions non valides et 0 pour les actions valides
        self.payoff_per_trajectory_action = {action: None for action in PlayerAction}
        for action in valid_actions:
            self.payoff_per_trajectory_action[action] = 0
        
        # Créer une instance de PokerGameOptimized avec le flat state
        replicated_game = PokerGameOptimized(flat_state_and_count)
        hero_name = replicated_game.simple_state['hero_name']

        # ---- Action abstraction ----
        real_valid_actions = valid_actions # On sauvegarde les actions valides réelles
        raise_actions = []
        raise_mapping = {}
        if PlayerAction.RAISE in valid_actions:
            # Identifier toutes les actions de raise disponibles et leurs pourcentages
            raise_percentages = {
                PlayerAction.RAISE: 0.1,  # Raise minimum (2x)
                PlayerAction.RAISE_25_POT: 0.25,
                PlayerAction.RAISE_33_POT: 0.33,
                PlayerAction.RAISE_50_POT: 0.50,
                PlayerAction.RAISE_66_POT: 0.66,
                PlayerAction.RAISE_75_POT: 0.75,
                PlayerAction.RAISE_100_POT: 1.00,
                PlayerAction.RAISE_125_POT: 1.25,
                PlayerAction.RAISE_150_POT: 1.50,
                PlayerAction.RAISE_175_POT: 1.75,
                PlayerAction.RAISE_2X_POT: 2.00,
                PlayerAction.RAISE_3X_POT: 3.00
            }
            
            # Filtrer les raises disponibles
            available_raises = [action for action in valid_actions if action.value.startswith("raise")]
            
            if available_raises:
                # Trier les raises par pourcentage
                available_raises.sort(key=lambda x: raise_percentages.get(x, 0))
                
                # Sélectionner 3 raises représentatives (début, milieu, fin)
                if len(available_raises) >= 3:
                    raise_actions = [
                        available_raises[0],  # Plus petite raise
                        available_raises[len(available_raises)//2],  # Raise médiane
                        available_raises[-1]  # Plus grande raise
                    ]
                else:
                    raise_actions = available_raises  # Garder toutes si moins de 3
                
                # Créer un mapping pour les autres raises
                raise_mapping = {}
                if len(available_raises) > 3:
                    median_idx = len(available_raises) // 2
                    for i, action in enumerate(available_raises):
                        if action not in raise_actions:
                            if i < median_idx:
                                raise_mapping[action] = raise_actions[0]  # Mapper aux petites raises
                            elif i == median_idx:
                                raise_mapping[action] = raise_actions[1]  # Mapper aux raises moyennes
                            else:
                                raise_mapping[action] = raise_actions[2]  # Mapper aux grandes raises
                
                # Mettre à jour valid_actions pour ne garder que les raises sélectionnées
                valid_actions = [action for action in valid_actions if not action.value.startswith("raise") or action in raise_actions]

        # Simulation des trajectoires7
        if DEBUG:
            print(f"Hero name: {hero_name}")
        start_time = time.time()
        for simulation_index in range(self.num_simulations):
            if DEBUG:
                print(f"Simulation [{simulation_index + 1}/{self.num_simulations}] pour {len(valid_actions)} actions")
                print('valid_actions :', valid_actions)
            rd_opponents_cards, rd_missing_community_cards = self.get_opponent_hands_and_community_cards(replicated_game.simple_state)
            for trajectory_action in valid_actions:
                # Créer une nouvelle instance pour chaque trajectoire
                game_copy = PokerGameOptimized(flat_state_and_count)
                payoff = game_copy.play_trajectory(trajectory_action, rd_opponents_cards, rd_missing_community_cards, valid_actions)
                self.payoff_per_trajectory_action[trajectory_action] += payoff / self.num_simulations

        end_time = time.time()

        if DEBUG:
            print('----------------------------------')
            print('real valid actions:', [real_valid_action.value for real_valid_action in real_valid_actions])
            print('explored actions:', [valid_action.value for valid_action in valid_actions])
            print('payoff_per_trajectory_action non mappé:', self.payoff_per_trajectory_action.values())

        # Afficher le temps de simulation (Non debug)
        print(f"Temps de simulation: {(end_time - start_time)*1000:.2f} ms")

        # Appliquer les payoffs aux raises non simulées
        if raise_mapping:
            for action, mapped_action in raise_mapping.items():
                self.payoff_per_trajectory_action[action] = self.payoff_per_trajectory_action[mapped_action]

        target_vector = self.compute_target_probability_vector(self.payoff_per_trajectory_action)
        
        if DEBUG:
            print('target_vector :', target_vector)
            print('payoff_per_trajectory_action :', self.payoff_per_trajectory_action.values())
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
    

    def compute_target_probability_vector(self, payoffs: Dict[PlayerAction, float]) -> np.ndarray:
        """
        Calcule le vecteur de probabilité cible basé sur les payoffs estimés.
        Convertit automatiquement les valeurs None en le minimum des valeurs non-None.
        """
        # Convertir les valeurs None en le minimum des valeurs non-None
        non_none_values = [v for v in payoffs.values() if v is not None]
        if not non_none_values:
            # Si toutes les valeurs sont None, retourner une distribution uniforme
            return np.ones(len(PlayerAction), dtype=np.float32) / len(PlayerAction)
        
        min_non_none = min(non_none_values)
        payoffs_cleaned = {k: min_non_none if v is None else v for k, v in payoffs.items()}
        
        # Normaliser les probabilités
        payoffs_vector = np.array(list(payoffs_cleaned.values()), dtype=np.float32)
        shifted_payoffs = payoffs_vector - np.min(payoffs_vector)
        if np.sum(shifted_payoffs) == 0:
            return np.ones(len(PlayerAction), dtype=np.float32) / len(PlayerAction)
        
        target_probs = shifted_payoffs / np.sum(shifted_payoffs)
        return target_probs

