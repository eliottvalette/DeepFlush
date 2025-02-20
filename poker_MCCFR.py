import numpy as np
import random as rd
import torch
from typing import List, Dict, Tuple, Optional
from poker_game import PokerGame, PlayerAction, GamePhase, Player
from collections import defaultdict

class MCCFRTrainer:
    """
    Implémente l'algorithme Monte Carlo Counterfactual Regret Minimization (MCCFR)
    pour l'apprentissage de stratégies GTO au poker.
    """
    
    def __init__(self, player : Player, num_simulations: int = 100000):
        self.player = player
        self.num_simulations = num_simulations

    def compute_expected_payoffs(self, current_state, valid_actions: List[PlayerAction]) -> Tuple[np.ndarray, Dict[PlayerAction, float]]:
        """
        Simule le futur d'une partie en parcourant les trajectoires des actions valides.
        """
        state_info = self.extract_simple_game_state(current_state)

        self.payoff_per_trajectory_action = defaultdict(float)

        for _ in range(self.num_simulations):
            rd_opponents_cards, rd_missing_community_cards = self.get_opponent_hands_and_community_cards(state_info)

            for trajectory_action in valid_actions:
                payoff = self.play_trajectory(trajectory_action, rd_opponents_cards, rd_missing_community_cards)
                self.payoff_per_trajectory_action[trajectory_action] += payoff / self.num_simulations # Moyenne des payoffs

        target_vector = self.compute_target_probability_vector(self.payoff_per_trajectory_action)
        
        return target_vector, self.payoff_per_trajectory_action
        
    def extract_simple_game_state(self, current_state) -> Dict:
        """
        Extrait un état de jeu simplifié pour une simulation optimisée.
        """
        # Récupérer les cartes du joueur
        player_cards = self.extract_cards(current_state[:10])
        # Récupérer les cartes communes connues
        community_cards = self.extract_cards(current_state[10:35])
        
        # Détecter la phase actuelle du jeu
        phase_index = np.argmax(current_state[47:52])  # Index de la phase active
        game_phase = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"][phase_index]

        # Identifier les joueurs actifs et leurs stacks
        player_info = [
            {"player_id": i, "stack": current_state[53+i], "bet": current_state[59+i]}
            for i in range(6) if current_state[65+i] == 1
        ]
        
        return {
            "player_id": self.player.seat_position,
            "player_cards": player_cards,
            "community_cards": community_cards,
            "game_phase": game_phase,
            "players_info": player_info,
            "num_active_players": len(player_info)
        }

    def extract_cards(self, state_vector: np.ndarray) -> List[Tuple[int, str]]:
        """
        Extrait les cartes sous forme de tuples (valeur, couleur) à partir d'un vecteur d'état.
        """
        cards = []
        for i in range(0, len(state_vector), 5):  # Chaque carte est codée sur 5 indices
            if state_vector[i] == -1:
                continue  # Carte non définie
            
            value = int(state_vector[i] * 14 + 2)  # Dénormalisation de la valeur
            suit = ["♠", "♥", "♦", "♣"][np.argmax(state_vector[i+1:i+5])]
            cards.append((value, suit))
        
        return cards
    
    def get_remaining_deck(self, player_cards: List[Tuple[int, str]], community_cards: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
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
        used_cards = player_cards + community_cards
        remaining_deck = [card for card in deck if card not in used_cards]
        
        return remaining_deck
    
    def get_opponent_hands_and_community_cards(self, state_info: Dict):
        """
        Génère des mains aléatoires pour les adversaires et complète les cartes communes restantes.
        """
        known_cards = state_info['player_cards'] + state_info['community_cards']
        remaining_deck = self.get_remaining_deck(known_cards)

        num_opponents = state_info['num_active_players'] - 1
        missing_community_cards = 5 - len(state_info["community_cards"])

        # Échantillonnage des cartes restantes
        drawn_cards = rd.sample(remaining_deck, missing_community_cards + 2 * num_opponents)
        
        community_cards = drawn_cards[:missing_community_cards]
        opponent_hands = [drawn_cards[missing_community_cards + i*2: missing_community_cards + i*2+2] for i in range(num_opponents)]

        return opponent_hands, community_cards

    
    def play_trajectory(self, trajectory_action: PlayerAction, rd_opponents_cards: List[str], rd_missing_community_cards: List[str]) -> float:    
        """
        Simule une trajectoire en prenant les actions valides jusqu'au showdown.
        Retourne uniquement le payoff du joueur cible.
        """
        game_clone = self.game.clone()
        game_clone.set_hidden_cards(rd_opponents_cards, rd_missing_community_cards)
        
        game_clone.step(trajectory_action)  # Appliquer l’action initiale
        
        while game_clone.current_phase != GamePhase.SHOWDOWN:
            current_player = game_clone.players[game_clone.current_player_seat]
            
            if current_player == self.player:
                action_probs = self.get_self_strategy(current_player, game_clone.get_state(), game_clone.get_valid_actions())
            else:
                action_probs = self.get_opponent_strategy(current_player, game_clone.get_state(), game_clone.get_valid_actions())

            next_action = np.random.choice(game_clone.get_valid_actions(), p=action_probs)
            game_clone.step(next_action)
        
        return game_clone.get_player_payoff(self.player)


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

