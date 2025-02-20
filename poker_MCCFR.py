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

    def extract_simple_game_state(self, current_state) -> Dict:
        """
        Extrait les informations essentielles du vecteur d'état de manière dévectorisée.
        
        Args:
            current_state: Le vecteur d'état du jeu
            
        Returns:
            Dict: Dictionnaire contenant les informations clés du state

        Example:
        {
            "player_id": 2,
            "player_cards": [(10, "♠"), (12, "♦")],
            "community_cards": [(11, "♠"), (13, "♦"), (14, "♣")],
            "game_phase": "FLOP",
            "valid_actions": ["FOLD", "CALL", "RAISE", "ALL_IN"],
            "players_info": [
                {
                    "player_id": 0,
                    "stack": 80,
                    "current_bet": 20,
                    "position": "SB"
                },
                {
                    "player_id": 1,
                    "stack": 80,
                    "current_bet": 20,
                    "position": "BB"
                },
                {
                    "player_id": 2,
                    "stack": 100,
                    "current_bet": 0,
                    "position": "UTG"
                }
            ],
            "num_active_players": 3
        }
        """
        # Extraction des cartes du joueur
        player_cards = []
        for i in range(2):  # 2 cartes
            base_idx = i * 5
            if current_state[base_idx] != -1:  # Si la carte existe
                value = int(current_state[base_idx] * 14 + 2)  # Dénormalisation de la valeur
                suit_vector = current_state[base_idx+1:base_idx+5].tolist()
                if 1 in suit_vector:
                    suit = ["♠", "♥", "♦", "♣"][suit_vector.index(1)]
                    player_cards.append((value, suit))

        # Extraction des cartes communes
        community_cards = []
        for i in range(5):  # 5 cartes max
            base_idx = 10 + (i * 5)
            if current_state[base_idx] != -1:  # Si la carte existe
                value = int(current_state[base_idx] * 14 + 2)
                suit_vector = current_state[base_idx+1:base_idx+5].tolist()
                if 1 in suit_vector:
                    suit = ["♠", "♥", "♦", "♣"][suit_vector.index(1)]
                    community_cards.append((value, suit))

        # Extraction de la phase de jeu
        phase_vector = current_state[47:52].tolist()
        phase = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"][phase_vector.index(1)]

        # Extraction des actions disponibles
        valid_actions = []
        action_names = ["FOLD", "CHECK", "CALL", "RAISE", "ALL_IN"]
        for i, is_valid in enumerate(current_state[77:82]):
            if is_valid == 1:
                valid_actions.append(action_names[i])

        # Extraction des informations sur les joueurs
        players_info = []
        positions = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
        
        for i in range(6):
            # Stack à l'index 53+i
            stack = float(current_state[53+i])
            # Mise actuelle à l'index 59+i
            current_bet = float(current_state[59+i])
            # État actif/inactif à l'index 65+i
            is_active = current_state[65+i] == 1
            
            # Position relative (one-hot vector de 71 à 77)
            position = None
            if i == self.player.seat_position:  # Pour le joueur courant uniquement
                pos_vector = current_state[71:77].tolist()
                position = positions[pos_vector.index(1)]
            
            if is_active:  # On n'ajoute que les joueurs actifs
                players_info.append({
                    "player_id": i,
                    "stack": stack,
                    "current_bet": current_bet,
                    "position": position
                })

        return {
            "player_id": self.player.seat_position,
            "player_cards": player_cards,
            "community_cards": community_cards,
            "game_phase": phase,
            "valid_actions": valid_actions,
            "players_info": players_info,
            "num_active_players": len(players_info)
            }
    
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
        Génère des mains aléatoires pour les adversaires et complète les cartes communes restantes de manière aléatoire en évitant les cartes déjà visibles.
        """
        player_cards = state_info['player_cards']
        community_cards = state_info['community_cards']
        num_opponents = state_info['num_active_players'] - 1
        missing_cards = 5 - len(state_info["community_cards"])
        
        remaining_deck = self.get_remaining_deck(player_cards, community_cards)

        extracted_cards = rd.sample(remaining_deck, missing_cards + 2*num_opponents)
        community_cards = extracted_cards[:missing_cards]
        opponent_hands = [extracted_cards[missing_cards+i*2:missing_cards+i*2+2] for i in range(num_opponents)]
        
        return opponent_hands, community_cards

    def estimate_future_payoffs(self, current_state, valid_actions: List[PlayerAction]) -> float:
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
        
        return self.payoff_per_trajectory_action
        
    def play_trajectory(self, trajectory_action: PlayerAction, rd_opponents_cards: List[str], rd_community_cards: List[str]) -> float:    
        """
        Simule une trajectoire en prenant les actions valides.
        """
        current_player = self.game.current_player
        current_state = self.game.current_state
        current_player_actions = []

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

        