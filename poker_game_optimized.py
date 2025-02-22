# poker_game_optimized.py
from poker_game import PlayerAction, PokerGame
import random as rd
import numpy as np
from typing import List, Tuple, Dict

class PokerGameOptimized:
    def __init__(self, game: PokerGame):
        self.simple_game_state = game.get_simple_state()
        self.players = game.players
        self.num_players = len(self.players)
        self.num_active_players = self.simple_game_state['num_active_players']
        self.players_info = self.simple_game_state['players_info']
        self.community_cards = self.simple_game_state['community_cards']
        self.player_cards = self.simple_game_state['player_cards']
    
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
        
