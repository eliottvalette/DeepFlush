# poker_game.py
"""
Texas Hold'em, No Limit, 6 max.
"""
import pygame
import random as rd
from enum import Enum
from typing import List, Dict, Optional, Tuple
import pygame.font
from collections import Counter, defaultdict
import numpy as np

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
POSITIONS = [
    (SCREEN_WIDTH-550, SCREEN_HEIGHT-250),     # Bas-Droite (Joueur 1)
    (420, SCREEN_HEIGHT-250),                  # Bas-Gauche (Joueur 2) 
    (80, (SCREEN_HEIGHT-150) / 2),             # Milieu-Gauche (Joueur 3)
    (420, 140),                                # Haut-Gauche (Joueur 4)
    (SCREEN_WIDTH-550, 140),                   # Haut-Droite (Joueur 5)
    (SCREEN_WIDTH-190, (SCREEN_HEIGHT-150)/2)  # Milieu-Droite (Joueur 6)
]

class Card:
    """ 
    Représente une carte à jouer avec une couleur et une valeur.
    """
    def __init__(self, suit: str, value: int):
        """
        Initialise une carte avec une couleur et une valeur.
        
        Args:
            suit (str): La couleur de la carte (♠, ♥, ♦, ♣)
            value (int): La valeur de la carte (2-14, où 14 est l'As)
        """
        self.suit = suit
        self.value = value
        
    def __str__(self):
        """
        Convertit la carte en représentation textuelle.
        
        Returns:
            str: Représentation textuelle de la carte (ex: "A♠")

        Exemple:
        >>> card = Card('♠', 14)
        >>> print(card)
        'A♠'
        """
        values = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        value_str = values.get(self.value, str(self.value))
        return f"{value_str}{self.suit}"

class HandRank(Enum):
    """
    Énumération des combinaisons possibles au poker, de la plus faible à la plus forte.
    """
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class PlayerAction(Enum):
    """
    Énumération des actions possibles pour un joueur pendant son tour.
    """
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all-in"

class GamePhase(Enum):
    """
    Énumération des phases d'une partie de poker, de la distribution au showdown.
    """
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

class Button:
    """
    Représente un bouton cliquable dans l'interface utilisateur du jeu.
    """
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int]):
        """
        Initialise un bouton avec sa position, sa taille, son texte et sa couleur.
        
        Args:
            x (int): Position X sur l'écran
            y (int): Position Y sur l'écran
            width (int): Largeur du bouton en pixels
            height (int): Hauteur du bouton en pixels
            text (str): Texte affiché sur le bouton
            color (Tuple[int, int, int]): Couleur RGB du bouton
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.is_hovered = False
        self.enabled = True

    def draw(self, screen, font):
        """
        Dessine le bouton sur l'écran avec les effets visuels appropriés.
        
        Args:
            screen: Surface Pygame sur laquelle dessiner le bouton
            font: Police Pygame pour le rendu du texte
        """
        if not self.enabled:
            # Griser les boutons désactivés
            color = (128, 128, 128)
            text_color = (200, 200, 200)
        else:
            color = (min(self.color[0] + 30, 255), min(self.color[1] + 30, 255), min(self.color[2] + 30, 255)) if self.is_hovered else self.color
            text_color = (255, 255, 255)
        
        pygame.draw.rect(screen, color, self.rect)
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class Player:
    """
    Représente un joueur de poker avec ses cartes, son stack et son état de jeu.
    """
    def __init__(self, name: str, stack: int, seat_position: int):
        """
        Initialise un joueur avec son nom, son stack de départ et sa position à la table.
        
        Args:
            name (str): Nom du joueur
            stack (int): Stack de départ en jetons
            position (int): Position à la table (0-5)
        """
        self.name = name
        self.stack = stack
        self.seat_position = seat_position # 0-5
        self.role_position = None # 0-5 (0 = SB, 1 = BB, 2 = UTG, 3 = HJ, 4 = CO, 5 = BU)
        self.cards: List[Card] = []
        self.is_active = True # True si le joueur a assez de fonds pour jouer (stack > big_blind)
        self.has_folded = False
        self.is_human = True # True si on veut voir les cartes du joueur
        self.is_all_in = False
        self.range = None # Range du joueur (à initialiser comme l'ensemble des mains possibles)
        self.current_bet = 0 # Montant de la mise actuelle du joueur
        self.x, self.y = POSITIONS[self.seat_position]
        self.has_acted = False # True si le joueur a fait une action dans la phase courante (nécessaire pour savoir si le tour est terminé, car si le premier joueur de la phase check, tous les jouers sont a bet égal et ca déclencherait la phase suivante)

class PokerGame:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
    def __init__(self, list_names: List[str]):
        """
        Initialise la partie de poker avec les joueurs et les blindes.
        
        Args:
            num_players (int): Nombre de joueurs (défaut: 6)
        """
        self.num_players = 6
                
        self.small_blind = 0.5
        self.big_blind = 1
        self.starting_stack = 100 # Stack de départ en BB
        self.pot = 0

        self.deck: List[Card] = self._create_deck()
        self.community_cards: List[Card] = []

        self.current_phase = GamePhase.PREFLOP
        self.players = self._initialize_players(list_names) # self.players est une liste d'objets Player

        self.button_seat_position = rd.randint(0, 5) # 0-5
        for player in self.players:
            player.role_position = (player.seat_position - self.button_seat_position - 1) % 6

        # Initialiser les variables d'état du jeu
        self.current_player_seat = (self.button_seat_position + 1) % 6 # 0-5 initialisé à SB
        self.current_maximum_bet = 0 # initialisé à 0 mais s'updatera automatiquement à BB après la mise de SB puis BB
        self.last_raiser_seat = None # initialisé à None, s'updatera automatiquement à BB après la mise de SB puis BB
        
        self.number_raise_this_game_phase = 0 # Nombre de raises dans la phase courante (4 max, 4 inclus)
        
        # Ajouter la gestion des side pots
        self.main_pot = 0
        self.side_pots = [0] * 4 # 4 side pots max (Pire des cas : 1 Main et 4 Side)
        
        # --------------------
        # Initialiser pygame et les variables d'interface
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("6-Max Poker")
        self.font = pygame.font.SysFont('Arial', 24)
        self.clock = pygame.time.Clock()

        # Initialiser les éléments de l'interface
        self.pygame_action_buttons = self._create_action_buttons()
        self.pygame_bet_slider = pygame.Rect(50, SCREEN_HEIGHT - 100, 200, 20)
        self.pygame_slider_bet_amount = self.big_blind # Valeur de la raise actualle déterminée grace a un slider (uniquement utilisé en jeu manuel)

        # Suivi de l'historique des actions
        self.pygame_action_history = []

        # Ajouter le suivi des informations du gagnant
        self.pygame_winner_info = None

        # Ajouter le timing d'affichage du gagnant
        self.pygame_winner_display_start = 0
        self.pygame_winner_display_duration = 2000  # 2 secondes en millisecondes
        # --------------------
        
        self.start_new_hand(first_hand = True)
        self._update_button_states()

    def reset(self):
        """
        Réinitialise complètement l'état du jeu pour une nouvelle partie.
        
        Returns:
            List[float]: État initial du jeu après réinitialisation
        """
        # Réinitialiser les variables d'état du jeu
        self.pot = 0
        self.deck = self._create_deck()
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP

        # Réinitialiser les variables de jeu
        self.button_seat_position = rd.randint(0, 5)  # 0-5 (c'est une toute nouvelle partie donc on réassigne la position du bouton aléatoirement)
        for player in self.players:
            player.role_position = (player.seat_position - self.button_seat_position - 1) % 6
        
        # Réinitialiser les états variables des joueurs
        for player in self.players:
            player.stack = self.starting_stack
            player.current_bet = 0
            player.cards = []
            player.is_active = True
            player.has_folded = False
            player.is_all_in = False
            player.range = None        
        
        # Initialiser les variables d'état du jeu
        self.current_player_seat = (self.button_seat_position + 1) % 6  # 0-5 initialisé à SB
        self.current_maximum_bet = 0  # initialisé à 0 mais s'updatera automatiquement à BB
        self.last_raiser_seat = None

        self.number_raise_this_game_phase = 0
        
        # Réinitialiser les pots
        self.main_pot = 0
        self.side_pots = [0] * 4

        # --------------------
        # Réinitialiser les variables d'interface Pygame
        self.pygame_winner_info = None
        self.pygame_winner_display_start = 0
        self.pygame_slider_bet_amount = self.big_blind
        self.pygame_action_history = []
        # --------------------

        self.start_new_hand(first_hand = True)
        self._update_button_states()
        
        return self.get_state()

    def _move_button(self):
        """
        Déplace le bouton vers le prochain joueur actif.
        Cette méthode incrémente la position du bouton en vérifiant que le joueur est actif.
        """
        next_seat = (self.button_seat_position + 1) % self.num_players
        # Si le joueur à la nouvelle position n'est pas actif, on continue à chercher.
        while not self.players[next_seat].is_active:
            next_seat = (next_seat + 1) % self.num_players
        self.button_seat_position = next_seat


    def start_new_hand(self, first_hand=False):
        """
        Distribue une nouvelle main sans réinitialiser la partie.
        
        Returns:
            L'état initial du jeu après distribution.
        """
        # Réinitialiser les variables d'état du jeu
        self.pot = 0
        self.deck = self._create_deck()  # Réinitialiser le deck
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP

        # Réinitialiser l'état des joueurs
        for player in self.players:
            player.cards = []
            player.current_bet = 0
            player.is_all_in = False
            player.has_folded = False
            player.range = None
            player.is_active = player.stack > 0

        # Vérifier qu'il y a au moins 2 joueurs actifs
        active_players = [player for player in self.players if player.is_active]
        if len(active_players) < 2:
            raise ValueError("Il doit y avoir au moins 2 joueurs pour continuer la partie")

        # Distribuer les cartes aux joueurs actifs
        self.deal_cards()

        # Pour les mains suivantes, déplacer le bouton vers le prochain joueur actif
        if not first_hand:
            self._move_button()

        # Construire une liste ordonnée des joueurs actifs en fonction de leur seat_position
        active_players = sorted([p for p in self.players if p.is_active],
                                key=lambda p: p.seat_position)

        # Trouver l'index du joueur possédant le bouton dans cette liste
        button_index = 0
        for i, player in enumerate(active_players):
            if player.seat_position == self.button_seat_position:
                button_index = i
                break

        # Réorganiser la liste pour que le premier joueur soit celui qui a le bouton
        ordered_players = active_players[button_index:] + active_players[:button_index]
        n = len(ordered_players)

        # Réattribuer les rôles en fonction du nombre de joueurs actifs
        if n == 2:
            # Heads-Up : le joueur en bouton joue la Small Blind
            ordered_players[0].role_position = 0  # Small Blind (SB)
            ordered_players[1].role_position = 5  # Button / Big Blind (BB)
        elif n == 3:
            ordered_players[0].role_position = 1  # Big Blind (BB)
            ordered_players[1].role_position = 5  # Button
            ordered_players[2].role_position = 0  # Small Blind (SB)
        elif n == 4:
            ordered_players[0].role_position = 2  # UTG (Under The Gun)
            ordered_players[1].role_position = 5  # Button
            ordered_players[2].role_position = 0  # Small Blind (SB)
            ordered_players[3].role_position = 1  # Big Blind (BB)
        elif n == 5:
            ordered_players[0].role_position = 3  # Cutoff (CO)
            ordered_players[1].role_position = 5  # Button
            ordered_players[2].role_position = 0  # Small Blind (SB)
            ordered_players[3].role_position = 1  # Big Blind (BB)
            ordered_players[4].role_position = 2  # UTG
        elif n == 6:
            ordered_players[0].role_position = 4  # Cutoff (CO)
            ordered_players[1].role_position = 5  # Button
            ordered_players[2].role_position = 0  # Small Blind (SB)
            ordered_players[3].role_position = 1  # Big Blind (BB)
            ordered_players[4].role_position = 2  # UTG
            ordered_players[5].role_position = 3  # Hijack (HJ)

        # Mettre à jour la position du bouton et du joueur courant en fonction des rôles réattribués
        for player in self.players:
            if player.is_active and player.role_position == 5:
                self.button_seat_position = player.seat_position
                break

        for player in self.players:
            if player.is_active and player.role_position == 0:
                self.current_player_seat = player.seat_position
                break

        # Initialiser les variables de jeu complémentaires
        self.current_maximum_bet = 0  # Sera mis à jour par les blinds
        self.last_raiser_seat = None
        self.number_raise_this_game_phase = 0

        # Réinitialiser les pots
        self.main_pot = 0
        self.side_pots = [0] * 4

        # Réinitialiser les variables d'interface (exemple avec Pygame)
        self.pygame_winner_info = None
        self.pygame_winner_display_start = 0
        self.pygame_slider_bet_amount = self.big_blind
        self.pygame_action_history = []

        self._update_button_states()
        self.deal_small_and_big_blind()

        return self.get_state()

    
    def deal_small_and_big_blind(self):
        """
        Méthode à run en début de main pour distribuer automatiquement les blindes
        """
        # Déterminer les positions SB et BB en se basant sur les rôles attribués
        sb_player = next((p for p in self.players if p.is_active and p.role_position == 0), None)
        bb_player = next((p for p in self.players if p.is_active and p.role_position == 1), None)
        
        if sb_player is None or bb_player is None:
            raise ValueError("Impossible de déterminer la position de la Small Blind ou Big Blind")
        
        sb_seat_position = sb_player.seat_position
        bb_seat_position = bb_player.seat_position

        # SB
        if self.players[sb_seat_position].stack < self.small_blind:
            self.players[sb_seat_position].is_all_in = True
            self.players[sb_seat_position].current_bet = self.players[sb_seat_position].stack  # Le bet du joueur n'ayant pas assez pour payer la SB devient son stack
            self.players[sb_seat_position].stack = 0  # Le stack du joueur est donc 0
            self.players[sb_seat_position].has_acted = True
        else:
            self.players[sb_seat_position].stack -= self.small_blind
            self.players[sb_seat_position].current_bet = self.small_blind
            self.players[sb_seat_position].has_acted = True

        self.current_maximum_bet = self.small_blind
        self._next_player()
        
        # BB
        if self.players[bb_seat_position].stack < self.big_blind:
            self.players[bb_seat_position].is_all_in = True
            self.players[bb_seat_position].current_bet = self.players[bb_seat_position].stack  # Le bet du joueur n'ayant pas assez pour payer la BB devient son stack
            self.players[bb_seat_position].stack = 0  # Le stack du joueur devient 0
            self.players[bb_seat_position].has_acted = True
        else:
            self.players[bb_seat_position].stack -= self.big_blind
            self.players[bb_seat_position].current_bet = self.big_blind
            self.players[bb_seat_position].has_acted = True
        
        self.current_maximum_bet = self.big_blind
        self._next_player()

    def _next_player(self):
        """
        Passe au prochain joueur actif et n'ayant pas fold dans le sens horaire.
        """
        self.current_player_seat = (self.current_player_seat + 1) % self.num_players
        while not self.players[self.current_player_seat].is_active or self.players[self.current_player_seat].has_folded:
            self.current_player_seat = (self.current_player_seat + 1) % self.num_players
    
    def check_phase_completion(self):
        """
        Vérifie si le tour d'enchères actuel est terminé.
        Le tour est terminé uniquement lorsque tous les joueurs actifs (et n'ayant pas foldé)
        ont déjà agi ET que pour chacun, la mise actuelle est égale à la mise maximale (ou qu'il est all-in).
        """
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        current_player = self.players[self.current_player_seat]

        # Cas particulier, au PREFLOP, si la BB est limpée, elle doit avoir un droit de parole
        print(f"\n\ncurrent_player = {current_player.name} role: {current_player.role_position}\n\n")
        if self.current_phase == GamePhase.PREFLOP and current_player.role_position == 2:
            return False
        
        # Si un seul joueur reste, le tour est terminé
        if len(active_players) == 1:
            return True
        
        for player in active_players:
            # Si le joueur n'a pas encore agi dans la phase, le tour n'est pas terminé
            if not player.has_acted:
                return False
            # Si le joueur n'a pas égalisé la mise maximale et n'est pas all-in, le tour n'est pas terminé
            if player.current_bet < self.current_maximum_bet and not player.is_all_in:
                return False
        
        return True

    def advance_phase(self):
        """
        Passe à la phase suivante du jeu (préflop -> flop -> turn -> river).
        Distribue les cartes communes appropriées et réinitialise les mises.
        """
        print(f"current_phase {self.current_phase}")
        
        # Check if all active players are all-in
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        all_in_players = [p for p in active_players if p.stack == 0]
        
        if len(all_in_players) == len(active_players) and len(active_players) > 1:
            print("All players are all-in - proceeding directly to showdown")
            # Deal all remaining community cards
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return
        
        # Normal phase progression
        if self.current_phase == GamePhase.PREFLOP:
            self.current_phase = GamePhase.FLOP
        elif self.current_phase == GamePhase.FLOP:
            self.current_phase = GamePhase.TURN
        elif self.current_phase == GamePhase.TURN:
            self.current_phase = GamePhase.RIVER
        
        # Increment round number when moving to a new phase
        self.number_raise_this_game_phase = 0
        
        # Deal community cards for the new phase
        self.deal_community_cards()
        
        # Réinitialiser les mises pour la nouvelle phase
        self.current_maximum_bet = 0
        for player in self.players:
            if player.is_active:
                player.current_bet = 0
                if not player.has_folded:
                    player.has_acted = False  # Réinitialisation du flag
        
        # Set first player after dealer button
        self.current_player_seat = (self.button_seat_position + 1) % self.num_players
        while not self.players[self.current_player_seat].is_active:
            self.current_player_seat = (self.current_player_seat + 1) % self.num_players
    
    def _update_button_states(self):
        """
        Met à jour l'état activé/désactivé des boutons d'action.
        Prend en compte la phase de jeu et les règles du poker.
        """
        current_player = self.players[self.current_player_seat]
        
        # Activer tous les boutons par défaut
        for button in self.pygame_action_buttons.values():
            button.enabled = True
        
        # ---- CHECK ----
        if self.current_phase == GamePhase.PREFLOP:
            # Le joueur ne peut check que si sa mise actuelle est égale à la mise courante
            if current_player.current_bet < self.current_maximum_bet:
                self.pygame_action_buttons[PlayerAction.CHECK].enabled = False
        else:
            # Post-flop: on peut check seulement si personne n'a misé
            if current_player.current_bet < self.current_maximum_bet:
                self.pygame_action_buttons[PlayerAction.CHECK].enabled = False

        # ---- FOLD ----
        if self.pygame_action_buttons[PlayerAction.CHECK].enabled:
            self.pygame_action_buttons[PlayerAction.FOLD].enabled = False

        # ---- CALL ----
        
        # Désactiver call si pas de mise à suivre ou pas assez de jetons
        if current_player.current_bet == self.current_maximum_bet:
            self.pygame_action_buttons[PlayerAction.CALL].enabled = False
        elif current_player.stack < (self.current_maximum_bet - current_player.current_bet):
            self.pygame_action_buttons[PlayerAction.CALL].enabled = False

        # ---- RAISE ----
        
        # Désactiver raise si pas assez de jetons pour la mise minimale
        min_raise = max(self.current_maximum_bet * 2, self.big_blind * 2)
        if current_player.stack + current_player.current_bet < min_raise:
            self.pygame_action_buttons[PlayerAction.RAISE].enabled = False

        # ---- ALL-IN ----
        
        # Désactiver raise si déjà 4 relances dans le tour
        if self.number_raise_this_game_phase >= 4:
            self.pygame_action_buttons[PlayerAction.RAISE].enabled = False
        
        # All-in toujours disponible si le joueur a des jetons
        self.pygame_action_buttons[PlayerAction.ALL_IN].enabled = current_player.stack > 0

    def _create_action_buttons(self) -> Dict[PlayerAction, Button]:
        """
        Crée et initialise les boutons d'action pour l'interaction des joueurs.
        
        Returns:
            Dict[PlayerAction, Button]: Dictionnaire associant les actions aux objets boutons
        """
        buttons = {
            PlayerAction.FOLD: Button(300, SCREEN_HEIGHT - 100, 100, 40, "Fold", (200, 0, 0)),
            PlayerAction.CHECK: Button(450, SCREEN_HEIGHT - 100, 100, 40, "Check", (0, 200, 0)),
            PlayerAction.CALL: Button(600, SCREEN_HEIGHT - 100, 100, 40, "Call", (0, 0, 200)),
            PlayerAction.RAISE: Button(750, SCREEN_HEIGHT - 100, 100, 40, "Raise", (200, 200, 0)),
            PlayerAction.ALL_IN: Button(900, SCREEN_HEIGHT - 100, 100, 40, "All-in", (150, 0, 150))
        }
        return buttons

    def process_action(self, player: Player, action: PlayerAction, bet_amount: Optional[int] = None):
        """
        Traite l'action d'un joueur pendant son tour et met à jour l'état du jeu.
        
        Args:
            player (Player): Le joueur qui effectue l'action
            action (PlayerAction): L'action choisie
            bet_amount (Optional[int]): Le montant de la mise si applicable
        """
        # Check if player has sufficient funds for any action
        if player.stack <= 0:
            self._next_player()
            return False
        
        # Don't process actions during showdown
        if self.current_phase == GamePhase.SHOWDOWN:
            return action
            
        # Debug print for action start
        print(f"\n=== Action by {player.name} ===")
        print(f"Player activity: {player.is_active}")
        print(f"Action: {action.value}")
        print(f"Current phase: {self.current_phase}")
        print(f"Current pot: {self.pot}B")
        print(f"Current Maximum bet: {self.current_maximum_bet}B")
        print(f"Player stack before: {player.stack}B")
        print(f"Player current bet: {player.current_bet}B")
        
        # Record the action
        action_text = f"{player.name}: {action.value}"
        if bet_amount is not None and action == PlayerAction.RAISE:
            action_text += f" {bet_amount}B"
        elif action == PlayerAction.RAISE:
            # Calculate minimum and maximum possible raise amounts
            min_raise = max(self.current_maximum_bet * 2, self.big_blind * 2)
            bet_amount = min_raise
            action_text += f" {bet_amount}B"

        # Add round separator before action if phase is changing
        if self.check_phase_completion() and self.current_phase != GamePhase.SHOWDOWN:
            self.pygame_action_history.append(f"--- {self.current_phase.value.upper()} ---")
        
        self.pygame_action_history.append(action_text)
        if len(self.pygame_action_history) > 10:
            self.pygame_action_history.pop(0)
        
        # Process the action
        if action == PlayerAction.FOLD:
            player.has_folded = True
            print(f"{player.name} folds")
            
        elif action == PlayerAction.CHECK:
            print(f"{player.name} checks")

        elif action == PlayerAction.CALL:
            call_amount = self.current_maximum_bet - player.current_bet
            player.stack -= call_amount
            player.current_bet = self.current_maximum_bet
            self.pot += call_amount
            print(f"{player.name} calls {call_amount}B")
            
        elif action == PlayerAction.RAISE and bet_amount is not None:
            total_to_put_in = bet_amount - player.current_bet
            player.stack -= total_to_put_in
            player.current_bet = bet_amount
            self.current_maximum_bet = bet_amount
            self.pot += total_to_put_in
            self.last_raiser_seat = player
            print(f"{player.name} raises to {bet_amount}B")
        
        elif action == PlayerAction.ALL_IN:
            all_in_amount = player.stack + player.current_bet
            total_to_put_in = player.stack
            player.stack = 0
            player.current_bet = all_in_amount
            self.pot += total_to_put_in
            player.is_all_in = True
            
            if all_in_amount > self.current_maximum_bet:
                self.current_maximum_bet = all_in_amount
                self.last_raiser_seat = player
            
            # Créer un side pot si nécessaire
            if any(p.current_bet > all_in_amount for p in self.players if p.is_active and not p.has_folded):
                self._create_side_pot(all_in_amount)
            
            print(f"{player.name} fait tapis avec {all_in_amount}B")
        
        player.has_acted = True
        
        
        # Debug print post-action state
        print(f"Player stack after: {player.stack}B")
        print(f"New pot: {self.pot}B")
        print(f"Active players: {sum(1 for p in self.players if p.is_active)}")
        
        # Check for all-in situations after the action
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        all_in_players = [p for p in active_players if p.stack == 0]
        
        # Check if only one player remains (others folded or inactive)
        if len(active_players) == 1:
            print("Moving to showdown (only one player remains)")
            self.handle_showdown()
            return action
        
        # Check if all remaining active players are all-in
        if (len(all_in_players) == len(active_players)) and (len(active_players) > 1):
            print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return action
        
        # Check if round is complete and handle next phase
        if self.check_phase_completion():
            print("Round complete - advancing phase")
            if self.current_phase == GamePhase.RIVER:
                print("River complete - going to showdown")
                self.handle_showdown()
            else:
                self.advance_phase()
                print(f"Advanced to {self.current_phase}")

        else:
            self._next_player()
            print(f"Next player: {self.players[self.current_player_seat].name}")
        
        return action

    def evaluate_final_hand(self, player: Player) -> Tuple[HandRank, List[int]]:
        """
        Évalue la meilleure main possible d'un joueur avec les cartes communes.
        
        Args:
            player (Player): Le joueur dont on évalue la main
            
        Returns:
            Tuple[HandRank, List[int]]: Le rang de la main et les valeurs pour départager
        """
        # Combine les cartes du joueur avec les cartes communes
        all_cards = player.cards + self.community_cards
        # Extrait les valeurs et couleurs de toutes les cartes
        values = [card.value for card in all_cards]
        suits = [card.suit for card in all_cards]
        
        # Vérifie si une couleur est possible (5+ cartes de même couleur)
        suit_counts = Counter(suits)
        # Trouve la première couleur qui apparaît 5 fois ou plus, sinon None
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 5), None)
        
        # Si une couleur est possible, on vérifie d'abord les mains les plus fortes
        if flush_suit:
            # Trie les cartes de la couleur par valeur décroissante
            flush_cards = sorted([card for card in all_cards if card.suit == flush_suit], key=lambda x: x.value, reverse=True)
            flush_values = [card.value for card in flush_cards]
            
            # Vérifie si on a une quinte flush
            for i in range(len(flush_values) - 4):
                # Vérifie si 5 cartes consécutives de même couleur
                if flush_values[i] - flush_values[i+4] == 4:
                    # Si la plus haute carte est un As, c'est une quinte flush royale
                    if flush_values[i] == 14 and flush_values[i+4] == 10:
                        return (HandRank.ROYAL_FLUSH, [14])
                    # Sinon c'est une quinte flush normale
                    return (HandRank.STRAIGHT_FLUSH, [flush_values[i]])
            
            # Vérifie la quinte flush basse (As-5)
            if set([14,2,3,4,5]).issubset(set(flush_values)):
                return (HandRank.STRAIGHT_FLUSH, [5])
        
        # Compte les occurrences de chaque valeur
        value_counts = Counter(values)
        
        # Vérifie le carré (4 cartes de même valeur)
        if 4 in value_counts.values():
            quads = [v for v, count in value_counts.items() if count == 4][0]
            # Trouve la plus haute carte restante comme kicker
            kicker = max(v for v in values if v != quads)
            return (HandRank.FOUR_OF_A_KIND, [quads, kicker])
        
        # Vérifie le full house (brelan + paire)
        if 3 in value_counts.values():
            # Trouve tous les brelans, triés par valeur décroissante
            trips = sorted([v for v, count in value_counts.items() if count >= 3], reverse=True)
            # Trouve toutes les paires potentielles, y compris les brelans qui peuvent servir de paire
            pairs = []
            for value, count in value_counts.items():
                if count >= 2:  # La carte peut former une paire
                    if count >= 3 and value != trips[0]:  # C'est un second brelan
                        pairs.append(value)
                    elif count == 2:  # C'est une paire simple
                        pairs.append(value)
            
            if pairs:  # Si on a au moins une paire ou un second brelan utilisable comme paire
                return (HandRank.FULL_HOUSE, [trips[0], max(pairs)])
        
        # Vérifie la couleur simple
        if flush_suit:
            flush_cards = sorted([card.value for card in all_cards if card.suit == flush_suit], reverse=True)
            return (HandRank.FLUSH, flush_cards[:5])
        
        # Vérifie la quinte (5 cartes consécutives)
        unique_values = sorted(set(values), reverse=True)
        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i+4] == 4:
                return (HandRank.STRAIGHT, [unique_values[i]])
                
        # Vérifie la quinte basse (As-5)
        if set([14,2,3,4,5]).issubset(set(values)):
            return (HandRank.STRAIGHT, [5])
        
        # Vérifie le brelan
        if 3 in value_counts.values():
            # Trouve tous les brelans et sélectionne le plus haut
            trips = max(v for v, count in value_counts.items() if count >= 3)
            # Garde les 2 meilleures cartes restantes comme kickers
            kickers = sorted([v for v in values if v != trips], reverse=True)[:2]
            return (HandRank.THREE_OF_A_KIND, [trips] + kickers)
        
        # Vérifie la double paire
        pairs = sorted([v for v, count in value_counts.items() if count >= 2], reverse=True)
        if len(pairs) >= 2:
            # Garde la meilleure carte restante comme kicker
            kickers = [v for v in values if v not in pairs[:2]]
            return (HandRank.TWO_PAIR, pairs[:2] + [max(kickers)])
        
        # Vérifie la paire simple
        if pairs:
            # Garde les 3 meilleures cartes restantes comme kickers
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, [pairs[0]] + kickers)
        
        # Si aucune combinaison, retourne la carte haute avec les 5 meilleures cartes
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])

    def handle_showdown(self):
        """
        Gère la phase de showdown en tenant compte des side pots.
        """
        print("\n=== SHOWDOWN ===")
        self.current_phase = GamePhase.SHOWDOWN
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        
        # Désactiver tous les boutons pendant le showdown
        for button in self.pygame_action_buttons.values():
            button.enabled = False
        
        # S'assurer que toutes les cartes communes sont distribuées
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        
        # Traiter d'abord les side pots (du plus petit au plus grand)
        total_winnings = defaultdict(int)
        
        # Traiter les side pots dans l'ordre inverse (du plus grand au plus petit)
        for i in range(len(self.side_pots) - 1, -1, -1):
            if self.side_pots[i] > 0:
                # Identifier les joueurs éligibles pour ce side pot
                eligible_players = [p for p in active_players if not p.is_all_in or p.stack == 0]
                if len(eligible_players) > 1:
                    # Évaluer les mains des joueurs éligibles
                    player_hands = [(player, self.evaluate_final_hand(player)) for player in eligible_players]
                    player_hands.sort(key=lambda x: (x[1][0].value, x[1][1]), reverse=True)
                    
                    # Trouver le(s) gagnant(s) du side pot
                    best_hand = player_hands[0][1]
                    winners = [p for p, h in player_hands if h == best_hand]
                    
                    # Distribuer le side pot équitablement entre les gagnants
                    split_amount = self.side_pots[i] / len(winners)
                    for winner in winners:
                        total_winnings[winner] += split_amount
                elif len(eligible_players) == 1:
                    total_winnings[eligible_players[0]] += self.side_pots[i]
        
        # Traiter le pot principal
        if len(active_players) == 1:
            winner = active_players[0]
            total_winnings[winner] += self.pot
        else:
            player_hands = [(player, self.evaluate_final_hand(player)) for player in active_players]
            player_hands.sort(key=lambda x: (x[1][0].value, x[1][1]), reverse=True)
            best_hand = player_hands[0][1]
            winners = [p for p, h in player_hands if h == best_hand]
            split_amount = self.pot / len(winners)
            for winner in winners:
                total_winnings[winner] += split_amount
                winning_hand = best_hand[0].name.replace('_', ' ').title()
                print(f"{winner.name}'s winning hand: {winning_hand}")
        
        # Distribuer les gains
        for winner, amount in total_winnings.items():
            winner.stack += amount
            if len(total_winnings) == 1:
                self.pygame_winner_info = f"{winner.name} wins {amount}B"
            else:
                self.pygame_winner_info = "Multiple winners: " + ", ".join(f"{p.name} ({amt}B)" for p, amt in total_winnings.items())
        
        # Reset pots
        self.pot = 0
        self.side_pots = [0] * 4
        
        # Set the winner display start time
        self.pygame_winner_display_start = pygame.time.get_ticks()
        self.pygame_winner_display_duration = 2000  # 2 seconds in milliseconds

    def _create_deck(self) -> List[Card]:
        """
        Crée et mélange un nouveau jeu de 52 cartes.
        
        Returns:
            List[Card]: Un jeu de cartes mélangé
        """
        suits = ['♠', '♥', '♦', '♣']
        values = range(2, 15)  # 2-14 (Ace is 14)
        deck = [Card(suit, value) for suit in suits for value in values]
        rd.shuffle(deck)
        return deck
    
    def _initialize_players(self, list_names: List[str]) -> List[Player]:
        """
        Crée et initialise tous les joueurs pour la partie.
        
        Returns:
            List[Player]: Liste des objets joueurs initialisés
        """
        players = []
        for idx, name in enumerate(list_names):
            player = Player(name, self.starting_stack, idx)
            players.append(player)
        return players
    
    def deal_cards(self):
        """
        Distribue deux cartes à chaque joueur actif.
        Réinitialise et mélange le jeu avant la distribution.
        """
        # Deal two cards to each active player
        for _ in range(2):
            for player in self.players:
                if player.is_active:
                    player.cards.append(self.deck.pop())
    
    def deal_community_cards(self):
        """
        Distribue les cartes communes selon la phase de jeu actuelle.
        Distribue 3 cartes pour le flop, 1 pour le turn et 1 pour la river.
        """
        if self.current_phase == GamePhase.FLOP:
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
        elif self.current_phase in [GamePhase.TURN, GamePhase.RIVER]:
            self.community_cards.append(self.deck.pop())

    def _create_side_pot(self, all_in_amount: int):
        """
        Crée un side pot lorsqu'un joueur est all-in.
        
        Args:
            all_in_amount (int): Montant du all-in du joueur
        """
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        
        # Calculer les contributions au side pot
        side_pot_amount = 0
        main_pot_amount = 0
        
        # Première passe : calculer les montants des pots
        for player in active_players:
            if player.current_bet > all_in_amount:
                # L'excédent va dans le side pot
                excess = player.current_bet - all_in_amount
                side_pot_amount += excess
                # La partie égale va dans le pot principal
                main_pot_amount += all_in_amount
            else:
                # Tout va dans le pot principal
                main_pot_amount += player.current_bet
        
        # Deuxième passe : ajuster les mises des joueurs
        for player in active_players:
            if player.current_bet > all_in_amount:
                player.current_bet = all_in_amount
        
        # Mettre à jour les pots
        self.pot = main_pot_amount
        
        # Trouver le premier side pot vide
        for i, pot in enumerate(self.side_pots):
            if pot == 0:
                self.side_pots[i] = side_pot_amount
                break

    # --------------------------------
    # Methodes Nécessaires pour le RL
    # --------------------------------
    def get_state(self):
        """
        Obtient l'état actuel du jeu pour l'apprentissage par renforcement.
        
        Returns:
            List[float]: État normalisé du jeu incluant:
            - Cartes (joueur et communes)
            - Rang de la main
            - Mises et positions
            - État des joueurs
            - Phase et actions disponibles
        """
        current_player = self.players[self.current_player_seat]
        state = []

        # Correspondance des couleurs avec des nombres ♠, ♥, ♦, ♣
        suit_map = {
            "♠" : 0,
            "♥" : 1,
            "♦" : 2,
            "♣" : 3
        }

        # 1. Informations sur les cartes (encodage one-hot)
        # Cartes du joueur
        for card in current_player.cards:
            value_range = [0.01] * 13
            value_range[card.value - 2] = 1
            state.extend(value_range)  # Extension pour la valeur
            suit_range = [0.01] * 4
            suit_range[suit_map[card.suit]] = 1
            state.extend(suit_range)  # Extension pour la couleur
        
        # Ajout de remplissage pour les cartes manquantes du joueur
        remaining_player_cards = 2 - len(current_player.cards)
        for _ in range(remaining_player_cards):
            state.extend([0.01] * 13)  # Remplissage des valeurs
            state.extend([0.01] * 4)   # Remplissage des couleurs
        
        # Cartes communes
        for i, card in enumerate(self.community_cards):
            value_range = [0.01] * 13
            value_range[card.value - 2] = 1
            state.extend(value_range)  # Extension
            suit_range = [0.01] * 4
            suit_range[suit_map[card.suit]] = 1
            state.extend(suit_range)  # Extension
        
        # Ajout de remplissage pour les cartes communes manquantes
        remaining_community_cards = 5 - len(self.community_cards)
        for _ in range(remaining_community_cards):
            state.extend([0.01] * 13)  # Remplissage des valeurs
            state.extend([0.01] * 4)   # Remplissage des couleurs
        
        # 2. Rang de la main actuelle (si assez de cartes sont visibles)
        if len(current_player.cards) + len(self.community_cards) >= 5:
            hand_rank, _ = self.evaluate_final_hand(current_player)
            state.append(hand_rank.value / len(HandRank))  # Normalisation de la valeur du rang (taille = 1)
        else:
            hand_rank, _ = self.evaluate_current_hand(current_player)
            state.append(hand_rank.value / len(HandRank))  # Normalisation de la valeur du rang (taille = 1)

        # 3. Informations sur le tour
        phase_values = {
            GamePhase.PREFLOP: 0,
            GamePhase.FLOP: 1,
            GamePhase.TURN: 2,
            GamePhase.RIVER: 3,
            GamePhase.SHOWDOWN: 4
        }

        phase_range = [0.01] * 5
        phase_range[phase_values[self.current_phase]] = 1
        state.extend(phase_range)

        # 4. Mise actuelle normalisée par la grosse blinde
        state.append(self.current_maximum_bet / self.big_blind)  # Normalisation de la mise (taille = 1)

        # 6. Argent restant (tailles des stacks normalisées par le stack initial)
        initial_stack = self.starting_stack
        for player in self.players:
            state.append(player.stack / initial_stack) # (taille = 3)

        # 7. Informations sur les mises (normalisées par la grosse blinde)
        for player in self.players:
            state.append(player.current_bet / initial_stack) # (taille = 3)

        # 8. Informations sur l'activité (binaire extrême : actif/ruiné)
        for player in self.players:
            state.append(1 if player.is_active else -1) # (taille = 3)
        
        # 9. Informations sur l'activité in game (binaire extrême : en jeu/a foldé)
        for player in self.players:
            state.append(1 if player.has_folded else -1) # (taille = 3)

        # 10. Informations sur la position (encodage one-hot des positions relatives)
        relative_positions = [0.1] * self.num_players
        relative_pos = (self.current_player_seat - self.button_seat_position) % self.num_players
        relative_positions[relative_pos] = 1
        state.extend(relative_positions) # (taille = 3)

        # 11. Actions disponibles (binaire extrême : disponible/indisponible)
        action_availability = []
        for action in PlayerAction:
            if action in self.pygame_action_buttons and self.pygame_action_buttons[action].enabled:
                action_availability.append(1)
            else:
                action_availability.append(-1)
        state.extend(action_availability) # (taille = 3)

        # 12. Actions précédentes (dernière action de chaque joueur, encodée en vecteurs one-hot)
        action_encoding = {
            None: 0,
            PlayerAction.FOLD: 1,
            PlayerAction.CHECK: 2,
            PlayerAction.CALL: 3,
            PlayerAction.RAISE: 4,
            PlayerAction.ALL_IN: 5
        }

        # Initialisation du tableau des dernières actions avec des zéros
        last_actions = [[0.1] * 6 for _ in range(self.num_players)]  # 6 actions possibles (y compris None)

        # Traitement des actions récentes
        for action_text in reversed(self.pygame_action_history[-self.num_players:]):
            if ":" in action_text:
                player_name, action = action_text.split(":")
                player_idx = int(player_name.split("_")[-1]) - 1
                action = action.strip()
                
                # Recherche du type d'action correspondant
                for action_type in PlayerAction:
                    if action_type.value in action:
                        # Création de l'encodage one-hot
                        last_actions[player_idx] = [0.1] * 6  # Réinitialisation à zéro
                        last_actions[player_idx][action_encoding[action_type]] = 1
                        break

        # 12. Liste des actions précédentes des 3 joueurs
        flattened_actions = [val for sublist in last_actions for val in sublist]
        state.extend(flattened_actions)  

        # 13. Estimation de la probabilité de victoire
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        num_opponents = len(active_players) - 1
        
        if num_opponents <= 0:
            win_prob = 1.0  # Plus d'adversaires
        else:
            # Obtention de la force de la main
            hand_strength = self._evaluate_hand_strength(current_player)
            
            # Ajustement pour le nombre d'adversaires (2-3)
            win_prob = hand_strength ** num_opponents
            
            # Ajustement spécifique au pré-flop (utilisant l'approximation Sklansky-Chubukov)
            if self.current_phase == GamePhase.PREFLOP:
                # Réduction de la confiance dans les estimations pré-flop
                win_prob *= 0.8
        
        state.append(win_prob) # (taille = 1)

        # 14. Cotes du pot
        call_amount = self.current_maximum_bet - current_player.current_bet
        pot_odds = call_amount / (self.pot + call_amount) if (self.pot + call_amount) > 0 else 0
        state.append(pot_odds) # (taille = 1)

        # 15. Équité
        equity = self._evaluate_equity(current_player)
        state.append(equity) # (taille = 1)

        # 16. Facteur d'agressivité
        state.append(self.number_raise_this_game_phase / 4) # (taille = 1)

        # Avant de retourner, conversion en tableau numpy
        state = np.array(state, dtype=np.float32)
        return state

    def step(self, action: PlayerAction) -> Tuple[List[float], float]:
        """
        Exécute une action et calcule la récompense associée.
        
        Args:
            action (PlayerAction): L'action à exécuter
            
        Returns:
            Tuple[List[float], float]: Nouvel état et récompense
        """
        current_player = self.players[self.current_player_seat]
        reward = 0.0

        # Capturer l'état du jeu avant de traiter l'action pour le calcul des cotes du pot
        call_amount_before = self.current_maximum_bet - current_player.current_bet
        pot_before = self.pot

        # --- Récompenses stratégiques des actions ---
        # Récompense basée sur l'action par rapport à la force de la main
        hand_strength = self._evaluate_hand_strength(current_player)
        pot_potential = self.pot / (self.big_blind * 100)
    
        if action == PlayerAction.RAISE:
            reward += 0.2 * hand_strength  # Ajuster la récompense selon la force de la main
            if hand_strength > 0.7:
                reward += 0.5 * pot_potential  # Bonus pour jeu agressif avec main forte
                
        elif action == PlayerAction.ALL_IN:
            reward += 0.3 * hand_strength
            if hand_strength > 0.8:
                reward += 1.0 * pot_potential
                
        elif action == PlayerAction.CALL:
            reward += 0.1 * min(hand_strength, 0.6)  # Rendements décroissants pour jeu passif
        
        elif action == PlayerAction.CHECK: # Pénaliser le check si la main est forte
            if hand_strength > 0.5:
                reward -= 0.5 * pot_potential
            else:
                reward += 0.3

        elif action == PlayerAction.FOLD:
            if hand_strength < 0.2:
                reward += 0.3  # Récompenser les bons folds
            else:
                reward -= 0.5  # Pénaliser les folds avec bonnes mains
            
        # --- Bonus de position ---
        # Bonus pour actions agressives en dernière position (bouton)
        if current_player.seat_position == self.button_seat_position:
            if action in [PlayerAction.RAISE, PlayerAction.ALL_IN]:
                reward += 0.2
        
        # Traiter l'action (met à jour l'état du jeu)
        self.process_action(current_player, action)

        # --- Évaluation des cotes du pot ---
        if action == PlayerAction.CALL and call_amount_before > 0:
            total_pot_after_call = pot_before + call_amount_before
            pot_odds = call_amount_before / total_pot_after_call if total_pot_after_call > 0 else 0
            if hand_strength > pot_odds:
                reward += 0.3  # Call mathématiquement justifié
            else:
                reward -= 0.3  # Mauvais call considérant les cotes

        return self.get_state(), reward

    # --------------------------------
    # Methodes de calculs (à exporter dans un autre fichier)
    # --------------------------------
    def evaluate_current_hand(self, player) -> Tuple[HandRank, List[int]]:
        """
        Évalue la main actuelle d'un joueur avec les cartes communes disponibles meme si il y en a moins de 5.
        
        Args:
            player (Player): Le joueur dont on évalue la main
        """
        # Si le joueur n'a pas de cartes ou a foldé
        if not player.cards or player.has_folded:
            return (HandRank.HIGH_CARD, [0])
        
        # Obtenir toutes les cartes disponibles
        all_cards = player.cards + self.community_cards
        values = [card.value for card in all_cards]
        suits = [card.suit for card in all_cards]
        
        # Au pré-flop, évaluer uniquement les cartes du joueur
        if self.current_phase == GamePhase.PREFLOP:
            # Paire de départ
            if player.cards[0].value == player.cards[1].value:
                return (HandRank.PAIR, [player.cards[0].value])
            # Cartes hautes
            return (HandRank.HIGH_CARD, sorted([c.value for c in player.cards], reverse=True))
        
        # Compter les occurrences des valeurs et couleurs
        value_counts = Counter(values)
        suit_counts = Counter(suits)
        
        # Vérifier les combinaisons possibles avec les cartes disponibles
        # Paire
        pairs = [v for v, count in value_counts.items() if count >= 2]
        if pairs:
            if len(pairs) >= 2:  # Double paire
                pairs.sort(reverse=True)
                kicker = max(v for v in values if v not in pairs[:2])
                return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
            # Simple paire
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, pairs + kickers)
        
        # Brelan
        trips = [v for v, count in value_counts.items() if count >= 3]
        if trips:
            kickers = sorted([v for v in values if v != trips[0]], reverse=True)[:2]
            return (HandRank.THREE_OF_A_KIND, [trips[0]] + kickers)
        
        # Couleur potentielle (4 cartes de la même couleur)
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 4), None)
        if flush_suit:
            flush_cards = sorted([card.value for card in all_cards if card.suit == flush_suit], reverse=True)
            if len(flush_cards) >= 5:
                return (HandRank.FLUSH, flush_cards[:5])
        
        # Quinte potentielle
        unique_values = sorted(set(values))
        for i in range(len(unique_values) - 3):
            if unique_values[i+3] - unique_values[i] == 3:  # 4 cartes consécutives
                return (HandRank.STRAIGHT, [unique_values[i+3]])
        
        # Si aucune combinaison, retourner la plus haute carte
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])
    
    def _evaluate_preflop_strength(self, cards) -> float:
        """
        Évalue la force d'une main preflop selon des heuristiques.
        
        Args:
            cards (List[Card]): Les deux cartes à évaluer
            
        Returns:
            float: Force de la main entre 0 et 1
        """
        # Vérification de sécurité pour mains vides ou incomplètes
        if not cards or len(cards) < 2:
            return 0.0
        
        card1, card2 = cards
        # Paires
        if card1.value == card2.value:
            return 0.5 + (card1.value / 28)  # Plus haute est la paire, plus fort est le score
        
        # Cartes assorties
        suited = card1.suit == card2.suit
        # Connecteurs
        connected = abs(card1.value - card2.value) == 1
        
        # Score de base basé sur les valeurs des cartes
        base_score = (card1.value + card2.value) / 28  # Normaliser par le max possible
        
        # Bonus pour suited et connected
        if suited:
            base_score += 0.1
        if connected:
            base_score += 0.05
        
        return min(base_score, 1.0)  # Garantir que le score est entre 0 et 1

    def _evaluate_hand_strength(self, player) -> float:
        """
        Évalue la force relative d'une main (0 à 1) similaire à _evaluate_preflop_strength 
        mais avec les cartes communes
        
        Args:
            player (Player): Le joueur dont on évalue la main
            
        Returns:
            float: Force de la main entre 0 (très faible) et 1 (très forte)
        """
        # Retourner 0 si le joueur a foldé ou n'a pas de cartes
        if not player.is_active or not player.cards:
            return 0.0
        
        # Au pré-flop, utiliser l'évaluation spécifique pré-flop
        if self.current_phase == GamePhase.PREFLOP:
            return self._evaluate_preflop_strength(player.cards)
        
        # Obtenir toutes les cartes disponibles (main + cartes communes)
        all_cards = player.cards + self.community_cards
        
        # Évaluer la main actuelle
        hand_rank, kickers = self.evaluate_final_hand(player)
        base_score = hand_rank.value / len(HandRank)  # Score de base normalisé
        
        # Bonus/malus selon la phase de jeu et les kickers
        phase_multiplier = {
            GamePhase.FLOP: 0.8,   # Moins certain au flop
            GamePhase.TURN: 0.9,   # Plus certain au turn
            GamePhase.RIVER: 1.0,  # Certitude maximale à la river
        }.get(self.current_phase, 1.0)
        
        # Calculer le score des kickers (normalisé)
        kicker_score = sum(k / 14 for k in kickers) / len(kickers) if kickers else 0
        
        # Vérifier les tirages possibles
        draw_potential = 0.0
        
        # Compter les cartes de chaque couleur
        suits = [card.suit for card in all_cards]
        suit_counts = Counter(suits)
        
        # Compter les cartes consécutives
        values = sorted(set(card.value for card in all_cards))
        
        # Tirage couleur
        flush_draw = any(count == 4 for count in suit_counts.values())
        if flush_draw:
            draw_potential += 0.15
        
        # Tirage quinte
        for i in range(len(values) - 3):
            if values[i+3] - values[i] == 3:  # 4 cartes consécutives
                draw_potential += 0.15
                break
        
        # Tirage quinte flush
        if flush_draw and any(values[i+3] - values[i] == 3 for i in range(len(values) - 3)):
            draw_potential += 0.1
        
        # Le potentiel de tirage diminue à mesure qu'on avance dans les phases
        draw_potential *= {
            GamePhase.FLOP: 1.0,
            GamePhase.TURN: 0.5,
            GamePhase.RIVER: 0.0,
        }.get(self.current_phase, 0.0)
        
        # Calculer le score final
        final_score = (
            base_score * 0.7 +      # Score de base (70% du score)
            kicker_score * 0.2 +    # Score des kickers (20% du score)
            draw_potential          # Potentiel de tirage (jusqu'à 10% supplémentaires)
        ) * phase_multiplier
        
        return min(1.0, max(0.0, final_score))  # Garantir un score entre 0 et 1
    
    def _evaluate_equity(self, player) -> float:
        """
        Calcule l'équité au pot pour la main d'un joueur.
        Prend en compte la position, les cotes et la phase de jeu.
        
        Args:
            player (Player): Le joueur dont on évalue l'équité
            
        Returns:
            float: Équité entre 0 et 1
        """
        # Return 0 equity if player has folded or has no cards
        if not player.is_active or not player.cards:
            return 0.0
        
        # Get base equity from hand strength
        hand_strength = self._evaluate_hand_strength(player)
        
        # Count active players
        active_players = [p for p in self.players if p.is_active]
        num_active = len(active_players)
        if num_active <= 1:
            return 1.0  # Only player left
        
        # Position multiplier (better position = higher equity)
        # Calculate relative position from button (0 = button, 1 = SB, 2 = BB)
        relative_pos = (player.seat_position - self.button_seat_position) % self.num_players
        position_multiplier = 1.0 + (0.1 * (self.num_players - relative_pos) / self.num_players)
        
        # Pot odds consideration
        total_pot = self.pot + sum(p.current_bet for p in self.players)
        call_amount = self.current_maximum_bet - player.current_bet
        if call_amount > 0 and total_pot > 0:
            pot_odds = call_amount / (total_pot + call_amount)
            # Adjust equity based on pot odds
            if hand_strength > pot_odds:
                equity_multiplier = 1.2  # Good pot odds
            else:
                equity_multiplier = 0.8  # Poor pot odds
        else:
            equity_multiplier = 1.0
        
        # Phase multiplier (later streets = more accurate equity)
        phase_multipliers = {
            GamePhase.PREFLOP: 0.7,  # Less certain
            GamePhase.FLOP: 0.8,
            GamePhase.TURN: 0.9,
            GamePhase.RIVER: 1.0     # Most certain
        }
        phase_multiplier = phase_multipliers.get(self.current_phase, 1.0)
        
        # Calculate final equity
        equity = (
            hand_strength 
            * position_multiplier 
            * equity_multiplier 
            * phase_multiplier
        )
        
        # Clip to [0, 1]
        return np.clip(equity, 0.0, 1.0)
    
    # --------------------------------
    # Interface graphique et affichage
    # --------------------------------
    def _draw_card(self, card: Card, x: int, y: int):
        """
        Dessine une carte sur l'écran avec sa valeur et sa couleur.
        
        Args:
            card (Card): La carte à dessiner
            x (int): Position X sur l'écran
            y (int): Position Y sur l'écran
        """
        # Draw card background
        card_width, card_height = 50, 70
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, card_width, card_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, card_width, card_height), 1)
        
        # Draw card text
        color = (255, 0, 0) if card.suit in ['♥', '♦'] else (0, 0, 0)
        text = self.font.render(str(card), True, color)
        self.screen.blit(text, (x + 5, y + 5))
    
    def _draw_player(self, player: Player):
        """
        Dessine les informations d'un joueur sur l'écran (cartes, stack, mises).
        
        Args:
            player (Player): Le joueur à dessiner
        """
        # Draw neon effect for active player
        if player.seat_position == self.current_player_seat:
            # Draw multiple circles with decreasing alpha for glow effect
            for radius in range(40, 20, -5):
                glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                alpha = int(255 * (1 - (radius - 20) / 20))  # Fade from center
                pygame.draw.circle(glow_surface, (0, 255, 255, alpha), (radius, radius), radius)  # Cyan glow
                self.screen.blit(glow_surface, (player.x + 25 - radius, player.y + 35 - radius))
        
        # Draw player info with 2 decimal places
        name_color = (0, 255, 255) if player.seat_position == self.current_player_seat else (255, 255, 255)
        name_text = self.font.render(f"{player.name} ({player.stack:.2f}B)", True, name_color)
        self.screen.blit(name_text, (player.x - 50, player.y - 40))
        
        # Draw player cards
        if player.is_active and not player.has_folded and len(player.cards) > 0:
            if player.is_human or self.current_phase == GamePhase.SHOWDOWN:
                for i, card in enumerate(player.cards):
                    self._draw_card(card, player.x + i * 60, player.y)
            else:
                # Draw card backs for non-human players
                for i in range(2):
                    pygame.draw.rect(self.screen, (200, 0, 0), (player.x + i * 60, player.y, 50, 70))
        elif player.is_active and player.has_folded :
            for i in range(2): # Draw 2 back cards a bit transparent for folded players
                pygame.draw.rect(self.screen, (100, 100, 100), (player.x + i * 60, player.y, 50, 70))


        # Draw current bet with 2 decimal places
        if player.current_bet > 0:
            bet_text = self.font.render(f"Bet: {player.current_bet:.2f}B", True, (255, 255, 0))
            self.screen.blit(bet_text, (player.x - 30, player.y + 80))
    
        # Draw dealer button (D) - Updated positioning logic
        if player.seat_position == self.button_seat_position:  # Only draw if this player is the dealer
            button_x = player.x + 52
            button_y = player.y + 80
            pygame.draw.circle(self.screen, (255, 255, 255), (button_x, button_y), 15)
            dealer_text = self.font.render("D", True, (0, 0, 0))
            dealer_rect = dealer_text.get_rect(center=(button_x, button_y))
            self.screen.blit(dealer_text, dealer_rect)
    
    def _draw(self):
        """
        Dessine l'état complet du jeu sur l'écran.
        """
        # Clear screen
        self.screen.fill((0, 100, 0))  # Green felt background
        
        # Draw table as a racetrack shape with smooth edges
        outer_rect = pygame.Rect(100, 150, SCREEN_WIDTH-200, SCREEN_HEIGHT-360)
        inner_rect = pygame.Rect(120, 170, SCREEN_WIDTH-240, SCREEN_HEIGHT-400)
        pygame.draw.rect(self.screen, (139, 69, 19), outer_rect, border_radius=400)
        pygame.draw.rect(self.screen, (165, 42, 42), inner_rect, border_radius=360)
        
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            self._draw_card(card, 400 + i * 60, 350)
        
        # Draw pot with 2 decimal places
        pot_text = self.font.render(f"Pot: {self.pot:.2f}B", True, (255, 255, 255))
        self.screen.blit(pot_text, (550, 300))
        
        # Draw players
        for player in self.players:
            self._draw_player(player)
        
        # Draw current player indicator in bottom right
        current_player = self.players[self.current_player_seat]
        current_player_text = self.font.render(f"Current Player: {current_player.name}", True, (255, 255, 255))
        self.screen.blit(current_player_text, (SCREEN_WIDTH - 300, SCREEN_HEIGHT - 50))
        
        # Update button states before drawing
        self._update_button_states()
        
        # Draw action buttons for current player's turn
        for button in self.pygame_action_buttons.values():
            button.draw(self.screen, self.font)
        
        # Draw bet slider with min and max values
        current_player = self.players[self.current_player_seat]
        min_raise = max(self.current_maximum_bet * 2, self.big_blind * 2)
        max_raise = current_player.stack + current_player.current_bet
        
        pygame.draw.rect(self.screen, (200, 200, 200), self.pygame_bet_slider)
        bet_text = self.font.render(f"Bet: {int(self.pygame_slider_bet_amount)}B", True, (255, 255, 255))
        min_text = self.font.render(f"Min: {min_raise}B", True, (255, 255, 255))
        max_text = self.font.render(f"Max: {max_raise}B", True, (255, 255, 255))
        
        self.screen.blit(bet_text, (50, SCREEN_HEIGHT - 75))
        self.screen.blit(min_text, (self.pygame_bet_slider.x, SCREEN_HEIGHT - 125))
        self.screen.blit(max_text, (self.pygame_bet_slider.x, SCREEN_HEIGHT - 150))
        
        # Draw action history in top right corner with better formatting
        history_x = SCREEN_WIDTH - 300
        history_y = 50
        history_text = self.font.render("Action History:", True, (255, 255, 255))
        self.screen.blit(history_text, (history_x, history_y - 30))
        
        for i, action in enumerate(self.pygame_action_history):
            # Use different colors for different types of text
            if action.startswith("==="):  # New hand separator
                color = (255, 215, 0)  # Gold
            elif action.startswith("---"):  # Round separator
                color = (0, 255, 255)  # Cyan
            else:  # Normal action
                color = (255, 255, 255)  # White
            
            text = self.font.render(action, True, color)
            self.screen.blit(text, (history_x, history_y + i * 25))
        
        # Draw game info
        game_info_text = self.font.render(f"Game Info: {self.current_phase}", True, (255, 255, 255))
        self.screen.blit(game_info_text, (50, 50))
        
        # Draw winner announcement if there is one and within display duration
        if self.pygame_winner_info:
            current_time = pygame.time.get_ticks()
            if current_time - self.pygame_winner_display_start < self.pygame_winner_display_duration:
                # Create semi-transparent overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                overlay.fill((0, 0, 0))
                overlay.set_alpha(128)
                self.screen.blit(overlay, (0, 0))
                
                # Draw winner text with shadow for better visibility
                winner_font = pygame.font.SysFont('Arial', 48, bold=True)
                shadow_text = winner_font.render(self.pygame_winner_info, True, (0, 0, 0))  # Shadow
                winner_text = winner_font.render(self.pygame_winner_info, True, (255, 215, 0))  # Gold color
                
                # Position for center of screen
                text_rect = winner_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                
                # Draw shadow slightly offset
                shadow_rect = text_rect.copy()
                shadow_rect.x += 2
                shadow_rect.y += 2
                self.screen.blit(shadow_text, shadow_rect)
                
                # Draw main text
                self.screen.blit(winner_text, text_rect)
            else:
                # After display duration, start new hand
                self.pygame_winner_info = None
                self.button_seat_position = (self.button_seat_position + 1) % self.num_players
                active_players = [p for p in self.players if p.stack >= self.big_blind]
                if len(active_players) > 1:
                    self.start_new_hand()
                else:
                    self.reset()

        # Ajouter l'affichage des blindes actuelles
        blind_text = self.font.render(f"Blindes: {self.small_blind}/{self.big_blind}", True, (255, 255, 255))
        self.screen.blit(blind_text, (50, 25))

    def handle_input(self, event):
        """
        Gère les événements d'entrée des joueurs (souris, clavier).
        
        Args:
            event: Objet événement Pygame à traiter
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            current_player = self.players[self.current_player_seat]
            
            # Check button clicks
            for action, button in self.pygame_action_buttons.items():
                if button.rect.collidepoint(mouse_pos) and button.enabled:
                    bet_amount = self.pygame_slider_bet_amount if action == PlayerAction.RAISE else None
                    # Validate bet amount doesn't exceed player's stack
                    if action == PlayerAction.RAISE:
                        max_bet = current_player.stack + current_player.current_bet
                        min_bet = max(self.current_maximum_bet * 2, self.big_blind * 2)
                        bet_amount = min(bet_amount, max_bet)
                        bet_amount = max(bet_amount, min_bet)
                    self.process_action(current_player, action, bet_amount)
            
            # Check bet slider
            if self.pygame_bet_slider.collidepoint(mouse_pos):
                # Calculate minimum raise (2x current bet)
                min_raise = max(self.current_maximum_bet * 2, self.big_blind * 2)
                # Calculate maximum raise (player's stack + current bet)
                max_raise = current_player.stack + current_player.current_bet
                
                # Calculate bet amount based on slider position
                slider_value = (mouse_pos[0] - self.pygame_bet_slider.x) / self.pygame_bet_slider.width
                bet_range = max_raise - min_raise
                self.pygame_slider_bet_amount = min(min_raise + (bet_range * slider_value), max_raise)
                self.pygame_slider_bet_amount = max(self.pygame_slider_bet_amount, min_raise)

    def manual_run(self):
        """
        Lance le jeu en mode manuel avec interface graphique.
        Gère la boucle principale du jeu et les événements.
        """        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.start_new_hand()
                    if event.key == pygame.K_r:
                        self.reset()
                    if event.key == pygame.K_s:
                        state = self.get_state()
                        print('--------------------------------')
                        
                        # Print player cards (2 cards, each with value and suit)
                        print("Player cards:")
                        for i in range(2):  # 2 cards
                            value_range = state[i*17:(i*17)+13]  # 13 possible values
                            suit_range = state[i*17+13:(i*17)+17]  # 4 possible suits
                            value = value_range.index(1) + 2  # Convert back to card value
                            suit = ["♠", "♥", "♦", "♣"][suit_range.index(1)]  # Convert back to suit
                            print(f"Card {i+1}: {value}{suit}")
                        
                        # Print community cards (up to 5 cards)
                        print("\nCommunity cards:")
                        for i in range(5):  # Up to 5 community cards
                            base_idx = 34 + (i*17)  # Starting index for each community card
                            value_range = state[base_idx:base_idx+13]
                            suit_range = state[base_idx+13:base_idx+17]
                            if 1 in value_range:  # Check if card exists
                                value = value_range.index(1) + 2
                                suit = ["♠", "♥", "♦", "♣"][suit_range.index(1)]
                                print(f"Card {i+1}: {value}{suit}")
                        
                        # Print rest of state information
                        print(f"\nHand rank: {state[119] * len(HandRank)}")  # Index after card encodings
                        print(f"Game phase: {state[120:125]}")  # 5 values for game phase
                        print(f"Round number: {state[125]}")
                        print(f"Current bet: {state[126]}")
                        print(f"Stack sizes: {[x * self.starting_stack for x in state[127:130]]}")
                        print(f"Current bets: {state[130:133]}")
                        print(f"Player activity: {state[133:136]}")
                        print(f"Relative positions: {state[136:139]}")
                        print(f"Available actions: {state[139:144]}")
                        print(f"Previous actions Player 1: {state[144:150]}")
                        print(f"Previous actions Player 2: {state[150:156]}")
                        print(f"Previous actions Player 3: {state[156:162]}")
                        print(f"Win probability: {state[162]}")
                        print(f"Pot odds: {state[163]}")
                        print(f"Equity: {state[164]}")
                        print(f"Aggression factor: {state[165]}")
                        print('--------------------------------')
                
                self.handle_input(event)
                
                # Update button hover states
                mouse_pos = pygame.mouse.get_pos()
                for button in self.pygame_action_buttons.values():
                    button.is_hovered = button.rect.collidepoint(mouse_pos)
            
            self._draw()
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    list_names = ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5", "Player_6"]
    game = PokerGame(list_names)
    game.manual_run()
