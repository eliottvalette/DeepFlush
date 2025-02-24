# poker_game.py
"""
Texas Hold'em, No Limit, 6 max.
"""
import torch
import pygame
import random as rd
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import pygame.font
from collections import Counter
import numpy as np
from dataclasses import dataclass

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
POSITIONS = [
    (SCREEN_WIDTH-550, SCREEN_HEIGHT-250),         # Bas-Droite (Joueur 1)
    (420             , SCREEN_HEIGHT-250),         # Bas-Gauche (Joueur 2) 
    (80              , (SCREEN_HEIGHT-150) / 2),   # Milieu-Gauche (Joueur 3)
    (420             , 140),                       # Haut-Gauche (Joueur 4)
    (SCREEN_WIDTH-550, 140),                       # Haut-Droite (Joueur 5)
    (SCREEN_WIDTH-190, (SCREEN_HEIGHT-150)/2)      # Milieu-Droite (Joueur 6)
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
    # -----------------------------------------------------   
    RAISE = "raise" # Raise minimum (2x la mise précédente)        
    RAISE_25_POT = "raise-25%"     # Raise de 25% du pot
    RAISE_33_POT = "raise-33%"     # Raise de 33% du pot
    RAISE_50_POT = "raise-50%"     # Raise de 50% du pot
    RAISE_66_POT = "raise-66%"     # Raise de 66% du pot
    RAISE_75_POT = "raise-75%"     # Raise de 75% du pot
    RAISE_100_POT = "raise-100%"   # Raise égal au pot
    RAISE_125_POT = "raise-125%"   # Raise de 125% du pot
    RAISE_150_POT = "raise-150%"   # Raise de 150% du pot
    RAISE_175_POT = "raise-175%"   # Raise de 175% du pot
    RAISE_2X_POT = "raise-200%"    # Raise de 2x le pot             
    RAISE_3X_POT = "raise-300%"    # Raise de 3x le pot            
    # -----------------------------------------------------
    ALL_IN = "all-in"                                                

    # 16 actions possibles

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
    def __init__(self, agent = None, name: str = "Player", stack: int = 100, seat_position: int = 0):
        """
        Initialise un joueur avec son agent associé, son stack et sa position.
        
        Args:
            agent (PokerAgent): L'agent qui contrôle ce joueur
            stack (int): Stack de départ en jetons
            position (int): Position à la table (0-5)
        """
        if agent:
            self.agent = agent  # Stockage direct de l'objet agent
            self.name = agent.name  # On garde le nom pour l'affichage
        else:
            self.agent = None
            self.name = name
        self.stack = stack
        self.seat_position = seat_position # 0-5
        self.role_position = None # 0-5 (0 = SB, 1 = BB, 2 = UTG, 3 = HJ, 4 = CO, 5 = BTN)
        self.cards: List[Card] = []
        self.is_active = True # True si le joueur a assez de fonds pour jouer (stack > big_blind)
        self.has_folded = False
        self.show_cards = True # True si on veut voir les cartes du joueur
        self.is_all_in = False
        self.range = None # Range du joueur (à initialiser comme l'ensemble des mains possibles)
        self.current_player_bet = 0 # Montant de la mise actuelle du joueur
        self.total_bet = 0  # Cumul des mises effectuées dans la main
        self.x, self.y = POSITIONS[self.seat_position]
        self.has_acted = False # True si le joueur a fait une action dans la phase courante (nécessaire pour savoir si le tour est terminé, car si le premier joueur de la phase check, tous les jouers sont a bet égal et ca déclencherait la phase suivante)

class SidePot:
    """
    Représente un pot additionnel qui peut être créé lors d'un all-in.

    Pour la répartition en side pots, on laisse les plus pauvres all-in dans le main pot et on attend la fin de la phase.
    Si les joeurs les plus pauvres son all-in et que les plus riches sont soit all-in aussi soit à un bet égal, la phase est terminée et on réparti les surplus en side pots.
    """
    def __init__(self, id: int):
        self.id = id # 0-4 (5 Pots max pour 6 joueurs, un main pot et 4 side pots)
        self.players = []
        self.contributions_dict = {} # Dictionnaire des contributions de chaque joueur dans le side pot
        self.sum_of_contributions = 0 # Montant total dans le side pot

class PokerGame:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
    class ActionRecord:
        """
        Stocke les informations d'une action prise par un joueur.
        Fait partie intégrante du jeu de poker.
        """
        def __init__(self, 
                     phase: GamePhase, 
                     player: str, 
                     position: int, 
                     stack_of_player_before_the_action: int, 
                     pot_before_the_action: int, 
                     action_taken: PlayerAction, 
                     stack_of_player_after_the_action: int, 
                     pot_after_the_action: int, 
                     bet_amount: float = -1):
            
            self.phase = phase
            self.player = player  # nom du joueur
            self.position = position  # position du joueur (0-5)
            self.stack_of_player_before_the_action = stack_of_player_before_the_action
            self.pot_before_the_action = pot_before_the_action
            self.action_taken = action_taken
            self.stack_of_player_after_the_action = stack_of_player_after_the_action
            self.pot_after_the_action = pot_after_the_action
            self.bet_amount = bet_amount  # -1 si fold, sinon le montant misé

    def __init__(self, agents):
        """
        Initialise la partie de poker avec les joueurs et les blindes.
        
        Args:
            num_players (int): Nombre de joueurs (défaut: 6)
        """
        self.num_players = 6
                
        self.small_blind = 0.5
        self.big_blind = 1
        self.starting_stack = 100 # Stack de départ en BB
        self.main_pot = 0
        self.side_pots = self._create_side_pots() 

        self.deck: List[Card] = self._create_deck()
        self.community_cards: List[Card] = []

        self.current_phase = GamePhase.PREFLOP
        self.players = self._initialize_players(agents) # self.players est une liste d'objets Player

        self.button_seat_position = rd.randint(0, 5) # 0-5
        for player in self.players:
            player.role_position = (player.seat_position - self.button_seat_position - 1) % 6

        # Initialiser les variables d'état du jeu
        self.current_player_seat = (self.button_seat_position + 1) % 6 # 0-5 initialisé à SB
        self.current_maximum_bet = 0 # initialisé à 0 mais s'updatera automatiquement à BB après la mise de SB puis BB
        
        self.number_raise_this_game_phase = 0 # Nombre de raises dans la phase courante (4 max, 4 inclus)
        self.last_raiser = None  # Dernier joueur ayant raisé
        self.action_buttons = self._create_action_buttons() # Dictionnaire des boutons d'action, a chaque bouton est associé la propriété enabled qui détermine si le bouton est actif ou non
        
        # --------------------
        # Initialiser pygame et les variables d'interface
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("6-Max Poker")
        self.font = pygame.font.SysFont('Arial', 24)
        self.clock = pygame.time.Clock()

        # Initialiser les éléments de l'interface
        self.pygame_bet_slider = pygame.Rect(50, SCREEN_HEIGHT - 100, 200, 20)
        self.pygame_slider_bet_amount = self.big_blind # Valeur de la raise actualle déterminée grace a un slider (uniquement utilisé en jeu manuel)

        # Suivi de l'historique des actions
        self.pygame_action_history = {'Player_1': [], 'Player_2': [], 'Player_3': [], 'Player_4': [], 'Player_5': [], 'Player_6': []}

        # Ajouter le suivi des informations du gagnant
        self.pygame_winner_info = None

        # Ajouter le timing d'affichage du gagnant
        self.pygame_winner_display_start = 0
        self.pygame_winner_display_duration = 2000  # 2 secondes en millisecondes
        # --------------------

        # Ajouter cette ligne pour stocker l'historique des actions de la main courante
        self.current_hand_history: List[PokerGame.ActionRecord] = []

    def reset(self):
        """
        Réinitialise complètement l'état du jeu pour une nouvelle partie.
        
        Returns:
            List[float]: État initial du jeu après réinitialisation
        """
        # Réinitialiser les variables d'état du jeu
        self.main_pot = 0
        self.side_pots = self._create_side_pots() 
        self.deck = self._create_deck()
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.last_raiser = None

        # Réinitialiser les variables de jeu
        self.button_seat_position = rd.randint(0, 5)  # 0-5 (c'est une toute nouvelle partie donc on réassigne la position du bouton aléatoirement)
        for player in self.players:
            player.role_position = (player.seat_position - self.button_seat_position - 1) % 6
        
        # Réinitialiser les états variables des joueurs
        for player in self.players:
            player.stack = self.starting_stack
            player.current_player_bet = 0
            player.total_bet = 0
            player.cards = []
            player.is_active = True
            player.has_folded = False
            player.is_all_in = False
            player.range = None        
        
        # Initialiser les variables d'état du jeu
        self.current_player_seat = (self.button_seat_position + 1) % 6  # 0-5
        self.current_maximum_bet = 0  # initialisé à 0 mais s'updatera automatiquement à BB
        self.number_raise_this_game_phase = 0

        # --------------------
        # Réinitialiser les variables d'interface Pygame
        self.pygame_winner_info = None
        self.pygame_winner_display_start = 0
        self.pygame_slider_bet_amount = self.big_blind
        self.pygame_action_history = {'Player_1': [], 'Player_2': [], 'Player_3': [], 'Player_4': [], 'Player_5': [], 'Player_6': []}
        # --------------------

        self.start_new_hand()
        self._update_button_states()
        
        return self.get_state_hero()

    def start_new_hand(self):
        """
        Démarre une nouvelle main et réinitialise l'historique.
        """
        # Nettoyer l'historique de la main précédente
        self.current_hand_history.clear()
        
        print('Called start_new_hand')
        # Réinitialiser les variables d'état du jeu
        self.main_pot = 0
        self.side_pots = self._create_side_pots()
        self.deck = self._create_deck()  # Réinitialiser le deck
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP
        self.last_raiser = None

        # Réinitialiser l'état des joueurs
        for player in self.players:
            player.cards = []
            player.current_player_bet = 0
            player.total_bet = 0
            player.is_all_in = False
            player.has_folded = False
            player.range = None
            player.is_active = player.stack > 0

        # --- Nouvelle ligne ajoutée pour enregistrer les stacks initiales des joueurs
        self.initial_stacks = {player.name: player.stack for player in self.players}
        self.net_stack_changes = {player.name: 0 for player in self.players}
        self.final_stacks = {player.name: 0 for player in self.players}

        # Vérifier qu'il y a au moins 2 joueurs actifs sinon on réinitialise la partie
        active_players = [player for player in self.players if player.is_active]
        if len(active_players) < 2:
            self.reset()

        # Distribuer les cartes aux joueurs actifs
        self.deal_cards()

        # Faire tourner le bouton d'une position
        self.button_seat_position = (self.button_seat_position + 1) % self.num_players

        # Construire une liste ordonnée des joueurs actifs en fonction de leur seat_position
        active_players = sorted([p for p in self.players if p.is_active],
                                key=lambda p: p.seat_position)

        # Réorganiser la liste pour que le premier joueur soit celui qui a le bouton
        self.ordered_players = active_players[self.button_seat_position:] + active_players[:self.button_seat_position]
        n = len(self.ordered_players)

        # Réattribuer les rôles en fonction du nombre de joueurs actifs
        if n == 2:
            # Heads-Up : dans le heads-up, le joueur en bouton (qui est également SB) et l'autre (BB)
            self.ordered_players[0].role_position = 5  # Small Blind (et Dealer)
            self.ordered_players[1].role_position = 0  # Big Blind
        elif n == 3:
            # Dans le 3-handed, le joueur en bouton est SB.
            self.ordered_players[0].role_position = 5  # Button (BTN)
            self.ordered_players[1].role_position = 0  # Small Blind (SB)
            self.ordered_players[2].role_position = 1  # Big Blind (BB)
        elif n == 4:
            self.ordered_players[0].role_position = 5  # Button (BTN)
            self.ordered_players[1].role_position = 0  # Small Blind (SB)
            self.ordered_players[2].role_position = 1  # Big Blind (BB)
            self.ordered_players[3].role_position = 2  # UTG
        elif n == 5:
            self.ordered_players[0].role_position = 5  # Button (BTN)
            self.ordered_players[1].role_position = 0  # Small Blind (SB)
            self.ordered_players[2].role_position = 1  # Big Blind (BB) 
            self.ordered_players[3].role_position = 2  # UTG
            self.ordered_players[4].role_position = 3  # Hijack (HJ)
        elif n == 6:
            self.ordered_players[0].role_position = 5  # Button (BTN)
            self.ordered_players[1].role_position = 0  # Small Blind (SB)
            self.ordered_players[2].role_position = 1  # Big Blind (BB)
            self.ordered_players[3].role_position = 2  # UTG
            self.ordered_players[4].role_position = 3  # Hijack (HJ)
            self.ordered_players[5].role_position = 4  # Cutoff (CO)

        # Mettre à jour la position du bouton et du joueur courant en fonction des rôles réattribués
        if n == 2:
            # En heads-up, le Bouton (qui poste la Small Blind) est celui avec role_position == 0
            for player in self.players:
                if player.is_active and player.role_position == 0:
                    self.button_seat_position = player.seat_position
                    break
        else:
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
        self.number_raise_this_game_phase = 0

        # Réinitialiser les variables d'interface (exemple avec Pygame)
        self.pygame_winner_info = None
        self.pygame_winner_display_start = 0
        self.pygame_slider_bet_amount = self.big_blind
        self.pygame_action_history = {'Player_1': [], 'Player_2': [], 'Player_3': [], 'Player_4': [], 'Player_5': [], 'Player_6': []}

        self._update_button_states()
        self.deal_small_and_big_blind()

        # Clear action history for new hand
        self.pygame_action_history = {player.name: [] for player in self.players}

        # In poker_game.py, inside start_new_hand() after resetting players:
        print("Player positions and roles for the new hand:")
        for p in self.players:
            if p.is_active:
                print(f"{p.name}: seat={p.seat_position}, role={p.role_position}, stack={p.stack}")
            else :
                print(f"{p.name}: seat={p.seat_position}, role=(inactive), stack={p.stack}")

        return self.get_state_hero()

    def _initialize_players(self, agents) -> List[Player]:
        """
        Crée et initialise tous les joueurs pour la partie.
        
        Returns:
            List[Player]: Liste des objets joueurs initialisés
        """
        players = []
        for idx, agent in enumerate(agents):
            player = Player(agent=agent, name=agent.name, stack=self.starting_stack, seat_position=idx)
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

        if self.current_phase == GamePhase.PREFLOP:
            raise ValueError(
                "Erreur d'état : Distribution des community cards pendant le pré-flop. "
                "Les cartes communes ne doivent être distribuées qu'après le pré-flop."
            )
        
        if self.current_phase == GamePhase.FLOP:
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
        elif self.current_phase in [GamePhase.TURN, GamePhase.RIVER]:
            self.community_cards.append(self.deck.pop())
    
    def deal_small_and_big_blind(self):
        """
        Méthode à run en début de main pour distribuer automatiquement les blindes
        """
        print('Called deal_small_and_big_blind')
        active_players = [p for p in self.players if p.is_active]
        # Déterminer les positions SB et BB en se basant sur les rôles attribués
        if len(active_players) == 2:
            # Heads-Up : le joueur en position 0 est la Small Blind et le joueur en position 5 est le Big Blind (bouton)
            sb_player = next((p for p in self.players if p.is_active and p.role_position == 0), None)
            bb_player = next((p for p in self.players if p.is_active and p.role_position == 5), None)
        else:
            sb_player = next((p for p in self.players if p.is_active and p.role_position == 0), None)
            bb_player = next((p for p in self.players if p.is_active and p.role_position == 1), None)
        
        if sb_player is None or bb_player is None:
            raise ValueError(
                f"Impossible de déterminer la position de la Small Blind ou Big Blind : "
                f"sb_player={sb_player}, bb_player={bb_player}. Vérifiez l'assignation des rôles des joueurs."
            )
        
        sb_seat_position = sb_player.seat_position
        bb_seat_position = bb_player.seat_position

        # SB
        if self.players[sb_seat_position].stack < self.small_blind:
            self.players[sb_seat_position].is_all_in = True
            self.players[sb_seat_position].current_player_bet = self.players[sb_seat_position].stack  # Le bet du joueur n'ayant pas assez pour payer la SB devient son stack
            self.main_pot += self.players[sb_seat_position].stack  # Le pot est augmenté du stack du joueur
            self.players[sb_seat_position].total_bet = self.players[sb_seat_position].stack
            self.players[sb_seat_position].stack = 0  # Le stack du joueur est donc 0
            self.players[sb_seat_position].has_acted = True
        else:
            self.players[sb_seat_position].stack -= self.small_blind
            self.main_pot += self.small_blind  # Le pot est augmenté de la SB
            self.players[sb_seat_position].total_bet = self.small_blind
            self.players[sb_seat_position].current_player_bet = self.small_blind
            self.players[sb_seat_position].has_acted = True 

        self.current_maximum_bet = self.small_blind
        self._next_player()
        
        # BB
        if self.players[bb_seat_position].stack < self.big_blind:
            self.players[bb_seat_position].is_all_in = True
            self.players[bb_seat_position].current_player_bet = self.players[bb_seat_position].stack  # Le bet du joueur n'ayant pas assez pour payer la BB devient son stack
            self.main_pot += self.players[bb_seat_position].stack  # Le pot est augmenté du stack du joueur
            self.players[bb_seat_position].total_bet = self.players[bb_seat_position].stack
            self.players[bb_seat_position].stack = 0  # Le stack du joueur devient 0
            self.players[bb_seat_position].has_acted = True
        else:
            self.players[bb_seat_position].stack -= self.big_blind
            self.main_pot += self.big_blind  # Le pot est augmenté de la BB
            self.players[bb_seat_position].total_bet = self.big_blind
            self.players[bb_seat_position].current_player_bet = self.big_blind
            # Modification : ne pas marquer la BB comme ayant déjà agi au preflop comme ça si tout le monde a juste call, elle peut checker pour voir le flop ou raise
            self.players[bb_seat_position].has_acted = False
        
        self.current_maximum_bet = self.big_blind
        self._next_player()

    def _next_player(self):
        """
        Passe au prochain joueur actif et n'ayant pas fold dans le sens horaire.
        Skip les joueurs all-in.
        """
        initial_seat = self.current_player_seat
        self.current_player_seat = (self.current_player_seat + 1) % self.num_players
        
        # Vérifier qu'on ne boucle pas indéfiniment
        while (not self.players[self.current_player_seat].is_active or 
               self.players[self.current_player_seat].has_folded or 
               self.players[self.current_player_seat].is_all_in):
            # Ajouter le fait que le joueur a passé son tour dans l'historique
            skipped_player = self.players[self.current_player_seat]
            if skipped_player.has_folded:
                self.pygame_action_history[skipped_player.name].append("none")
                # On ne garde que les 5 dernières actions du joueur
                if len(self.pygame_action_history[skipped_player.name]) > 5:
                    self.pygame_action_history[skipped_player.name].pop(0)
            
            self.current_player_seat = (self.current_player_seat + 1) % self.num_players
            if self.current_player_seat == initial_seat:
                # Affiche l'état de chaque joueur pour faciliter le débogage
                details = [(p.name, p.is_active, p.has_folded, p.is_all_in) for p in self.players]
                raise RuntimeError(
                    "Aucun joueur valide trouvé. Cela signifie que tous les joueurs sont inactifs, foldés ou all-in. "
                    f"Détails des joueurs : {details}"
                )

    def check_phase_completion(self):
        """
        Vérifie si le tour d'enchères actuel est terminé et gère la progression du jeu.
        
        Le tour est terminé quand :
        1. Tous les joueurs actifs ont agi
        2. Tous les joueurs ont égalisé la mise maximale (ou sont all-in)
        3. Cas particuliers : un seul joueur reste, tous all-in, ou BB preflop
        """
        
        # Récupérer les joueurs actifs et all-in
        in_game_players = [p for p in self.players if p.is_active and not p.has_folded]
        all_in_players = [p for p in in_game_players if p.is_all_in]
        
        # Vérifier s'il ne reste qu'un seul joueur actif
        if len(in_game_players) == 1:
            print("Moving to showdown (only one player remains)")
            self.handle_showdown()
            return

        current_player = self.players[self.current_player_seat]

        # Cas particulier, au PREFLOP, si la BB est limpée, elle doit avoir un droit de parole
        # Vérification d'un cas particulier en phase préflop :
        # En phase préflop, l'ordre d'action est particulier car après avoir posté les blinds
        # l'action se prolonge jusqu'à ce que le joueur en petite blinde (role_position == 0) puisse agir.
        # Même si, en apparence, tous les joueurs ont déjà joué et égalisé la mise maximale,
        # il est nécessaire de laisser le temps au joueur en small blind d'intervenir.
        # C'est pourquoi, si le joueur actif est en position 0 durant le préflop,
        # la méthode retourne False et indique que la phase d'enchères ne peut pas encore être terminée
        
        # Si tous les joueurs actifs sont all-in, la partie est terminée, on va vers le showdown pour déterminer le vainqueur
        if (len(all_in_players) == len(in_game_players)) and (len(in_game_players) > 1):
            print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return # Ne rien faire d'autre, la partie est terminée
        
        for player in in_game_players:
            # Si le joueur n'a pas encore agi dans la phase, le tour n'est pas terminé
            if not player.has_acted:
                print(f'{player.name} n\'a pas encore agi')
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée

            # Si le joueur n'a pas égalisé la mise maximale et n'est pas all-in, le tour n'est pas terminé
            if player.current_player_bet < self.current_maximum_bet and not player.is_all_in:
                print('Un des joueurs en jeu n\'a pas égalisé la mise maximale')
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        
        # Atteindre cette partie du code signifie que la phase est terminée
        if self.current_phase == GamePhase.RIVER:
            print("River complete - going to showdown")
            self.handle_showdown()
            return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        else:
            self.advance_phase()
            print(f"Advanced to {self.current_phase}")
            # Réinitialiser has_acted pour tous les joueurs actifs et non fold au début d'une nouvelle phase
            for p in self.players:
                if p.is_active and not p.has_folded and not p.is_all_in:
                    p.has_acted = False

        # Si l'action revient au dernier raiser, terminer le tour d'enchères
        if self.last_raiser is not None and self.current_player_seat == self.last_raiser:
            # Exception : au préflop, la BB qui a limpé (c'est-à-dire dont la mise reste égale à la big blind) peut checker pour voir le flop
            if not (self.current_phase == GamePhase.PREFLOP and 
                    ((current_player.role_position == 1 or current_player.role_position == 5) and 
                     current_player.current_player_bet == self.big_blind)):
                if self.current_phase == GamePhase.RIVER:
                    print("River complete - going to showdown")
                    self.handle_showdown()
                else:
                    self.advance_phase()
                    print(f"Advanced to {self.current_phase}")
                    for p in self.players:
                        if p.is_active and not p.has_folded and not p.is_all_in:
                            p.has_acted = False
                return

        # Nouvelle vérification : si un seul joueur non all-in reste, déclencher le showdown
        # Si un seul joueur non all-in reste, déclencher le showdown car il ne va pas jouer seul.
        # Cette situation arrive lorsque qu'un joueur raise un montant supérieur au stack de ses adversaires. Dès lors, les autres joueurs peuvent soit call, soit all-in. 
        # Or s'ils all-in, le joueur qui a raise est le seul actif, non all-in, non foldé restant. 
        # Dès lors, il n'y a pas de sens à continuer la partie, donc on va au showdown.
        non_all_in_players = [p for p in in_game_players if not p.is_all_in]
        if len(non_all_in_players) == 1 and len(in_game_players) > 1:
            print("Moving to showdown (only one non all-in player remains)")
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return

    def advance_phase(self):
        """
        Passe à la phase suivante du jeu (préflop -> flop -> turn -> river).
        Distribue les cartes communes appropriées et réinitialise les mises.
        """
        print(f"current_phase {self.current_phase}")
        self.last_raiser = None  # Réinitialiser le dernier raiser pour la nouvelle phase
        
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
                player.current_player_bet = 0
                if not player.has_folded and not player.is_all_in:
                    player.has_acted = False  # Réinitialisation du flag
        
        # Set first player after dealer button
        self.current_player_seat = (self.button_seat_position + 1) % self.num_players
        while (not self.players[self.current_player_seat].is_active or 
               self.players[self.current_player_seat].has_folded or 
               self.players[self.current_player_seat].is_all_in):
            self.current_player_seat = (self.current_player_seat + 1) % self.num_players
    
    def _update_button_states(self):
        """
        Met à jour l'état activé/désactivé des boutons d'action.
        Prend en compte la phase de jeu et les règles du poker.
        """
        current_player = self.players[self.current_player_seat]
        
        # Désactiver tous les boutons si le joueur est all-in
        if current_player.is_all_in:
            for button in self.action_buttons.values():
                button.enabled = False
            return
        
        # ---- Activer tous les boutons par défaut ----
        for button in self.action_buttons.values():
            button.enabled = True
        
        # ---- CHECK ----
        if current_player.current_player_bet < self.current_maximum_bet:  # Si le joueur n'a pas égalisé la mise maximale, il ne peut pas check
            self.action_buttons[PlayerAction.CHECK].enabled = False
        
        # ---- FOLD ----
        if self.action_buttons[PlayerAction.CHECK].enabled:  # Si le joueur peut check, il ne peut pas fold
            self.action_buttons[PlayerAction.FOLD].enabled = False
        
        # ---- CALL ----
        # Désactiver call si pas de mise à suivre ou pas assez de jetons
        if current_player.current_player_bet == self.current_maximum_bet:  # Si le joueur a égalisé la mise maximale, il ne peut pas call
            self.action_buttons[PlayerAction.CALL].enabled = False
        elif current_player.stack < (self.current_maximum_bet - current_player.current_player_bet): # Si le joueur n'a pas assez de jetons pour suivre la mise maximale, il ne peut pas call
            self.action_buttons[PlayerAction.CALL].enabled = False
            # Activer all-in si le joueur a des jetons même si insuffisants pour call
            if current_player.stack > 0:
                self.action_buttons[PlayerAction.ALL_IN].enabled = True
            self.action_buttons[PlayerAction.RAISE].enabled = False
        
        # ---- RAISE standard ----
        if self.current_maximum_bet == 0:
            min_raise = self.big_blind
        else:
            min_raise = (self.current_maximum_bet - current_player.current_player_bet) * 2
        
        if current_player.stack < min_raise:
            self.action_buttons[PlayerAction.RAISE].enabled = False

        # Désactiver raise si déjà 4 relances dans le tour
        if self.number_raise_this_game_phase >= 4:
            self.action_buttons[PlayerAction.RAISE].enabled = False
        
        # ---- Raises pot-based (custom) ----
        pot_raise_actions = [
            PlayerAction.RAISE_25_POT,
            PlayerAction.RAISE_33_POT,
            PlayerAction.RAISE_50_POT,
            PlayerAction.RAISE_66_POT,
            PlayerAction.RAISE_75_POT,
            PlayerAction.RAISE_100_POT,
            PlayerAction.RAISE_125_POT,
            PlayerAction.RAISE_150_POT,
            PlayerAction.RAISE_175_POT,
            PlayerAction.RAISE_2X_POT,
            PlayerAction.RAISE_3X_POT
        ]
        raise_percentages = {
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
        for action in pot_raise_actions:
            # Si le nombre de raises dans la phase est déjà atteint, désactiver la raise pot-based
            if self.number_raise_this_game_phase >= 4:
                self.action_buttons[action].enabled = False
            else:
                percentage = raise_percentages[action]
                # Calcul du montant d'augmentation requis via le pourcentage du pot
                required_increase = self.main_pot * percentage
                # Si le joueur n'a pas suffisamment de jetons pour augmenter d'au moins "required_increase" ou si le required_increase est inférieur à min_raise, désactiver le bouton
                if current_player.stack < required_increase or required_increase < min_raise:
                    self.action_buttons[action].enabled = False
        
        # ---- ALL-IN ----
        # All-in disponible si le joueur a des jetons, qu'il soit le premier à agir ou non
        self.action_buttons[PlayerAction.ALL_IN].enabled = current_player.stack > 0
        
        if self.current_phase == GamePhase.SHOWDOWN:
            for button in self.action_buttons.values():
                button.enabled = False

    def _create_action_buttons(self) -> Dict[PlayerAction, Button]:
        """
        Crée et initialise les boutons d'action pour l'interaction des joueurs.
        
        Returns:
            Dict[PlayerAction, Button]: Dictionnaire associant les actions aux objets boutons
        """
        buttons = {
            PlayerAction.FOLD: Button(300, SCREEN_HEIGHT - 50, 100, 40, "Fold", (200, 0, 0)),
            PlayerAction.CHECK: Button(450, SCREEN_HEIGHT - 50, 100, 40, "Check", (0, 200, 0)),
            PlayerAction.CALL: Button(600, SCREEN_HEIGHT - 50, 100, 40, "Call", (0, 0, 200)),
            PlayerAction.RAISE: Button(750, SCREEN_HEIGHT - 50, 100, 40, "Raise", (200, 200, 0)),
            PlayerAction.ALL_IN: Button(900, SCREEN_HEIGHT - 50, 100, 40, "All-in", (150, 0, 150))
        }
        # Ajout des boutons pour les raises pot-based
        pot_raise_color = (255, 140, 0)  # Couleur orange pour les raises pot-based
        # Première rangée (6 boutons)
        start_x_row1 = 315
        y_row1 = SCREEN_HEIGHT - 140
        btn_width = 120
        btn_height = 30
        gap = 5
        buttons[PlayerAction.RAISE_25_POT] = Button(start_x_row1, y_row1, btn_width, btn_height, "25%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_33_POT] = Button(start_x_row1 + (btn_width + gap) * 1, y_row1, btn_width, btn_height, "33%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_50_POT] = Button(start_x_row1 + (btn_width + gap) * 2, y_row1, btn_width, btn_height, "50%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_66_POT] = Button(start_x_row1 + (btn_width + gap) * 3, y_row1, btn_width, btn_height, "66%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_75_POT] = Button(start_x_row1 + (btn_width + gap) * 4, y_row1, btn_width, btn_height, "75%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_100_POT] = Button(start_x_row1 + (btn_width + gap) * 5, y_row1, btn_width, btn_height, "100%Pot", pot_raise_color)
        
        # Deuxième rangée (5 boutons)
        start_x_row2 = 380
        y_row2 = SCREEN_HEIGHT - 100
        buttons[PlayerAction.RAISE_125_POT] = Button(start_x_row2, y_row2, btn_width, btn_height, "125%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_150_POT] = Button(start_x_row2 + (btn_width + gap) * 1, y_row2, btn_width, btn_height, "150%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_175_POT] = Button(start_x_row2 + (btn_width + gap) * 2, y_row2, btn_width, btn_height, "175%Pot", pot_raise_color)
        buttons[PlayerAction.RAISE_2X_POT]  = Button(start_x_row2 + (btn_width + gap) * 3, y_row2, btn_width, btn_height, "2xPot", pot_raise_color)
        buttons[PlayerAction.RAISE_3X_POT]  = Button(start_x_row2 + (btn_width + gap) * 4, y_row2, btn_width, btn_height, "3xPot", pot_raise_color)
        
        return buttons

    def process_action(self, player: Player, action: PlayerAction, bet_amount: Optional[int] = None):
        """
        Traite l'action d'un joueur et l'enregistre dans l'historique.
        """
        # Enregistrer l'état avant l'action
        current_state = self.get_state_hero()


        # Variables pour enregistrer l'état avant l'action pour l'historique
        action_phase = self.current_phase
        start_stack = player.stack
        pot_before_the_action = self.main_pot

        """
        Traite l'action d'un joueur, met à jour l'état du jeu et gère la progression du tour.

        Cette méthode réalise plusieurs vérifications essentielles :
        - S'assurer que le joueur dispose de suffisamment de fonds.
        - Interrompre le traitement en cas de phase SHOWDOWN.
        - Construire un historique des actions pour le suivi.
        - Gérer distinctement les différents types d'actions : FOLD, CHECK, CALL, RAISE et ALL_IN.
        - Mettre à jour le pot, les mises des joueurs et la mise maximale en cours.
        - Traiter les situations d'all-in et créer des side pots le cas échéant.
        - Déterminer, à l'issue de l'action, si le tour d'enchères est clôturé ou s'il faut passer au joueur suivant.
        
        Returns:
            PlayerAction: L'action traitée (pour garder une cohérence dans le type de retour).
        """
        #----- Vérification que c'est bien au tour du joueur de jouer -----
        if player.seat_position != self.current_player_seat:
            current_turn_player = self.players[self.current_player_seat].name
            raise ValueError(
                f"Erreur d'action : Ce n'est pas le tour de {player.name}. "
                f"C'est au tour de {current_turn_player} d'agir."
            )
        #----- Vérification des fonds disponibles -----
        if not player.is_active or player.is_all_in or player.has_folded or self.current_phase == GamePhase.SHOWDOWN:
            raise ValueError(f"{player.name} n'était pas censé pouvoir faire une action, Raisons : actif = {player.is_active}, all-in = {player.is_all_in}, folded = {player.has_folded}")
        
        valid_actions = [a for a in PlayerAction if self.action_buttons[a].enabled]
        if action not in valid_actions:
            raise ValueError(f"{player.name} n'a pas le droit de faire cette action, actions valides : {valid_actions}")
        
        #----- Affichage de débogage (pour le suivi durant l'exécution) -----
        print(f"\n=== Action par {player.name} ===")
        print(f"Joueur actif : {player.is_active}")
        print(f"Action choisie : {action.value}")
        print(f"Phase actuelle : {self.current_phase}")
        print(f"Pot actuel : {self.main_pot}BB")
        print(f"A agi : {player.has_acted}")
        print(f"Est all-in : {player.is_all_in}")
        print(f"Mise maximale actuelle : {self.current_maximum_bet}BB")
        print(f"Stack du joueur avant action : {player.stack}BB")
        print(f"Mise actuelle du joueur : {player.current_player_bet}BB")
        
        #----- Traitement de l'action en fonction de son type -----
        if action == PlayerAction.FOLD:
            # Le joueur se couche il n'est plus actif pour ce tour.
            player.has_folded = True
            print(f"{player.name} se couche (Fold).")
            history_amount = -1
        
        elif action == PlayerAction.CHECK:
            print(f"{player.name} check.")
            history_amount = 0
        elif action == PlayerAction.CALL:
            print(f"{player.name} call.")
            call_amount = self.current_maximum_bet - player.current_player_bet
            if call_amount > player.stack: 
                raise ValueError(f"{player.name} n'a pas assez de jetons pour suivre la mise maximale, il n'aurait pas du avoir le droit de call")
        
            player.stack -= call_amount
            player.current_player_bet += call_amount
            self.main_pot += call_amount
            player.total_bet += call_amount
            if player.stack == 0:
                player.is_all_in = True
            print(f"{player.name} a call {call_amount}BB")
            history_amount = call_amount  # Ajout de cette ligne pour enregistrer le montant du call

        elif action == PlayerAction.RAISE:
            print(f"{player.name} raise.")
            # Si aucune mise n'a encore été faite, fixer un minimum raise basé sur la big blind.
            if self.current_maximum_bet == 0:
                min_raise = self.big_blind
            else:
                min_raise = (self.current_maximum_bet - player.current_player_bet) * 2
        
            # Si aucune valeur n'est fournie ou si elle est inférieure au minimum, utiliser le minimum raise.
            if bet_amount is None or (bet_amount < min_raise and action != PlayerAction.ALL_IN):
                bet_amount = min_raise
        
            # Vérifier si le joueur a assez de jetons pour couvrir le montant de raise.
            if player.stack < (bet_amount - player.current_player_bet):
                raise ValueError(
                    f"Fonds insuffisants pour raise : {player.name} a {player.stack}BB tandis que le montant "
                    f"additionnel requis est {bet_amount - player.current_player_bet}BB. Mise minimum requise : {min_raise}BB."
                )
        
            # Traitement du raise standard
            actual_bet = bet_amount - player.current_player_bet  # Calculer combien de jetons le joueur doit ajouter
            player.stack -= actual_bet
            player.current_player_bet = bet_amount
            self.main_pot += actual_bet
            player.total_bet += actual_bet
            self.current_maximum_bet = bet_amount
            self.number_raise_this_game_phase += 1
            self.last_raiser = player.seat_position
        
            print(f"{player.name} a raise {bet_amount}BB")
        
        # --- Nouvelles actions pot-based ---
        elif action in {
            PlayerAction.RAISE_25_POT,
            PlayerAction.RAISE_33_POT,
            PlayerAction.RAISE_50_POT,
            PlayerAction.RAISE_66_POT,
            PlayerAction.RAISE_75_POT,
            PlayerAction.RAISE_100_POT,
            PlayerAction.RAISE_125_POT,
            PlayerAction.RAISE_150_POT,
            PlayerAction.RAISE_175_POT,
            PlayerAction.RAISE_2X_POT,
            PlayerAction.RAISE_3X_POT
        }:
            raise_percentages = {
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
            percentage = raise_percentages[action]
            # Calcul de la raise additionnelle basée sur le pourcentage du pot
            custom_raise_amount = self.main_pot * percentage
            # La nouvelle mise est : la mise actuelle + montant pour caller + raise additionnel
            computed_bet = player.current_player_bet + custom_raise_amount
        
            if self.current_maximum_bet == 0:
                min_raise = self.big_blind
            else:
                min_raise = (self.current_maximum_bet - player.current_player_bet) * 2
        
            # Vérifier que le montant additionnel respecte le minimum exigé
            if computed_bet - player.current_player_bet < min_raise:
                computed_bet = player.current_player_bet + min_raise
        
            bet_amount = computed_bet
        
            # Vérifier que le joueur a suffisamment de jetons pour cette raise
            if player.stack < (bet_amount - player.current_player_bet):
                raise ValueError(
                    f"Fonds insuffisants pour raise : {player.name} a {player.stack}BB tandis que le montant "
                    f"additionnel requis est {bet_amount - player.current_player_bet}BB. Mise minimum requise : {min_raise}BB."
                )
        
            # Traitement de la raise pot-based
            actual_bet = bet_amount - player.current_player_bet  # Calcul du supplément à miser
            player.stack -= actual_bet
            player.current_player_bet = bet_amount
            self.main_pot += actual_bet
            player.total_bet += actual_bet
            self.current_maximum_bet = bet_amount
            self.number_raise_this_game_phase += 1
            self.last_raiser = player.seat_position
        
            print(f"{player.name} a raise (pot-based {percentage*100:.0f}%) à {bet_amount}BB")
        
        
        elif action == PlayerAction.ALL_IN:
            print(f"{player.name} all-in.")
            # Si aucune valeur n'est passée pour bet_amount, on assigne automatiquement tout le stack
            if bet_amount is None:
                bet_amount = player.stack
            elif bet_amount != player.stack:
                raise ValueError(
                    f"Erreur ALL-IN : {player.name} doit miser exactement tout son stack ({player.stack}BB)."
                )
            
            # Mise à jour de la mise maximale seulement si l'all-in est supérieur
            if bet_amount + player.current_player_bet > self.current_maximum_bet:  # Si le all-in est supérieur à la mise maximale, on met à jour la mise maximale
                self.current_maximum_bet = bet_amount + player.current_player_bet  # On met à jour la mise maximale
                self.number_raise_this_game_phase += 1  # On incrémente le nombre de raise dans la phase
                self.last_raiser = player.seat_position  # Enregistrer le all-in comme raise
            
            player.stack -= bet_amount  # On retire le all-in du stack du joueur
            player.current_player_bet += bet_amount  # On ajoute le all-in à la mise du joueur
            self.main_pot += bet_amount  # On ajoute le all-in au pot de la phase
            player.total_bet += bet_amount  # On ajoute le all-in à la mise totale du joueur
            player.is_all_in = True  # On indique que le joueur est all-in
            print(f"{player.name} a all-in {bet_amount}BB")


        pot_after_the_action = self.main_pot
        
        player.has_acted = True
        self.check_phase_completion()
        
        # Mise à jour de l'historique des actions du joueur
        action_text = f"{action.value}"
        if action in [PlayerAction.RAISE, PlayerAction.ALL_IN] or action in {
            PlayerAction.RAISE_25_POT,
            PlayerAction.RAISE_33_POT,
            PlayerAction.RAISE_50_POT,
            PlayerAction.RAISE_66_POT,
            PlayerAction.RAISE_75_POT,
            PlayerAction.RAISE_100_POT,
            PlayerAction.RAISE_125_POT,
            PlayerAction.RAISE_150_POT,
            PlayerAction.RAISE_175_POT,
            PlayerAction.RAISE_2X_POT,
            PlayerAction.RAISE_3X_POT
        }:
            action_text += f" {bet_amount}BB"
        elif action == PlayerAction.CALL:
            action_text += f" {history_amount}BB"  # Utiliser history_amount au lieu de call_amount qui n'est pas accessible ici
        
        # Add action to player's history
        self.pygame_action_history[player.name].append(action_text)
        # Keep only last 5 actions per player
        if len(self.pygame_action_history[player.name]) > 5:
            self.pygame_action_history[player.name].pop(0)

        if action == PlayerAction.FOLD:
            amount = -1
        elif action == PlayerAction.CHECK:
            amount = 0
        elif action == PlayerAction.CALL:
            amount = history_amount  # Utiliser le montant du call calculé plus haut
        else:
            amount = bet_amount
        # Créer un enregistrement de l'action en utilisant la classe interne
        action_record = self.ActionRecord(
            phase=action_phase,                  
            player=player.name,
            position=player.role_position,
            stack_of_player_before_the_action=start_stack,
            pot_before_the_action=pot_before_the_action,
            action_taken=action,
            stack_of_player_after_the_action=player.stack,
            pot_after_the_action=pot_after_the_action,
            bet_amount=amount
        )
        self.current_hand_history.append(action_record)

        print(self.print_hand_history())
        print(self.get_tokenized_history())
        
        return action

    def evaluate_final_hand(self, player: Player) -> Tuple[HandRank, List[int]]:
        if not player.cards:
            raise ValueError(
                f"Erreur d'évaluation : le joueur {player.name} n'a pas de cartes pour évaluer sa main. "
                "Assurez-vous que les cartes ont été distribuées correctement."
            )
        """
        Évalue la meilleure main possible d'un joueur.
        """
        if not player.cards:
            raise ValueError("Cannot evaluate hand - player has no cards")
        
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
            kicker = max(kickers) if kickers else 0
            return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
        
        # Vérifie la paire simple
        if pairs:
            # Garde les 3 meilleures cartes restantes comme kickers
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, [pairs[0]] + kickers)
        
        # Si aucune combinaison, retourne la carte haute avec les 5 meilleures cartes
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])

    def handle_showdown(self):
        """
        Gère la phase de showdown en tenant compte correctement des side pots.
        """
        print(self.print_hand_history())
        print("\n=== DÉBUT SHOWDOWN ===")
        self.current_phase = GamePhase.SHOWDOWN
        # On considère ici TOUS les joueurs qui ont contribué (même s'ils ont foldé)
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        print(f"Joueurs actifs au showdown: {[p.name for p in active_players]}")
        
        # Désactiver tous les boutons pendant le showdown
        for button in self.action_buttons.values():
            button.enabled = False
        
        # S'assurer que toutes les cartes communes sont distribuées
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        print(f"Cartes communes finales: {[str(card) for card in self.community_cards]}")
        
        # Afficher les mains des joueurs actifs
        print("\nMains des joueurs actifs:")
        for player in active_players:
            print(f"- {player.name}: {[str(card) for card in player.cards]}")
        
        # --- Distribution des gains ---
        # Cas particulier : victoire par fold (il ne reste qu'un joueur actif)
        if len(active_players) == 1:
            winner = active_players[0]
            contributions = {player: player.total_bet for player in self.players if player.total_bet > 0}
            total_pot = sum(contributions.values())
            print(f"\nVictoire par fold - {winner.name} gagne {total_pot:.2f}BB")
            winner.stack += total_pot
            self.pygame_winner_info = f"{winner.name} gagne {total_pot:.2f}BB (tous les autres joueurs ont fold)"
        else:
            # --- Répartition des contributions en pots (Main Pot et Side Pots) ---
            # IMPORTANT : On prend en compte tous les joueurs qui ont misé,
            # même s'ils se sont couchés.
            contributions = {player: player.total_bet for player in self.players if player.total_bet > 0}
            
            print("\nContributions totales (tous joueurs ayant misé):")
            for player, amount in contributions.items():
                print(f"- {player.name}: {amount:.2f}BB")
            
            pots = []
            pot_index = 0
            
            # Tant qu'au moins un joueur a une contribution positive
            while any(amount > 0 for amount in contributions.values()):
                # Les joueurs qui ont encore misé quelque chose (foldés ou non)
                current_contributors = [player for player, amount in contributions.items() if amount > 0]
                if not current_contributors:
                    break
                # Contribution minimale parmi tous les joueurs concernés
                min_contrib = min(contributions[player] for player in current_contributors)
                # Le montant du pot est : min_contrib * (nombre total de contributeurs)
                pot_amount = min_contrib * len(current_contributors)
                # Seuls les joueurs non-foldés sont éligibles pour remporter ce pot
                eligible = [player for player in current_contributors if not player.has_folded]
                
                pot_name = "Main Pot" if pot_index == 0 else f"Side Pot {pot_index}"
                pots.append({
                    "name": pot_name,
                    "amount": pot_amount,
                    "eligible": eligible
                })
                
                print(f"\n{pot_name} calculé:")
                print(f"- Contribution minimale: {min_contrib:.2f}BB")
                print(f"- Nombre total de contributeurs: {len(current_contributors)}")
                print(f"- Montant du pot: {pot_amount:.2f}BB")
                print(f"- Joueurs éligibles: {[p.name for p in eligible]}")
                
                # Déduire la contribution minimale de tous les joueurs contributeurs
                for player in contributions:
                    if contributions[player] > 0:
                        contributions[player] -= min_contrib
                pot_index += 1
            
            # --- Distribution des gains pour chaque pot ---
            print("\nDistribution des gains par pot:")
            for pot in pots:
                print(f"\nÉvaluation du {pot['name']} ({pot['amount']:.2f}BB):")
                if not pot["eligible"]:
                    print("Aucun joueur éligible pour ce pot.")
                    continue
                
                best_eval = None
                winners = []
                print("Évaluation des mains:")
                for player in pot["eligible"]:
                    hand_eval = self.evaluate_final_hand(player)
                    # Création d'une description textuelle de la main (exemple simplifié)
                    if hand_eval[0] == HandRank.ROYAL_FLUSH:
                        hand_str = "Quinte Flush Royale"
                    elif hand_eval[0] == HandRank.STRAIGHT_FLUSH:
                        hand_str = f"Quinte Flush hauteur {hand_eval[1][0]}"
                    elif hand_eval[0] == HandRank.FOUR_OF_A_KIND:
                        hand_str = f"Carré de {hand_eval[1][0]}, kicker {hand_eval[1][1]}"
                    elif hand_eval[0] == HandRank.FULL_HOUSE:
                        hand_str = f"Full aux {hand_eval[1][0]} par les {hand_eval[1][1]}"
                    elif hand_eval[0] == HandRank.FLUSH:
                        hand_str = f"Couleur {', '.join(str(x) for x in hand_eval[1])}"
                    elif hand_eval[0] == HandRank.STRAIGHT:
                        hand_str = f"Quinte hauteur {hand_eval[1][0]}"
                    elif hand_eval[0] == HandRank.THREE_OF_A_KIND:
                        hand_str = f"Brelan de {hand_eval[1][0]}, kickers {hand_eval[1][1:]}"
                    elif hand_eval[0] == HandRank.TWO_PAIR:
                        hand_str = f"Double paire {hand_eval[1][0]} et {hand_eval[1][1]}, kicker {hand_eval[1][2]}"
                    elif hand_eval[0] == HandRank.PAIR:
                        hand_str = f"Paire de {hand_eval[1][0]}, kickers {hand_eval[1][1:]}"
                    else:
                        hand_str = f"Carte haute: {', '.join(str(x) for x in hand_eval[1])}"
        
                    print(f"- {player.name}: {hand_str}")
                    current_key = (hand_eval[0].value, tuple(hand_eval[1]))
                    if best_eval is None or current_key > best_eval:
                        best_eval = current_key
                        winners = [player]
                    elif current_key == best_eval:
                        winners.append(player)
        
                if winners:
                    share = pot["amount"] / len(winners)
                    print(f"Gagnant(s): {[w.name for w in winners]}, {share:.2f}BB chacun")
                    for winner in winners:
                        winner.stack += share
                else:
                    share = 0
                    print("Aucun gagnant déterminé pour ce pot.")
        
            # (Optionnel) Mettre à jour l'info affichée par Pygame
            self.pygame_winner_info = "\n".join(
                [f"{pot['name']} ({pot['amount']:.2f}BB): {[p.name for p in pot['eligible'] if not p.has_folded]} gagnent {pot['amount'] / len([p for p in pot['eligible'] if not p.has_folded]):.2f}BB"
                for pot in pots if len([p for p in pot['eligible'] if not p.has_folded]) > 0]
            )
        
        print("\nStacks initiaux:")
        for player in self.players:
            print(f"- {player.name}: {self.initial_stacks[player.name]}BB")
        
        print("\nStacks finaux:")
        for player in self.players:
            print(f"- {player.name}: {player.stack:.2f}BB")
        
        print("\nVariations :")
        for player in self.players:
            initial = self.initial_stacks.get(player.name, 0)
            variation = player.stack - initial
            signe = "+" if variation >= 0 else ""
            print(f"- {player.name}: {signe}{variation:.2f}BB")
        
        # Calcul des changements nets de piles
        self.net_stack_changes = {player.name: (player.stack - self.initial_stacks.get(player.name, 0)) for player in self.players}
        self.final_stacks = {player.name: player.stack for player in self.players}
            
        # Réinitialiser les pots
        self.main_pot = 0
        self.side_pots = self._create_side_pots() 
            
        # Paramétrer le début de l'affichage du gagnant via Pygame
        self.pygame_winner_display_start = pygame.time.get_ticks()
        self.pygame_winner_display_duration = 2000  # 2 secondes
        print("=== FIN SHOWDOWN ===\n")

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

    def _create_side_pots(self) -> List[SidePot]:
        """
        Crée 6 side pots vierges.
        
        Returns:
            List[SidePot]: Liste de 6 side pots vierges
        """

        side_pots = []
        for i in range(6):
            side_pots.append(SidePot(id=i))

        return side_pots

    def _distribute_side_pots(self, in_game_players: List[Player], side_pots: List[SidePot], main_pot: float):
        """
        Répartit les surplus en side pots.

        Pour la répartition en side pots, on laisse les joueurs - qui ne seront pas capable d'atteindre le maxbet - all-in dans le main pot 
        On attend la fin de la phase.
        Si les joueurs les plus pauvres sont all-in et que les plus riches sont soit all-in aussi soit à un bet égal au bet_maximum, 
        La phase est terminée et on réparti les surplus en side pots.
        
        _distribute_side_pots est appelée sachant qu'on moins un joueur est all-in, et que tous les non-all-in égalisent la mise maximale.
        side_pot_list est une liste de 4 SidePot, on verra d'après la logique suivante que maximum 4 SidePots distincts seront nécessaires pour une partie à 6 joueurs.

        Exemple : J1 et J2 sont pauvres et sont all-in. J3 et J4 sont plus riches qu'eux, non all-in avec des bets égaux à la mise maximale.
        Notons s_i la mise du joueur i. s_1 < s_2 < s_3 = s_4.
        J1 met toute sa mise dans le main pot.
        J2 met s_1 dans le main pot et met s_2 - s_1 dans le premier side pot.
        J3 et J4 mettent s_1 dans le main pot, puis s_2 - s_1 dans le premier side pot.
        Il leur reste s_3 - s_2 qu'il mettent dans le deuxième side pot.        
        """
        ordered_players = sorted(in_game_players, key=lambda x: x.current_player_bet)
        ordered_bets = [p.current_player_bet for p in ordered_players]
        
        nb_equal_diff = 0
        for i in range(len(ordered_players)):

            diff_bet = ordered_bets[i] - (ordered_bets[i+1] if i < len(ordered_players) - 1 else 0)
            if diff_bet < 0:
                for player in ordered_players[i-nb_equal_diff:]:
                    side_pots[i].contributions_dict[player] = diff_bet
                    ordered_bets[i] -= diff_bet
                side_pots[i].sum_of_contributions = diff_bet * (len(ordered_players) - i - nb_equal_diff +1)
                nb_equal_diff = 0
            else:
                nb_equal_diff +=1
        
        return side_pots

    
    def get_state_hero(self):
        """
        Retourne l'état du jeu du point de vue du joueur courant sous forme d'un tenseur.
        
        Le tenseur d'état contient 190 dimensions réparties comme suit :

        1. Phase de jeu [0-4] :
            - One-hot encoding sur 5 dimensions (PREFLOP, FLOP, TURN, RIVER, SHOWDOWN)
        
        2. Stack et position du joueur courant [5-11] :
            - Stack normalisé (divisé par le stack initial) [5]
            - Position one-hot sur 6 dimensions (SB, BB, UTG, HJ, CO, BTN) [6-11]
        
        3. Cartes personnelles [12-45] :
            - 2 cartes × 17 dimensions par carte = 34 dimensions
            - Pour chaque carte :
                - Valeur one-hot (2-14) sur 13 dimensions
                - Couleur one-hot (♠,♥,♦,♣) sur 4 dimensions
        
        4. Cartes communes [46-130] :
            - 5 cartes × 17 dimensions par carte = 85 dimensions
            - Pour chaque carte :
                - Valeur one-hot (2-14) sur 13 dimensions
                - Couleur one-hot (♠,♥,♦,♣) sur 4 dimensions
        
        5. Mise maximale actuelle [131] :
            - Valeur normalisée (divisée par le stack initial)
        
        6. Informations sur les autres joueurs [132-173] :
            - 6 joueurs × 7 dimensions = 42 dimensions
            - Pour chaque joueur :
                - Stack normalisé [0]
                - Position one-hot sur 6 dimensions [1-6]
                - Si joueur inactif/fold : -1 sur toutes les dimensions
                - Si joueur courant : 0 sur toutes les dimensions
        
        7. Actions disponibles [174-189] :
            - 16 dimensions (une par action possible)
            - 1 si l'action est disponible
            - -1 si l'action est indisponible

        Returns:
            torch.Tensor: Tenseur de dimension 190 représentant l'état complet du jeu
        """
        current_player = self.players[self.current_player_seat]

        # Correspondance des couleurs avec des nombres ♠, ♥, ♦, ♣
        suit_map = {"♠": 0, "♥": 1, "♦": 2, "♣": 3}

        # 1. ------ Informations sur la phase de jeu ------ [0-4]
        info_phase = torch.zeros(5)
        phase_values = {GamePhase.PREFLOP: 0, GamePhase.FLOP: 1, GamePhase.TURN: 2, GamePhase.RIVER: 3, GamePhase.SHOWDOWN: 4}
        # On récupère directement l'indice correspondant à la phase courante
        phase_idx = phase_values[self.current_phase]
        # Puis, on place 1 à la bonne position dans state (indices 0 à 4)
        info_phase[phase_idx] = 1

        # 2. ------ Informations sur la position et le stack du current_player ------ [5-11]
        info_stack = torch.zeros(7)
        # Normalisation du stack du joueur courant et affectation directe
        info_stack[0] = current_player.stack / self.starting_stack
        # One-hot encoding pour la position du joueur courant
        info_stack[1 + current_player.role_position] = 1

        # 3. ------ Cartes Personnelles ------ [12-45]
        # Remplir directement state pour les 5 cartes communes (2 x 17 = 34 éléments)
        info_cards = torch.zeros(34)
        for i, card in enumerate(current_player.cards):
            base_idx = i * 17
            # One-hot encoding pour la valeur (indices 0 à 12 pour 2-14)
            info_cards[base_idx + (card.value - 2)] = 1
            # One-hot encoding pour la couleur (indices 13 à 16)
            info_cards[base_idx + 13 + suit_map[card.suit]] = 1 # + 13 pour être après la valeur de la carte

        
        # 4. ------ Cartes communes ------ [46-130]
        info_community_cards = torch.zeros(85)
        # Remplir directement state pour les 5 cartes communes (5 x 17 = 85 éléments)
        for i, card in enumerate(self.community_cards):
            base_idx = i * 17
            # One-hot encoding pour la valeur (indices 0 à 12 pour 2-14)
            info_community_cards[base_idx + (card.value - 2)] = 1
            # One-hot encoding pour la couleur (indices 13 à 16)
            info_community_cards[base_idx + 13 + suit_map[card.suit]] = 1


        # 5. ------ Informations sur la mise actuelle ------ [131]
        actual_bet = torch.tensor([self.current_maximum_bet / self.starting_stack])

        # 6. ------ Informations sur les stacks et la position des joueurs des joueurs restants ------ [132-173]
        info_players = torch.zeros(42)
        for i, player in enumerate(self.players):
            # Calculer l'index de base pour ce joueur dans le vecteur d'état
            base_idx = i * 7  # 7 dimensions par joueur (1 pour stack + 6 pour position)
            
            # si c'est le current_player alors on met 0 pour le stack et la position
            if player == current_player:
                info_players[base_idx] = 0  # Stack à 0
                for j in range(base_idx + 1, base_idx + 7):  # Reset position encoding
                    info_players[j] = 0
            # si le joueur est inactif ou folded alors on met -1 pour le stack et la position
            elif not player.is_active or player.has_folded:
                info_players[base_idx] = -1  # Stack à -1
                for j in range(base_idx + 1, base_idx + 7):  # Reset position encoding
                    info_players[j] = -1
            else:
                # Normaliser le stack du joueur
                info_players[base_idx] = player.stack / self.starting_stack
                
                # Encoder la position en one-hot (6 positions possibles)
                position_idx = base_idx + 1 + player.role_position
                info_players[position_idx] = 1

        # 7. ------ Actions disponibles ------ [174-189]
        info_actions = torch.zeros(16)
        for idx, action in enumerate(PlayerAction):
            if action in self.action_buttons and self.action_buttons[action].enabled:
                info_actions[idx] = 1
            else:
                info_actions[idx] = -1

        state = torch.cat((info_phase, info_stack, info_cards, info_community_cards, actual_bet, info_players, info_actions))
        return state

    def get_final_state_hero(self, previous_state, final_stacks):
        """
        Reconstruit l'Etat final de la main pour get_state_hero.

        Cette méthode prend l'état précédent et les stacks finaux pour construire un état 
        représentant la fin de la main. Les modifications suivantes sont appliquées :

        1. Phase de jeu : Mise à jour vers SHOWDOWN [0, 0, 0, 0, 1]
        2. Mise actuelle : Remise à zéro car la main est terminée
        3. Stacks des joueurs : Mise à jour avec les valeurs finales (normalisées par le stack initial)
        4. Actions disponibles : Toutes désactivées (-1) car la main est terminée

        Args:
            previous_state (torch.Tensor): État précédent du jeu obtenu via get_state_hero
            final_stacks (dict): Dictionnaire {nom_joueur: stack_final} contenant les stacks finaux

        Returns:
            torch.Tensor: Etat final modifié pour la fin de la main (version get_state).
        """
        final_state = previous_state.clone()

        # 1. Mettre à jour la phase de jeu en SHOWDOWN (indices 0:5)
        final_state[0:5] = torch.tensor([0, 0, 0, 0, 1], dtype=final_state.dtype)

        # 2. Mettre l'actual_bet (indice 131) à 0
        final_state[131] = 0.0

        # 3. Mettre à jour les stacks des joueurs dans info_players.
        # Pour chaque joueur, info_players occupe 7 dimensions à partir de l'indice 132.
        # La première dimension de chaque bloc correspond au stack.
        for i, player in enumerate(self.players):
            final_state[132 + i * 7] = final_stacks[player.name] / self.starting_stack

        # 4. Désactiver toutes les actions disponibles (info_actions aux indices 174 à 189)
        final_state[174:189] = 0

        return final_state

    def get_tokenized_history(self) -> torch.Tensor:
        """
        Convertit l'historique des actions de la main courante en une représentation tensorielle.
        
        Cette méthode crée un tenseur de dimensions (30, N) où :
        - 30 est le nombre de caractéristiques pour chaque action
        - N est le nombre d'actions dans l'historique
        
        Les 30 caractéristiques sont réparties comme suit :
        - [0:5]   : Phase du jeu (one-hot) [PREFLOP, FLOP, TURN, RIVER, SHOWDOWN]
        - [5:11]  : Position du joueur (one-hot) [SB, BB, UTG, HJ, CO, BTN]
        - [11:27] : Action prise (one-hot) [FOLD, CHECK, CALL, RAISE, ALL_IN, RAISE_25_POT, ..., RAISE_3X_POT]
        - [27:29] : Stack du joueur [avant l'action, après l'action] (normalisé par le stack initial)
        - [29]    : Variation du stack (différence entre avant et après l'action)

        Returns:
            torch.Tensor: Un tenseur de dimensions (30, N) si l'historique existe,
                         ou un tenseur de zéros de dimension (30,) si l'historique est vide.
                         N est le nombre d'actions dans l'historique.
        """

        history = self.get_hand_history()
        if not history:
            return torch.zeros(33)  # Retourne un tenseur vide si pas d'historique

        # 30 lignes : 5 (phase) + 6 (position) + 16 (action) + 2 (stack) + 2 (pot) + 1 (stack change)
        history_tensor = torch.zeros(33, len(history))

        # Mapping des phases pour one-hot encoding
        phase_map = {
            GamePhase.PREFLOP: 0,
            GamePhase.FLOP: 1, 
            GamePhase.TURN: 2,
            GamePhase.RIVER: 3,
            GamePhase.SHOWDOWN: 4
        }

        # Mapping des actions pour encoding
        action_map = {
            PlayerAction.FOLD: 0,
            PlayerAction.CHECK: 1,
            PlayerAction.CALL: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.ALL_IN: 4,
            PlayerAction.RAISE_25_POT: 5,
            PlayerAction.RAISE_33_POT: 6,
            PlayerAction.RAISE_50_POT: 7,
            PlayerAction.RAISE_66_POT: 8,
            PlayerAction.RAISE_75_POT: 9,
            PlayerAction.RAISE_100_POT: 10,
            PlayerAction.RAISE_125_POT: 11,
            PlayerAction.RAISE_150_POT: 12,
            PlayerAction.RAISE_175_POT: 13,
            PlayerAction.RAISE_2X_POT: 14,
            PlayerAction.RAISE_3X_POT: 15
        }

        for i, action in enumerate(history):
            # One-hot encoding pour la phase (5 dimensions, indices 0 à 4)
            phase_idx = phase_map[action.phase]
            history_tensor[phase_idx, i] = 1

            # One-hot encoding de la position du joueur (6 dimensions, indices 5 à 10)
            position_idx = action.position
            history_tensor[position_idx + 5, i] = 1

            # One-hot encoding pour l'action (16 dimensions, indices 11 à 26)
            action_idx = action_map[action.action_taken]
            history_tensor[action_idx + 11, i] = 1

            # Calcul du montant misé (1 dimension, indice 27)
            history_tensor[27, i] = action.bet_amount / self.starting_stack

            # Calcul du changement de stack (2 dimension, indice 28 à 29)
            stack_idx = action.stack_of_player_before_the_action / self.starting_stack
            stack_idx_after = action.stack_of_player_after_the_action / self.starting_stack
            history_tensor[28, i] = stack_idx
            history_tensor[27, i] = stack_idx_after

            # Calcul du changement de pot (2 dimension, indice 30 à 31)
            pot_idx = action.pot_before_the_action / self.starting_stack
            pot_idx_after = action.pot_after_the_action / self.starting_stack
            history_tensor[30, i] = pot_idx
            history_tensor[31, i] = pot_idx_after

            # Stack change (1 dimension, indice 32)
            stack_change = stack_idx_after - stack_idx
            history_tensor[32, i] = stack_change

        return history_tensor
        

    def step(self, action: PlayerAction) -> Tuple[List[float], float]:
        """
        Exécute une action et calcule la récompense associée.
        
        Args:
            action (PlayerAction): L'action à exécuter
            
        Returns:
            Tuple[List[float], float]: Nouvel état et récompense
        """
        current_player = self.players[self.current_player_seat]
        bet_amount = None
        reward = 0.0

        # Traiter l'action (met à jour l'état du jeu)
        action = self.process_action(current_player, action)
        next_state = self.get_state_hero()

        return next_state, reward

# ======================================================================
# Interface graphique et affichage
# ======================================================================

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
        name_text = self.font.render(f"{player.name} ({player.stack:.2f}BB)", True, name_color)
        self.screen.blit(name_text, (player.x - 50, player.y - 40))
        
        # Ajout de l'indicateur ALL-IN
        if player.is_all_in:
            # Créer une surface semi-transparente pour le fond
            allin_surface = pygame.Surface((100, 30))
            allin_surface.set_alpha(180)
            allin_surface.fill((200, 0, 0))  # Rouge foncé
            
            # Position de l'indicateur ALL-IN au-dessus des cartes
            allin_x = player.x - 25
            allin_y = player.y - 70
            
            # Dessiner le fond
            self.screen.blit(allin_surface, (allin_x, allin_y))
            
            # Dessiner le texte "ALL-IN" en blanc
            allin_text = self.font.render("ALL-IN", True, (255, 255, 255))
            self.screen.blit(allin_text, (allin_x + 10, allin_y + 2))
        
        # Draw player cards
        if player.is_active and not player.has_folded and len(player.cards) > 0:
            if player.show_cards or self.current_phase == GamePhase.SHOWDOWN:
                for i, card in enumerate(player.cards):
                    self._draw_card(card, player.x + i * 60, player.y)
            else:
                # Draw card backs for non-human players
                for i in range(2):
                    pygame.draw.rect(self.screen, (200, 0, 0), (player.x + i * 60, player.y, 50, 70))
        elif player.is_active and player.has_folded :
            for i in range(2): # Draw 2 back cards a bit transparent for folded players
                pygame.draw.rect(self.screen, (100, 100, 100), (player.x + i * 60, player.y, 50, 70))


        # Draw current bet with better formatting
        if player.current_player_bet > 0:
            bet_lines = []
            total_contribution = player.current_player_bet
            
            # Draw total bet with background
            total_text = f"Bet: {total_contribution:.2f}BB"
            bet_surface = pygame.Surface((150, 25 * (len(bet_lines) + 1)))
            bet_surface.set_alpha(128)
            bet_surface.fill((0, 0, 0))
            
            # Position the bet display
            bet_x = player.x - 30
            bet_y = player.y + 80
            self.screen.blit(bet_surface, (bet_x, bet_y))
            
            # Draw total first
            total_text_surface = self.font.render(total_text, True, (255, 255, 0))
            self.screen.blit(total_text_surface, (bet_x + 5, bet_y))
            
            # Draw contribution breakdown
            for i, line in enumerate(bet_lines, 1):
                text_surface = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(text_surface, (bet_x + 10, bet_y + i * 25))
    
    
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
        
        # Draw pots with better formatting and positioning
        total_pots = []
        
        # Add main pot if it exists
        if self.main_pot > 0:
            total_pots.append(("Phase Pot", self.main_pot))
        
        # Add side pots if they exist
        for i, pot in enumerate(self.side_pots):
            if pot.sum_of_contributions > 0:
                total_pots.append((f"Side Pot {i+1}", pot.sum_of_contributions))

        if total_pots:
            # Calculate center position and spacing
            center_x = SCREEN_WIDTH // 2
            start_y = 280  # Position above community cards
            pot_spacing = 30
            
            # Draw decorative pot icon and background
            for i, (pot_name, amount) in enumerate(total_pots):
                # Calculate position for this pot display
                y_pos = start_y + i * pot_spacing
                
                # Draw semi-transparent background
                pot_surface = pygame.Surface((300, 25))
                pot_surface.set_alpha(128)
                pot_surface.fill((50, 50, 50))
                pot_rect = pot_surface.get_rect(center=(center_x, y_pos))
                self.screen.blit(pot_surface, pot_rect)
                
                # Draw pot text with shadow for better visibility
                pot_text = f"{pot_name}: {amount:.2f}BB"
                # Shadow
                shadow_text = self.font.render(pot_text, True, (0, 0, 0))
                shadow_rect = shadow_text.get_rect(center=(center_x + 1, y_pos + 1))
                self.screen.blit(shadow_text, shadow_rect)
                # Main text
                text = self.font.render(pot_text, True, (255, 215, 0))  # Gold color
                text_rect = text.get_rect(center=(center_x, y_pos))
                self.screen.blit(text, text_rect)

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
        for button in self.action_buttons.values():
            button.draw(self.screen, self.font)
        
        # Draw bet slider with min and max values
        current_player = self.players[self.current_player_seat]
        min_raise = max(self.current_maximum_bet * 2, self.big_blind)
        max_raise = current_player.stack + current_player.current_player_bet
        
        pygame.draw.rect(self.screen, (200, 200, 200), self.pygame_bet_slider)
        bet_text = self.font.render(f"Bet: {int(self.pygame_slider_bet_amount)}BB", True, (255, 255, 255))
        min_text = self.font.render(f"Min: {min_raise}BB", True, (255, 255, 255))
        max_text = self.font.render(f"Max: {max_raise}BB", True, (255, 255, 255))
        
        self.screen.blit(bet_text, (50, SCREEN_HEIGHT - 75))
        self.screen.blit(min_text, (self.pygame_bet_slider.x, SCREEN_HEIGHT - 125))
        self.screen.blit(max_text, (self.pygame_bet_slider.x, SCREEN_HEIGHT - 150))
        
        # Draw action history in top right corner with better formatting
        history_x = SCREEN_WIDTH - 300
        history_y = 50
        history_text = self.font.render("Action History:", True, (255, 255, 255))
        self.screen.blit(history_text, (history_x, history_y - 30))
        
        y_offset = 0
        for player_name, actions in self.pygame_action_history.items():
            if actions:  # Only show players with actions
                # Draw player name
                player_text = self.font.render(f"{player_name}:", True, (255, 215, 0))  # Gold color
                self.screen.blit(player_text, (history_x, history_y + y_offset))
                y_offset += 25
                
                # Draw last 3 actions for this player
                for action in actions[-3:]:
                    action_text = self.font.render(f"  {action}", True, (255, 255, 255))
                    self.screen.blit(action_text, (history_x + 20, history_y + y_offset))
                    y_offset += 25
                y_offset += 5  # Add spacing between players

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
                # Après la durée d'affichage, commencer une nouvelle main
                self.pygame_winner_info = None
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
            
            # Vérifier les clics sur les boutons
            for action, button in self.action_buttons.items():
                if button.rect.collidepoint(mouse_pos) and button.enabled:
                    bet_amount = None
                    if action == PlayerAction.RAISE:
                        bet_amount = self.pygame_slider_bet_amount
                    elif action == PlayerAction.ALL_IN:
                        bet_amount = current_player.stack
                    # Suppression de l'ancienne logique qui remplaçait bet_amount
                    self.process_action(current_player, action, bet_amount)
            
            # Check bet slider
            if self.pygame_bet_slider.collidepoint(mouse_pos):
                # Calculate minimum raise (2x current bet)
                min_raise = max(self.current_maximum_bet * 2, self.big_blind * 2)
                # Calculate maximum raise (player's stack + current bet)
                max_raise = current_player.stack + current_player.current_player_bet
                
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
                    if event.key == pygame.K_c:
                        print(self.players)
                    if event.key == pygame.K_s:
                        state = self.get_state_hero()
                        print('\n=== État actuel du jeu ===')
                        
                        # Affichage complet du state pour vérifier toutes les informations (par ex. si les cartes sont suited ou non)
                        print("\n[DEBUG] Contenu complet du state:")
                        print(state.tolist())
                        
                        # 1. Cartes du joueur
                        print("\n1. Cartes du joueur:")
                        cards_suits = []
                        for i in range(2):
                            base_idx = i * 5
                            valeur = state[base_idx] * 14 + 2
                            suit_vector = state[base_idx+1:base_idx+5].tolist()
                            couleur_idx = suit_vector.index(1) if 1 in suit_vector else -1
                            couleur = ["♠", "♥", "♦", "♣"][couleur_idx] if couleur_idx != -1 else "?"
                            cards_suits.append(couleur)
                            print(f"Carte {i+1}: {int(valeur)}{couleur}")
                        
                        # Vérification de si les cartes sont suited
                        if len(cards_suits) == 2:
                            if cards_suits[0] == cards_suits[1]:
                                print("=> Les cartes du joueur sont SUITED.")
                            else:
                                print("=> Les cartes du joueur ne sont PAS suited.")
                        else:
                            print("=> Informations insuffisantes pour déterminer le suited.")
                        
                        # 2. Cartes communes
                        print("\n2. Cartes communes:")
                        for i in range(5):
                            base_idx = 10 + (i * 5)
                            if state[base_idx] != -1:
                                valeur = state[base_idx] * 14 + 2
                                suit_vector = state[base_idx+1:base_idx+5].tolist()
                                couleur_idx = suit_vector.index(1) if 1 in suit_vector else -1
                                couleur = ["♠", "♥", "♦", "♣"][couleur_idx] if couleur_idx != -1 else "?"
                                print(f"Carte {i+1}: {int(valeur)}{couleur}")
                        
                        # 3. Information sur la main
                        print("\n3. Information sur la main:")
                        hand_rank_idx = state[35:45].tolist().index(1) if 1 in state[35:45].tolist() else -1
                        print(f"Rang de la main: {HandRank(hand_rank_idx).name if hand_rank_idx != -1 else 'Inconnu'}")
                        print(f"Kicker: {int(state[45] * 13 + 2)}")
                        print(f"Rang normalisé: {state[46]:.2f}")
                        
                        # 4. Phase de jeu
                        print("\n4. Phase de jeu:")
                        phase_idx = state[47:52].tolist().index(1) if 1 in state[47:52].tolist() else -1
                        phases = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
                        print(f"Phase actuelle: {phases[phase_idx] if phase_idx != -1 else 'Inconnu'}")
                        
                        # 5-8. Informations sur les joueurs
                        print("\n5-8. Informations sur les joueurs:")
                        for i in range(6):
                            print(f"\nJoueur {i+1}:")
                            print(f"Stack: {state[53+i]:.2f}")
                            print(f"Mise actuelle: {state[59+i]:.2f}")
                            print(f"État: {'Actif' if state[65+i] == 1 else 'Inactif/Fold'}")
                        
                        # 9. Position relative
                        print("\n9. Position relative:")
                        pos_idx = state[71:77].tolist().index(1) if 1 in state[71:77].tolist() else -1
                        positions = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
                        print(f"Position: {positions[pos_idx] if pos_idx != -1 else 'Inconnu'}")
                        
                        # 10. Actions disponibles
                        print("\n10. Actions disponibles:")
                        actions = ["FOLD", "CHECK", "CALL", "RAISE", "ALL_IN"]
                        for i, action in enumerate(actions):
                            print(f"{action}: {'Disponible' if state[77+i] == 1 else 'Indisponible'}")
                        
                        # 11. Historique des actions
                        print("\n11. Historique des dernières actions:")
                        for i in range(6):
                            base_idx = 82 + (i * 5)
                            action_vector = state[base_idx:base_idx+5]
                            action_idx = action_vector.tolist().index(1) if 1 in action_vector.tolist() else -1
                            if action_idx != -1:
                                print(f"Joueur {i+1}: {actions[action_idx]}")
                            else:
                                print(f"Joueur {i+1}: Aucune action")
                        
                        # 12. Informations stratégiques
                        print("\n12. Informations stratégiques:")
                        print(f"Probabilité de victoire préflop: {state[112]:.3f}")
                        print(f"Cotes du pot: {state[113]:.3f}")
                        
                        print('\n=== Fin de l\'état ===\n')
                
                self.handle_input(event)
                
                # Update button hover states
                mouse_pos = pygame.mouse.get_pos()
                for button in self.action_buttons.values():
                    button.is_hovered = button.rect.collidepoint(mouse_pos)
            
            self._draw()
            pygame.display.flip()
        
        pygame.quit()

    def get_hand_history(self) -> List[ActionRecord]:
        """
        Retourne l'historique complet de la main courante.
        
        Returns:
            List[ActionRecord]: Liste des actions enregistrées pour la main courante
        """
        return self.current_hand_history

    def print_hand_history(self):
        """
        Affiche l'historique de la main courante de manière formatée.
        """
        print("\n=== Historique de la main courante ===")
        for i, record in enumerate(self.current_hand_history, 1):
            print(f"\nAction {i}:")
            print(f"Phase: {record.phase.value}")
            print(f"Joueur: {record.player}")
            print(f"Position: {record.position}")
            print(f"Action: {record.action_taken.value}")
            print(f"Montant misé: {record.bet_amount if record.bet_amount != -1 else 'FOLD'}BB")
            print(f"Stack avant l'action: {record.stack_of_player_before_the_action}BB")
            print(f"Stack après l'action: {record.stack_of_player_after_the_action}BB")
            print(f"Pot avant l'action: {record.pot_before_the_action}BB")
            print(f"Pot après l'action: {record.pot_after_the_action}BB")
        print("\n=====================================")

class HumanPlayer(Player):
    def __init__(self, agent, name, stack, seat_position):
        super().__init__(agent, name, stack, seat_position)

    def get_action(self, state):
        return self.agent.get_action(state)

if __name__ == "__main__":
    human_players_list = []
    for i in range(6):
        # Ici, le premier argument (agent) est None
        human_players_list.append(HumanPlayer(None, f"Player_{i}", 100, i))
    game = PokerGame(human_players_list)
    game.reset()
    game.manual_run()