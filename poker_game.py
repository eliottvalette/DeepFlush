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
import logging

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
        self.phase_pot = 0
        self.side_pots = self._create_side_pots() 

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
        
        self.number_raise_this_game_phase = 0 # Nombre de raises dans la phase courante (4 max, 4 inclus)

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
        self.phase_pot = 0
        self.side_pots = self._create_side_pots() 
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
            player.current_player_bet = 0
            player.total_bet = 0
            player.cards = []
            player.is_active = True
            player.has_folded = False
            player.is_all_in = False
            player.range = None        
        
        # Initialiser les variables d'état du jeu
        self.current_player_seat = (self.button_seat_position + 1) % 6  # 0-5 initialisé à SB
        self.current_maximum_bet = 0  # initialisé à 0 mais s'updatera automatiquement à BB

        self.number_raise_this_game_phase = 0

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
        self.phase_pot = 0
        self.side_pots = self._create_side_pots() 
        self.deck = self._create_deck()  # Réinitialiser le deck
        self.community_cards = []
        self.current_phase = GamePhase.PREFLOP

        # Réinitialiser l'état des joueurs
        for player in self.players:
            player.cards = []
            player.current_player_bet = 0
            player.total_bet = 0
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
            # Heads-Up : dans le heads-up, le joueur en bouton (qui est également SB) et l'autre (BB)
            ordered_players[0].role_position = 0  # Small Blind (et Dealer)
            ordered_players[1].role_position = 5  # Big Blind
        elif n == 3:
            # Dans le 3-handed, le joueur en bouton est SB.
            ordered_players[0].role_position = 5  # Button (et Small Blind)
            ordered_players[1].role_position = 0  # Small Blind (si nécessaire)
            ordered_players[2].role_position = 1  # Big Blind
        elif n == 4:
            ordered_players[0].role_position = 5  # Button
            ordered_players[1].role_position = 0  # Small Blind (SB)
            ordered_players[2].role_position = 1  # Big Blind (BB)
            ordered_players[3].role_position = 2  # UTG
        elif n == 5:
            ordered_players[0].role_position = 5  # Button
            ordered_players[1].role_position = 0  # Small Blind (SB)
            ordered_players[2].role_position = 1  # Big Blind (BB)
            ordered_players[3].role_position = 2  # UTG
            ordered_players[4].role_position = 3  # Cutoff (CO)
        elif n == 6:
            ordered_players[0].role_position = 5  # Button
            ordered_players[1].role_position = 0  # Small Blind (SB)
            ordered_players[2].role_position = 1  # Big Blind (BB)
            ordered_players[3].role_position = 2  # UTG
            ordered_players[4].role_position = 3  # Hijack (HJ)
            ordered_players[5].role_position = 4  # Cutoff (CO)

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
        self.pygame_action_history = []

        self._update_button_states()
        self.deal_small_and_big_blind()

        return self.get_state()

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
    
    def deal_small_and_big_blind(self):
        """
        Méthode à run en début de main pour distribuer automatiquement les blindes
        """
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
            raise ValueError("Impossible de déterminer la position de la Small Blind ou Big Blind")
        
        sb_seat_position = sb_player.seat_position
        bb_seat_position = bb_player.seat_position

        # SB
        if self.players[sb_seat_position].stack < self.small_blind:
            self.players[sb_seat_position].is_all_in = True
            self.players[sb_seat_position].current_player_bet = self.players[sb_seat_position].stack  # Le bet du joueur n'ayant pas assez pour payer la SB devient son stack
            self.phase_pot += self.players[sb_seat_position].stack  # Le pot est augmenté du stack du joueur
            self.players[sb_seat_position].total_bet = self.players[sb_seat_position].stack
            self.players[sb_seat_position].stack = 0  # Le stack du joueur est donc 0
            self.players[sb_seat_position].has_acted = True
        else:
            self.players[sb_seat_position].stack -= self.small_blind
            self.phase_pot += self.small_blind  # Le pot est augmenté de la SB
            self.players[sb_seat_position].total_bet = self.small_blind
            self.players[sb_seat_position].current_player_bet = self.small_blind
            self.players[sb_seat_position].has_acted = True

        self.current_maximum_bet = self.small_blind
        self._next_player()
        
        # BB
        if self.players[bb_seat_position].stack < self.big_blind:
            self.players[bb_seat_position].is_all_in = True
            self.players[bb_seat_position].current_player_bet = self.players[bb_seat_position].stack  # Le bet du joueur n'ayant pas assez pour payer la BB devient son stack
            self.phase_pot += self.players[bb_seat_position].stack  # Le pot est augmenté du stack du joueur
            self.players[bb_seat_position].total_bet = self.players[bb_seat_position].stack
            self.players[bb_seat_position].stack = 0  # Le stack du joueur devient 0
            self.players[bb_seat_position].has_acted = True
        else:
            self.players[bb_seat_position].stack -= self.big_blind
            self.phase_pot += self.big_blind  # Le pot est augmenté de la BB
            self.players[bb_seat_position].total_bet = self.big_blind
            self.players[bb_seat_position].current_player_bet = self.big_blind
            self.players[bb_seat_position].has_acted = True
        
        self.current_maximum_bet = self.big_blind
        self._next_player()

    def _next_player(self):
        """
        Passe au prochain joueur actif et n'ayant pas fold dans le sens horaire.
        Skip les joueurs all-in.
        """
        self.current_player_seat = (self.current_player_seat + 1) % self.num_players
        while (not self.players[self.current_player_seat].is_active or 
               self.players[self.current_player_seat].has_folded or
               self.players[self.current_player_seat].is_all_in):  # Added check for all-in
            self.current_player_seat = (self.current_player_seat + 1) % self.num_players
    
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
        if self.current_phase == GamePhase.PREFLOP and current_player.role_position == 0:
            self._next_player()
            return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        
        # Si un seul joueur reste, la partie est terminée, on déclenche le showdown en auto-win
        if len(in_game_players) == 1:
            print("Moving to showdown (only one player remains)")
            self.handle_showdown()
            return # Ne rien faire d'autre, la partie est terminée
        
        # Si tous les joueurs actifs sont all-in, la partie est terminée, on va vers le showdown pour déterminer le vainqueur
        elif (len(all_in_players) == len(in_game_players)) and (len(in_game_players) > 1):
            print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return # Ne rien faire d'autre, la partie est terminée
        
        for player in in_game_players:
            # Si le joueur n'a pas encore agi dans la phase, le tour n'est pas terminé
            if not player.has_acted:
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée
            # Si le joueur n'a pas égalisé la mise maximale et n'est pas all-in, le tour n'est pas terminé
            if player.current_player_bet < self.current_maximum_bet and not player.is_all_in:
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        
        # Atteindre cette partie du code signifie que la phase est terminée
        if self.current_phase == GamePhase.RIVER:
            print("River complete - going to showdown")
            self.handle_showdown()
        else:
            self.advance_phase()
            print(f"Advanced to {self.current_phase}")
            # Réinitialiser has_acted pour tous les joueurs actifs et non fold au début d'une nouvelle phase
            for p in self.players:
                if p.is_active and not p.has_folded:
                    p.has_acted = False

    def advance_phase(self):
        """
        Passe à la phase suivante du jeu (préflop -> flop -> turn -> river).
        Distribue les cartes communes appropriées et réinitialise les mises.
        """
        print(f"current_phase {self.current_phase}")

        # ---- SIDE POTS ----
        # Pour la répartition en side pots, on laisse tout le monde bet dans le phase_pot et on attend la fin de la phase.
        all_in_players = [p for p in self.players if p.is_all_in]
        in_game_players = [p for p in self.players if p.is_active and not p.has_folded]
        if len(all_in_players) > 1: 
            poorest_all_in_player = min(all_in_players, key=lambda x: x.current_player_bet)
            if poorest_all_in_player.current_player_bet < self.current_maximum_bet:
                self.side_pots = self._distribute_side_pots(in_game_players, self.side_pots, self.phase_pot)
        else:
            self.side_pots[0].sum_of_contributions = self.phase_pot
            self.side_pots[0].players = in_game_players
            self.side_pots[0].contributions_dict = {p: p.current_player_bet for p in in_game_players}
        
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
                if not player.has_folded:
                    player.has_acted = False  # Réinitialisation du flag
        
        # Set first player after dealer button
        self.current_player_seat = (self.button_seat_position + 1) % self.num_players
        while (not self.players[self.current_player_seat].is_active or self.players[self.current_player_seat].has_folded):
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
        if current_player.current_player_bet < self.current_maximum_bet: # Si le joueur n'a pas égalisé la mise maximale, il ne peut pas check
            self.action_buttons[PlayerAction.CHECK].enabled = False

        # ---- FOLD ----
        if self.action_buttons[PlayerAction.CHECK].enabled: # Si le joueur peut check, il ne peut pas fold
            self.action_buttons[PlayerAction.FOLD].enabled = False

        # ---- CALL ----
        # Désactiver call si pas de mise à suivre ou pas assez de jetons
        if current_player.current_player_bet == self.current_maximum_bet: # Si le joueur a égalisé la mise maximale, il ne peut pas call
            self.action_buttons[PlayerAction.CALL].enabled = False
        elif current_player.stack < (self.current_maximum_bet - current_player.current_player_bet): # Si le joueur n'a pas assez de jetons pour suivre la mise maximale, il ne peut pas call
            self.action_buttons[PlayerAction.CALL].enabled = False
            # Activer all-in si le joueur a des jetons mais pas assez pour call
            if current_player.stack > 0:
                self.action_buttons[PlayerAction.ALL_IN].enabled = True
            self.action_buttons[PlayerAction.RAISE].enabled = False

        # ---- RAISE ----
        # Désactiver raise si pas assez de jetons pour la mise minimale
        min_raise = (self.current_maximum_bet - current_player.current_player_bet) * 2 # La mise minimale est le double de la mise maximale ou du big blind
        if current_player.stack + current_player.current_player_bet < min_raise: # Si le joueur n'a pas assez de jetons pour la mise minimale, il ne peut pas raise
            self.action_buttons[PlayerAction.RAISE].enabled = False

        # Désactiver raise si déjà 4 relances dans le tour
        if self.number_raise_this_game_phase >= 4:
            self.action_buttons[PlayerAction.RAISE].enabled = False

        # ---- ALL-IN ----
        # All-in toujours disponible si le joueur a des jetons et n'a pas déjà égalisé la mise maximale
        self.action_buttons[PlayerAction.ALL_IN].enabled = (
            current_player.stack > 0 and 
            current_player.current_player_bet < self.current_maximum_bet
        )

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
            PlayerAction.FOLD: Button(300, SCREEN_HEIGHT - 100, 100, 40, "Fold", (200, 0, 0)),
            PlayerAction.CHECK: Button(450, SCREEN_HEIGHT - 100, 100, 40, "Check", (0, 200, 0)),
            PlayerAction.CALL: Button(600, SCREEN_HEIGHT - 100, 100, 40, "Call", (0, 0, 200)),
            PlayerAction.RAISE: Button(750, SCREEN_HEIGHT - 100, 100, 40, "Raise", (200, 200, 0)),
            PlayerAction.ALL_IN: Button(900, SCREEN_HEIGHT - 100, 100, 40, "All-in", (150, 0, 150))
        }
        return buttons

    def process_action(self, player: Player, action: PlayerAction, bet_amount: Optional[int] = None):
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
        print(f"Pot actuel : {self.phase_pot}BB")
        print(f"Mise maximale actuelle : {self.current_maximum_bet}BB")
        print(f"Stack du joueur avant action : {player.stack}BB")
        print(f"Mise actuelle du joueur : {player.current_player_bet}BB")

         #----- Traitement de l'action en fonction de son type -----
        if action == PlayerAction.FOLD:
            # Le joueur se couche il n'est plus actif pour ce tour.
            player.has_folded = True
            print(f"{player.name} se couche (Fold).")

        elif action == PlayerAction.CHECK:
            print(f"{player.name} check.")

        elif action == PlayerAction.CALL:
            print(f"{player.name} call.")
            call_amount = self.current_maximum_bet - player.current_player_bet
            if call_amount > player.stack: 
                raise ValueError(f"{player.name} n'a pas assez de jetons pour suivre la mise maximale, il n'aurait pas du avoir le droit de call")

            player.stack -= call_amount
            player.current_player_bet += call_amount
            self.phase_pot += call_amount
            player.total_bet += call_amount
            if player.stack == 0:
                player.is_all_in = True
            print(f"{player.name} a call {call_amount}BB")
        
        elif action == PlayerAction.RAISE:
            print(f"{player.name} raise.")
            min_raise = (self.current_maximum_bet - player.current_player_bet) * 2
            if bet_amount is None or bet_amount < min_raise or bet_amount > player.stack:
                raise ValueError(f"{player.name} n'a pas le droit de raise, mise minimale = {min_raise}, mise maximale = {player.stack}")
            player.stack -= bet_amount
            player.current_player_bet += bet_amount
            self.phase_pot += bet_amount
            player.total_bet += bet_amount
            self.current_maximum_bet = bet_amount
            self.number_raise_this_game_phase += 1

            print(f"{player.name} a raise {bet_amount}BB")

        elif action == PlayerAction.ALL_IN:
            print(f"{player.name} all-in.")
            if bet_amount is None or bet_amount != player.stack:
                raise ValueError(f"{player.name} n'a pas le droit de all-in, mise minimale = {player.stack}, mise maximale = {player.stack}")
            
            # Mise à jour de la mise maximale seulement si l'all-in est supérieur
            if bet_amount + player.current_player_bet > self.current_maximum_bet:
                self.current_maximum_bet = bet_amount + player.current_player_bet
                self.number_raise_this_game_phase += 1
            
            player.stack -= bet_amount
            player.current_player_bet += bet_amount
            self.phase_pot += bet_amount
            player.total_bet += bet_amount
            player.is_all_in = True
            print(f"{player.name} a all-in {bet_amount}BB")

        player.has_acted = True
        self.check_phase_completion()

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
        Gère la phase de showdown en tenant compte des side pots.
        """
        print("\n=== DÉBUT SHOWDOWN ===")
        self.current_phase = GamePhase.SHOWDOWN
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
        print("\nMains des joueurs:")
        for player in active_players:
            print(f"- {player.name}: {[str(card) for card in player.cards]}")
        
        # --- Distribution des gains ---
        # Si un seul joueur reste (tous les autres ont fold)
        if len(active_players) == 1:
            winner = active_players[0]
            total_pot = self.phase_pot + sum(pot.sum_of_contributions for pot in self.side_pots)
            print(f"\nVictoire par fold - {winner.name} gagne {total_pot:.2f}BB")
            print(f"- Main pot: {self.phase_pot:.2f}BB")
            for pot in self.side_pots:
                if pot.sum_of_contributions > 0:
                    print(f"- Side pot {pot.id}: {pot.sum_of_contributions:.2f}BB")
        
            winner.stack += total_pot
            self.pygame_winner_info = f"{winner.name} gagne {total_pot:.2f}BB (tous les autres joueurs ont fold)"
        else:
            print("\nCalcul des pots et des gagnants:")
            # Récupérer les contributions de chaque joueur
            contributions = {player: player.total_bet for player in self.players if player.total_bet > 0}
            print("\nContributions actuelles:")
            for player, amount in contributions.items():
                print(f"- {player.name}: {amount:.2f}BB")
        
            # Ajouter les contributions des side pots précédents
            print("\nAjout des contributions des side pots précédents:")
            for side_pot in self.side_pots:
                if side_pot.sum_of_contributions > 0:
                    print(f"\nSide pot {side_pot.id}:")
                    for player, contribution in side_pot.contributions_dict.items():
                        print(f"- {player.name}: +{contribution:.2f}BB")
                        if player in contributions:
                            contributions[player] += contribution
                        else:
                            contributions[player] = contribution
        
            print("\nContributions totales après fusion:")
            for player, amount in contributions.items():
                print(f"- {player.name}: {amount:.2f}BB")
        
            pots = []
            last = 0.0
            # Calculer les différents seuils de mises
            sorted_thresholds = sorted(set(contributions.values()))
            print(f"\nSeuils de mises identifiés: {[f'{x:.2f}BB' for x in sorted_thresholds]}")
            
            for i, threshold in enumerate(sorted_thresholds):
                print(f"\nTraitement du seuil {threshold:.2f}BB:")
                # Calculer le pot pour ce seuil
                count_all = sum(1 for bet in contributions.values() if bet >= threshold)
                pot_amount = (threshold - last) * count_all
                print(f"- Différence avec dernier seuil: {threshold - last:.2f}BB")
                print(f"- Nombre de contributeurs: {count_all}")
                print(f"- Montant du pot: {pot_amount:.2f}BB")
                
                # Identifier les joueurs éligibles
                eligible = [player for player in contributions if contributions[player] >= threshold and not player.has_folded]
                print(f"- Joueurs éligibles: {[p.name for p in eligible]}")
                
                pot_name = "Main Pot" if i == 0 else f"Side Pot {i}"
                pots.append({"name": pot_name, "amount": pot_amount, "eligible": eligible})
                last = threshold

            distribution_info = []
            print("\nDistribution des gains:")
            # Pour chaque pot, évaluer les mains
            for pot in pots:
                print(f"\nÉvaluation du {pot['name']} ({pot['amount']:.2f}BB):")
                if not pot["eligible"]:
                    print("Aucun joueur éligible pour ce pot")
                    continue
                
                best_eval = None
                winners = []
                print("Évaluation des mains:")
                for player in pot["eligible"]:
                    hand_eval = self.evaluate_final_hand(player)
                    # Formatage amélioré de l'affichage des mains
                    kickers_str = ""
                    if hand_eval[0] == HandRank.ROYAL_FLUSH:
                        hand_str = "Quinte Flush Royale"
                    elif hand_eval[0] == HandRank.STRAIGHT_FLUSH:
                        hand_str = f"Quinte Flush hauteur {hand_eval[1][0]}"
                    elif hand_eval[0] == HandRank.FOUR_OF_A_KIND:
                        hand_str = f"Carré de {hand_eval[1][0]} kicker {hand_eval[1][1]}"
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
                    else:  # HIGH_CARD
                        hand_str = f"Carte haute: {', '.join(str(x) for x in hand_eval[1])}"
                    
                    print(f"- {player.name}: {hand_str}")
                    current_key = (hand_eval[0].value, tuple(hand_eval[1]))
                    if best_eval is None or current_key > best_eval:
                        best_eval = current_key
                        winners = [player]
                    elif current_key == best_eval:
                        winners.append(player)
                
                share = pot["amount"] / len(winners) if winners else 0
                print(f"Gagnant(s): {[w.name for w in winners]}, {share:.2f}BB chacun")
                
                for winner in winners:
                    winner.stack += share
                winners_names = ", ".join([winner.name for winner in winners])
                distribution_info.append(f"{pot['name']} ({pot['amount']:.2f}BB): {winners_names} gagnent {share:.2f}BB chacun")
            
            # Enregistrer le résumé de la distribution pour l'affichage
            self.pygame_winner_info = "\n".join(distribution_info)
        
        print("\nStacks finaux:")
        for player in self.players:
            print(f"- {player.name}: {player.stack:.2f}BB")
            player.current_player_bet = 0

        # Reset pots
        self.phase_pot = 0
        self.side_pots = self._create_side_pots() 
        
        # Set the winner display start time
        self.pygame_winner_display_start = pygame.time.get_ticks()
        self.pygame_winner_display_duration = 2000  # 2 seconds in milliseconds
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

    def _distribute_side_pots(self, in_game_players: List[Player], side_pots: List[SidePot], phase_pot: float):
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
            state.append(player.stack / initial_stack) # (taille = 6)

        # 7. Informations sur les mises (normalisées par la grosse blinde)
        for player in self.players:
            state.append(player.current_player_bet / initial_stack) # (taille = 6)

        # 8. Informations sur l'activité (binaire extrême : actif/ruiné)
        for player in self.players:
            state.append(1 if player.is_active else -1) # (taille = 6)
        
        # 9. Informations sur l'activité in game (binaire extrême : en jeu/a foldé)
        for player in self.players:
            state.append(1 if player.has_folded else -1) # (taille = 6)

        # 10. Informations sur la position (encodage one-hot des positions relatives)
        relative_positions = [0.1] * self.num_players
        relative_pos = (self.current_player_seat - self.button_seat_position) % self.num_players
        relative_positions[relative_pos] = 1
        state.extend(relative_positions) # (taille = 6)

        # 11. Actions disponibles (binaire extrême : disponible/indisponible)
        action_availability = []
        for action in PlayerAction:
            if action in self.action_buttons and self.action_buttons[action].enabled:
                action_availability.append(1)
            else:
                action_availability.append(-1)
        state.extend(action_availability) # (taille = 6)

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
        for pygame_action_text in reversed(self.pygame_action_history[-self.num_players:]):
            if ":" in pygame_action_text:
                player_name, action = pygame_action_text.split(":")
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
        call_amount = self.current_maximum_bet - current_player.current_player_bet
        pot_odds = call_amount / (self.phase_pot + call_amount) if (self.phase_pot + call_amount) > 0 else 0
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
        bet_amount = None
        reward = 0.0

        # Capturer l'état du jeu avant de traiter l'action pour le calcul des cotes du pot
        call_amount_before = self.current_maximum_bet - current_player.current_player_bet
        pot_before = self.phase_pot

        # --- Récompenses stratégiques des actions ---
        # Récompense basée sur l'action par rapport à la force de la main
        hand_strength = self._evaluate_hand_strength(current_player)
        pot_potential = self.phase_pot / (self.big_blind * 100)
    
        if action == PlayerAction.RAISE:
            reward += 0.2 * hand_strength  # Ajuster la récompense selon la force de la main
            if hand_strength > 0.7:
                reward += 0.5 * pot_potential  # Bonus pour jeu agressif avec main forte
            bet_amount = self.current_maximum_bet * 2
                
        elif action == PlayerAction.ALL_IN:
            reward += 0.3 * hand_strength
            if hand_strength > 0.8:
                reward += 1.0 * pot_potential
            bet_amount = current_player.stack
                
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
        self.process_action(current_player, action, bet_amount)

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
        total_pot = self.phase_pot + sum(p.current_player_bet for p in self.players)
        call_amount = self.current_maximum_bet - player.current_player_bet
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


        # Draw current bet with better formatting
        if player.current_player_bet > 0:
            bet_lines = []
            total_contribution = player.current_player_bet
            
            # Calculate bet contributions
            main_pot_contrib = min(total_contribution, self.phase_pot) if self.phase_pot > 0 else total_contribution
            if main_pot_contrib > 0:
                bet_lines.append(f"Main: {main_pot_contrib:.2f}BB")
            
            remaining_contrib = total_contribution - main_pot_contrib
            for i, side_pot in enumerate(self.side_pots):
                if side_pot.sum_of_contributions > 0:  # Correction ici
                    pot_contrib = min(remaining_contrib, side_pot.sum_of_contributions)
                    if pot_contrib > 0:
                        bet_lines.append(f"Side {i+1}: {pot_contrib:.2f}BB")
                    remaining_contrib -= pot_contrib
            
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
        if self.phase_pot > 0:
            total_pots.append(("Main Pot", self.phase_pot))
        
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
        min_raise = max(self.current_maximum_bet * 2, self.big_blind * 2)
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
                    # RAISE : utiliser la formule correcte pour la mise minimale
                    if action == PlayerAction.RAISE:
                        max_bet = current_player.stack + current_player.current_player_bet
                        # Utiliser la même logique que dans process_action :
                        min_bet = (self.current_maximum_bet - current_player.current_player_bet) * 2
                        bet_amount = min(bet_amount, max_bet)
                        bet_amount = max(bet_amount, min_bet)
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
                for button in self.action_buttons.values():
                    button.is_hovered = button.rect.collidepoint(mouse_pos)
            
            self._draw()
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    list_names = ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5", "Player_6"]
    game = PokerGame(list_names)
    game.manual_run()
