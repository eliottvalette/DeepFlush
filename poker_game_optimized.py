# poker_game.py
"""
Texas Hold'em, No Limit, 6 max.
"""
import pygame
import random as rd
from typing import List, Dict, Optional, Tuple
from utils.config import DEBUG
from collections import Counter
import pygame.font
import numpy as np
from poker_game import Player, PlayerAction, GamePhase, Card, SidePot, HandRank, Button, SCREEN_HEIGHT

class PokerGameOptimized:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
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

        self.start_new_hand()
        self._update_button_states()
        
        return self.get_state()

    def start_new_hand(self):
        """
        Distribue une nouvelle main sans réinitialiser la partie.
        
        Returns:
            L'état initial du jeu après distribution.
        """
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

        self._update_button_states()
        self.deal_small_and_big_blind()

        # Clear action history for new hand
        self.pygame_action_history = {player.name: [] for player in self.players}

        # In poker_game.py, inside start_new_hand() after resetting players:
        if DEBUG:
            print("Player positions and roles for the new hand:")
            for p in self.players:
                if p.is_active:
                    print(f"{p.name}: seat={p.seat_position}, role={p.role_position}, stack={p.stack}")
                else :
                    print(f"{p.name}: seat={p.seat_position}, role=(inactive), stack={p.stack}")

        return self.get_state()

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
            if DEBUG:
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
            if DEBUG:
                print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            self.handle_showdown()
            return # Ne rien faire d'autre, la partie est terminée
        
        for player in in_game_players:
            # Si le joueur n'a pas encore agi dans la phase, le tour n'est pas terminé
            if not player.has_acted:
                if DEBUG:
                    print(f'{player.name} n\'a pas encore agi')
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée

            # Si le joueur n'a pas égalisé la mise maximale et n'est pas all-in, le tour n'est pas terminé
            if player.current_player_bet < self.current_maximum_bet and not player.is_all_in:
                if DEBUG:
                    print('Un des joueurs en jeu n\'a pas égalisé la mise maximale')
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        
        # Atteindre cette partie du code signifie que la phase est terminée
        if self.current_phase == GamePhase.RIVER:
            if DEBUG:
                print("River complete - going to showdown")
            self.handle_showdown()
            return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        else:
            self.advance_phase()
            if DEBUG:
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
                    if DEBUG:
                        print("River complete - going to showdown")
                    self.handle_showdown()
                else:
                    self.advance_phase()
                    if DEBUG:
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
            if DEBUG:
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
        if DEBUG:
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
        
        # Set first player after dealer button # TODO : Quand la Turn se termine sur le joueur 3 on donne la parole au premier a sa gauche et au lieu de le donner au premier actif a la gauche de la sb, sb inclue
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
        if DEBUG:
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
            if DEBUG : 
                print(f"{player.name} se couche (Fold).")
        
        elif action == PlayerAction.CHECK:
            if DEBUG : 
                print(f"{player.name} check.")
        
        elif action == PlayerAction.CALL:
            if DEBUG : 
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
            if DEBUG : 
                print(f"{player.name} a call {call_amount}BB")

        elif action == PlayerAction.RAISE:
            if DEBUG : 
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
        
            if DEBUG : 
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
        
            if DEBUG : 
                print(f"{player.name} a raise (pot-based {percentage*100:.0f}%) à {bet_amount}BB")
        
        
        elif action == PlayerAction.ALL_IN:
            if DEBUG : 
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
            if DEBUG : 
                print(f"{player.name} a all-in {bet_amount}BB")
        
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
            call_amount = self.current_maximum_bet - player.current_player_bet
            action_text += f" {call_amount}BB"
        
        # Add action to player's history
        self.pygame_action_history[player.name].append(action_text)
        # Keep only last 5 actions per player
        if len(self.pygame_action_history[player.name]) > 5:
            self.pygame_action_history[player.name].pop(0)
        
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
        if DEBUG:
            print("\n=== DÉBUT SHOWDOWN ===")
        self.current_phase = GamePhase.SHOWDOWN
        # On considère ici TOUS les joueurs qui ont contribué (même s'ils ont foldé)
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        if DEBUG:
            print(f"Joueurs actifs au showdown: {[p.name for p in active_players]}")
        
        # Désactiver tous les boutons pendant le showdown
        for button in self.action_buttons.values():
            button.enabled = False
        
        # S'assurer que toutes les cartes communes sont distribuées
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        if DEBUG:
            print(f"Cartes communes finales: {[str(card) for card in self.community_cards]}")
        
        # Afficher les mains des joueurs actifs
        if DEBUG:
            print("\nMains des joueurs actifs:")
            for player in active_players:
                print(f"- {player.name}: {[str(card) for card in player.cards]}")
        
        # --- Distribution des gains ---
        # Cas particulier : victoire par fold (il ne reste qu'un joueur actif)
        if len(active_players) == 1:
            winner = active_players[0]
            contributions = {player: player.total_bet for player in self.players if player.total_bet > 0}
            total_pot = sum(contributions.values())
            if DEBUG:
                print(f"\nVictoire par fold - {winner.name} gagne {total_pot:.2f}BB")
            winner.stack += total_pot
            self.pygame_winner_info = f"{winner.name} gagne {total_pot:.2f}BB (tous les autres joueurs ont fold)"
        else:
            # --- Répartition des contributions en pots (Main Pot et Side Pots) ---
            # IMPORTANT : On prend en compte tous les joueurs qui ont misé,
            # même s'ils se sont couchés.
            contributions = {player: player.total_bet for player in self.players if player.total_bet > 0}
            
            if DEBUG:
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
                
                if DEBUG:
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
            if DEBUG: 
                print("\nDistribution des gains par pot:")
            for pot in pots:
                if DEBUG:
                    print(f"\nÉvaluation du {pot['name']} ({pot['amount']:.2f}BB):")
                if not pot["eligible"]:
                    if DEBUG:
                        print("Aucun joueur éligible pour ce pot.")
                    continue
                
                best_eval = None
                winners = []
                if DEBUG:
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
                        hand_str = f"Carzzte haute: {', '.join(str(x) for x in hand_eval[1])}"
        
                    if DEBUG:
                        print(f"- {player.name}: {hand_str}")
                    current_key = (hand_eval[0].value, tuple(hand_eval[1]))
                    if best_eval is None or current_key > best_eval:
                        best_eval = current_key
                        winners = [player]
                    elif current_key == best_eval:
                        winners.append(player)
        
                if winners:
                    share = pot["amount"] / len(winners)
                    if DEBUG:
                        print(f"Gagnant(s): {[w.name for w in winners]}, {share:.2f}BB chacun")
                    for winner in winners:
                        winner.stack += share
                else:
                    share = 0
                    if DEBUG:
                        print("Aucun gagnant déterminé pour ce pot.")
        
            # (Optionnel) Mettre à jour l'info affichée par Pygame
            self.pygame_winner_info = "\n".join(
                [f"{pot['name']} ({pot['amount']:.2f}BB): {[p.name for p in pot['eligible'] if not p.has_folded]} gagnent {pot['amount'] / len([p for p in pot['eligible'] if not p.has_folded]):.2f}BB"
                for pot in pots if len([p for p in pot['eligible'] if not p.has_folded]) > 0]
            )
        
            if DEBUG:
                print("\nStacks initiaux:")
                for player in self.players:
                    print(f"- {player.name}: {self.initial_stacks[player.name]}BB")
        
            if DEBUG:
                print("\nStacks finaux:")
                for player in self.players:
                    print(f"- {player.name}: {player.stack:.2f}BB")
        if DEBUG:
            print("\nVariations :")
        for player in self.players:
            initial = self.initial_stacks.get(player.name, 0)
            variation = player.stack - initial
            signe = "+" if variation >= 0 else ""
            if DEBUG :
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
        if DEBUG:
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

    # --------------------------------
    # Methodes Nécessaires pour le RL
    # --------------------------------
    def get_state(self):
        """
        Obtient l'état actuel du jeu sous forme d'un vecteur d'état pour l'apprentissage par renforcement.
        
        Le vecteur d'état est composé des sous-vecteurs suivants :

        1. Cartes du joueur (2 cartes × (4 + 1) + 1 = 11 dimensions) :
            - Pour chaque carte :
                - Valeur normalisée (1 dimension)
                - Couleur one-hot (4 dimensions)
        
        2. Cartes communes (5 cartes × (4 + 1) + 1 = 26 dimensions) :
            - Pour chaque carte :
                - Valeur normalisée (1 dimension)
                - Couleur one-hot (4 dimensions)
            - -1 pour les cartes non distribuées
        
        3. Information sur la main (12 dimensions) :
            - Rang de la main one-hot (10 dimensions)
            - Valeur du kicker normalisée (1 dimension)
            - Rang normalisé (1 dimension)
        
        4. Phase de jeu (5 dimensions) :
            - One-hot encoding : [preflop, flop, turn, river, showdown]
        
        5. Information sur les mises (1 dimension) :
            - Mise maximale actuelle normalisée
        
        6. Stacks des joueurs (6 dimensions) :
            - Stack normalisé pour chaque joueur
        
        7. Mises actuelles (6 dimensions) :
            - Mise actuelle normalisée pour chaque joueur
        
        8. État des joueurs (6 dimensions) :
            - 1 si actif et non foldé, -1 sinon
        
        9. Positions relatives (6 dimensions) :
            - One-hot encoding de la position relative au bouton
        
        10. Actions disponibles (5 dimensions) :
            - Disponibilité de chaque action [fold, check, call, raise, all-in]
        
        11. Historique des actions (30 dimensions) :
            - Pour chaque joueur (6) :
                - One-hot encoding de la dernière action (5 dimensions)
        
        12. Informations stratégiques (2 dimensions) :
            - Probabilité de victoire préflop
            - Cotes du pot

        13. Potentiel de quinte et de couleur (2 dimensions) :
            - Potentiel de quinte
            - Potentiel de couleur
        
        Returns:
            numpy.ndarray: Vecteur d'état de dimension 106, normalisé entre -1 et 1
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
        # Cartes du joueur (2 cartes )
        for card in current_player.cards:
            state.append((card.value - 2) / 14)  # Extension pour la valeur
            suit_range = [-1] * 4
            suit_range[suit_map[card.suit]] = 1
            state.extend(suit_range)  # Extension pour la couleur
        
        # Ajout de remplissage pour les cartes manquantes du joueur
        remaining_player_cards = 2 - len(current_player.cards)
        for _ in range(remaining_player_cards):
            state.append(-1) # hauteur manquante
            state.extend([-1] * 4) # couleur manquante
        
        # Cartes communes
        for i, card in enumerate(self.community_cards):
            state.append((card.value - 2) / 14)  # Extension pour la valeur
            suit_range = [-1] * 4
            suit_range[suit_map[card.suit]] = 1
            state.extend(suit_range)  # Extension
        
        # Ajout de remplissage pour les cartes communes manquantes
        remaining_community_cards = 5 - len(self.community_cards)
        for _ in range(remaining_community_cards):
            state.append(-1)         # hauteur manquante
            state.extend([-1] * 4)   # couleur manquante
        
        # 2. Rang de la main actuelle (si assez de cartes sont visibles)
        kicker_idx_map = {
            HandRank.HIGH_CARD: 0,
            HandRank.PAIR: 1,
            HandRank.TWO_PAIR: 2,
            HandRank.THREE_OF_A_KIND: 1,
            HandRank.STRAIGHT: 0,
            HandRank.FLUSH: 0,
            HandRank.FULL_HOUSE: 0,
            HandRank.FOUR_OF_A_KIND: 1,
            HandRank.STRAIGHT_FLUSH: 0,
            HandRank.ROYAL_FLUSH: 0
        }
        if len(current_player.cards) + len(self.community_cards) >= 5:
            hand_rank, kickers = self.evaluate_final_hand(current_player)
            hand_rank_range = [-1] * len(HandRank)
            hand_rank_range[hand_rank.value] = 1
            kicker_idx = kicker_idx_map[hand_rank]
            state.extend(hand_rank_range)                  # Tokénisation du rang
            state.append((kickers[kicker_idx]- 2) / 13)    # Normalisation de la valeur du kicker (taille = 1)    
            state.append(hand_rank.value / len(HandRank))  # Normalisation de la valeur du rang (taille = 1)
        else:
            hand_rank, kickers = self.evaluate_current_hand(current_player)
            hand_rank_range = [-1] * len(HandRank)
            hand_rank_range[hand_rank.value] = 1
            state.extend(hand_rank_range)                  # Tokénisation du rang
            state.append((kickers[0]- 2) / 13)             # Normalisation de la valeur du kicker
            state.append(hand_rank.value / len(HandRank))  # Normalisation de la valeur du rang (taille = 1)

        # 3. Informations sur le tour
        phase_values = {
            GamePhase.PREFLOP: 0,
            GamePhase.FLOP: 1,
            GamePhase.TURN: 2,
            GamePhase.RIVER: 3,
            GamePhase.SHOWDOWN: 4
        }

        phase_range = [-1] * 5
        phase_range[phase_values[self.current_phase]] = 1
        state.extend(phase_range)

        # 4. Mise actuelle normalisée par le stack initial
        state.append(self.current_maximum_bet / self.starting_stack)  # Normalisation de la mise (taille = 1)

        # 5. Argent restant (tailles des stacks normalisées par le stack initial)
        for player in self.players:
            state.append(player.stack / self.starting_stack) # (taille = 6)

        # 6. Informations sur les mises (normalisées par le stack initial)
        for player in self.players:
            state.append(player.total_bet / self.starting_stack) # (taille = 6)

        # 7. Informations sur l'activité (fold ou inactif)
        for player in self.players:
            state.append(1 if ((not player.has_folded) and player.is_active) else -1) # (taille = 6)

        # 8. Informations sur la position (encodage one-hot des positions relatives)
        relative_positions = [-1] * self.num_players
        relative_pos = (self.current_player_seat - self.button_seat_position) % self.num_players
        relative_positions[relative_pos] = 1
        state.extend(relative_positions) # (taille = 6)

        # 9. Actions disponibles (binaire extrême : disponible/indisponible)
        action_availability = []
        for action in PlayerAction:
            if action in self.action_buttons and self.action_buttons[action].enabled:
                action_availability.append(1)
            else:
                action_availability.append(-1)
        state.extend(action_availability) # (taille = 6)

        # 10. Update action encoding for previous actions
        action_encoding = {
            None:     [-1, -1, -1, -1, -1],  # Default encoding for no action
            "fold":   [1, 0, 0, 0, 0],
            "check":  [0, 1, 0, 0, 0],
            "call":   [0, 0, 1, 0, 0],
            "raise":  [0, 0, 0, 1, 0],
            "all-in": [0, 0, 0, 0, 1]
        }

        # 11. Obtenir la dernière action de chaque joueur, ordonnée relativement au joueur actuel
        last_actions = []
        # Commencer par le joueur actuel et parcourir les positions dans l'ordre relatif
        for i in range(self.num_players):
            relative_seat = (self.current_player_seat + i) % self.num_players
            player = self.players[relative_seat]
            player_actions = self.pygame_action_history[player.name]
            
            if player_actions:
                # Extraction du type d'action de la dernière action
                last_action = player_actions[-1].split()[0]  # Obtenir juste le type d'action
                last_actions.append(action_encoding.get(last_action, action_encoding[None]))
            else:
                last_actions.append(action_encoding[None])
        
        # 12. Ajout des actions aplaties à l'état
        state.extend([val for sublist in last_actions for val in sublist])

        # 13 Probabilité de victoire de la main au préflop
        hand_win_prob = self._evaluate_preflop_strength(current_player.cards)
        state.append(hand_win_prob)

        # 14. Cotes du pot
        call_amount = self.current_maximum_bet - current_player.current_player_bet
        pot_odds = call_amount / (self.main_pot + call_amount) if (self.main_pot + call_amount) > 0 else 0
        state.append(pot_odds) # (taille = 1)

        # 15. Potentiel de quinte et de couleur
        straight_draw, flush_draw = self.compute_hand_draw_potential(current_player)
        state.append(straight_draw)
        state.append(flush_draw)

        # Avant de retourner, conversion en tableau numpy.
        state = np.array(state, dtype=np.float32)
        return state
    
    def get_final_state(self, previous_state, final_stacks):
        """
        Reconstruit l'Etat final de la main en fonction de l'état précédent et du stack final.
        Cette fonction est utilisée pour reconstruire l'état final de la main lorsque le showdown est déclenché car une fois
        déclenché, l'état est réinitialisé le get_state appelle l'état de la main suivante.

        1. Cartes du joueur (inchangées)
        
        2. Cartes communes (inchangées)
        
        3. Information sur la main (inchangées)
        
        4. Phase de jeu (changer en one-hot showdown)
        
        5. Information sur les mises (passer la mise max actuelle à 0)
        
        6. Stacks des joueurs (recupérer les stacks finaux dans le dict final_stack)
        
        7. Mises actuelles (changer en 0 pour tous)
        
        8. État des joueurs (changer en 0 pour tous)
        
        9. Positions relatives (inchangée)
        
        10. Actions disponibles (toutes les actions sont indisponibles)
        
        11. Historique des actions (changer en -1 pour tous)
        
        12. Informations stratégiques (changer en 0 pour tous)
        
        13. Potentiel de quinte (changer en 0 pour tous)

        14. Potentiel de couleur (changer en 0 pour tous)

        Args:
            previous_state (numpy.ndarray): Etat précédent
            final_stack (float): Stack final
        
        Returns:
            numpy.ndarray: Vecteur d'état final de dimension 106, normalisé entre -1 et 1
        """

        # Créer une copie modifiable du previous_state
        final_state = previous_state.copy()
        
        # 1-3. Garder les cartes du joueur, cartes communes et info de la main inchangées
        # (indices 0-46 restent les mêmes)
        
        # 4. Mettre à jour la phase de jeu en SHOWDOWN
        final_state[47:52] = np.array([-1, -1, -1, -1, 1])  # One-hot encoding pour SHOWDOWN
        
        # 5. Mettre la mise maximale actuelle à 0
        final_state[52] = 0
        
        # 6. Mettre à jour les stacks des joueurs avec les stacks finaux
        for i, player in enumerate(self.players):
            final_state[53 + i] = final_stacks[player.name] / self.starting_stack
        
        # 7. Mettre les mises actuelles à 0
        final_state[59:65] = 0
        
        # 8. Mettre l'état des joueurs à inactif (-1)
        final_state[65:71] = -1
        
        # 9. Garder les positions relatives inchangées
        # (indices 71-76 restent les mêmes)
        
        # 10. Désactiver toutes les actions disponibles
        final_state[77:82] = -1
        
        # 11. Réinitialiser l'historique des actions
        for i in range(6):  # Pour chaque joueur
            final_state[82+i*5:82+(i+1)*5] = -1
        
        # 12. Réinitialiser les informations stratégiques
        final_state[112] = 0  # Probabilité de victoire
        final_state[113] = 0  # Cotes du pot
        
        # 13-15. Réinitialiser les potentiels de tirage
        final_state[114:] = 0

        # Avant de retourner, conversion en tableau numpy
        final_state = np.array(final_state, dtype=np.float32)
        
        return final_state    
    
    def get_simple_state(self) -> Tuple[List, int]:
        """
        Extrait un état de jeu simplifié sous forme de vecteur plat contenant les informations brutes nécessaires.
        Format du vecteur:
        [
            hero_name_idx,                # index du nom du héros (0-5)
            hero_card1_value,             # valeur de la première carte (2-14)
            hero_card1_suit_idx,          # index de la couleur (0-3)
            hero_card2_value,             # valeur de la deuxième carte
            hero_card2_suit_idx,          # index de la couleur
            *community_cards_flat,        # 10 valeurs (5 cartes x [valeur, suit_idx])
            phase_idx,                    # index de la phase (0-4)
            pot,                          # valeur du pot
            current_maximum_bet,          # mise maximale actuelle
            big_blind,                    # valeur de la grosse blinde
            small_blind,                  # valeur de la petite blinde
            *players_info_flat           # Pour chaque joueur actif: [name_idx, stack, bet, active, folded, all_in, role_pos, acted, seat_pos]
        ]

        Returns:
            Tuple[List, int]: Vecteur d'état plat et nombre de joueurs actifs
        """
        current_player = self.players[self.current_player_seat]
        
        # Conversion des cartes en valeurs numériques
        hero_cards = [(card.value, ["♠", "♥", "♦", "♣"].index(card.suit)) for card in current_player.cards]
        community_cards = [(card.value, ["♠", "♥", "♦", "♣"].index(card.suit)) for card in self.community_cards]
        
        # Padding des cartes communes si nécessaire
        while len(community_cards) < 5:
            community_cards.append((-1, -1))  # -1 indique une carte non distribuée
        
        # Récupérer les joueurs qui peuvent encore agir
        player_that_can_still_act = [p for p in self.players if not (p.has_folded or not p.is_active)]
        
        # Construction du vecteur plat
        flat_state = [
            # Info du héros
            self.players.index(current_player),  # hero_name_idx
            
            # Cartes du héros
            hero_cards[0][0], hero_cards[0][1],  # card1 value, suit
            hero_cards[1][0], hero_cards[1][1],  # card2 value, suit
            
            # Cartes communes (aplatir la liste)
            *[val for card in community_cards for val in card],  # 10 valeurs
            
            # Phase de jeu
            self.current_phase.value,
            
            # Informations générales
            self.main_pot,
            self.current_maximum_bet,
            self.big_blind,
            self.small_blind
        ]
        
        # Informations des joueurs actifs
        for p in player_that_can_still_act:
            player_info = [
                self.players.index(p),  # name_idx
                p.stack,
                p.current_player_bet,
                1 if p.is_active else 0,
                1 if p.has_folded else 0,
                1 if p.is_all_in else 0,
                p.role_position,
                1 if p.has_acted else 0,
                p.seat_position
            ]
            flat_state.extend(player_info)
        
        return flat_state, len(player_that_can_still_act)


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

        # Traiter l'action (met à jour l'état du jeu)
        action = self.process_action(current_player, action, bet_amount)
        next_state = self.get_state()

        return next_state

# ======================================================================
# Methodes de calculs (à exporter dans un autre fichier)
# ======================================================================

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
                # Modification ici : gérer le cas où il n'y a pas de kicker disponible
                remaining_values = [v for v in values if v not in pairs[:2]]
                kicker = max(remaining_values) if remaining_values else 0
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
    
    def _evaluate_preflop_strength(self, cards : Tuple[Card, Card]) -> float:
        """
        Évalue la force d'une main preflop selon les calculs de l'algo de Monte Carlo.

        quartiles : [0, 0.25, 0.5, 0.75, 1] => [0.086 0.138 0.167 0.206 0.493]
        """
        hard_code_win_prob = [
            [0.493, 0.311, 0.295, 0.279, 0.268, 0.240, 0.232, 0.224, 0.216, 0.221, 0.217, 0.212, 0.206],
            [0.279, 0.432, 0.283, 0.269, 0.257, 0.234, 0.214, 0.205, 0.199, 0.197, 0.190, 0.184, 0.183],
            [0.258, 0.250, 0.378, 0.263, 0.251, 0.226, 0.208, 0.192, 0.186, 0.181, 0.175, 0.171, 0.167],
            [0.244, 0.235, 0.229, 0.334, 0.248, 0.225, 0.206, 0.187, 0.172, 0.169, 0.163, 0.160, 0.157],
            [0.231, 0.223, 0.217, 0.215, 0.299, 0.224, 0.205, 0.189, 0.174, 0.161, 0.157, 0.152, 0.148],
            [0.202, 0.194, 0.189, 0.187, 0.189, 0.266, 0.203, 0.189, 0.173, 0.158, 0.145, 0.143, 0.140],
            [0.192, 0.174, 0.168, 0.166, 0.169, 0.167, 0.241, 0.191, 0.177, 0.162, 0.148, 0.135, 0.134],
            [0.183, 0.167, 0.150, 0.149, 0.151, 0.150, 0.153, 0.217, 0.178, 0.166, 0.153, 0.140, 0.128],
            [0.174, 0.158, 0.144, 0.131, 0.133, 0.134, 0.138, 0.142, 0.200, 0.171, 0.158, 0.145, 0.132],
            [0.180, 0.152, 0.139, 0.127, 0.119, 0.119, 0.124, 0.129, 0.132, 0.186, 0.165, 0.154, 0.141],
            [0.175, 0.149, 0.133, 0.123, 0.115, 0.105, 0.109, 0.114, 0.120, 0.127, 0.172, 0.147, 0.135],
            [0.169, 0.143, 0.129, 0.119, 0.111, 0.101, 0.096, 0.099, 0.105, 0.114, 0.108, 0.163, 0.132],
            [0.163, 0.138, 0.124, 0.113, 0.106, 0.097, 0.092, 0.086, 0.091, 0.100, 0.095, 0.089, 0.155]
        ]

        first_card, second_card = cards

        # Conversion des valeurs en indices (pour obtenir des indices de 0 à 12)
        # Ici, l'As (14) devien 0, donc on fait: index = 14 - valeur
        i1 = 14 - first_card.value
        i2 = 14 - second_card.value

        if first_card.suit == second_card.suit:
            # Pour les mains suited, la matrice attend que la carte de valeur la plus haute (donc ici l'indice le plus petit)
            # soit en ligne (premier indice) et la plus faible en colonne (deuxième indice).
            if i1 > i2:
                i1, i2 = i2, i1  # on échange pour que i1 soit toujours le plus petit
            hand_win_prob = hard_code_win_prob[i1][i2]
        else:
            # Pour offsuit, la convention est d'utiliser l'élément situé en dessous de la diagonale :
            # donc on range les indices de manière décroissante.
            if i1 < i2:
                hand_win_prob = hard_code_win_prob[i2][i1]
            else:
                hand_win_prob = hard_code_win_prob[i1][i2]

        return hand_win_prob
        
    
    def compute_hand_draw_potential(self, player) -> float:
        """
        Calcule un heuristique simple du potentiel d'amélioration (draw potential)
        à partir des cartes du joueur et les community cards.
        
        Pour un tirage quinte :
          - Si vous avez 4 cartes consécutives, on donne par exemple 0.8 (très fort draw)
          - 3 cartes consécutives donnent 0.4, 2 cartes seulement 0.1
          
        Pour un tirage couleur :
          - Si vous avez 4 cartes d'une même couleur, cela vaut 0.8, 3 cartes 0.4, 2 cartes 0.1.
        
        On retourne ici la valeur maximale (parmi les deux) comme indicateur.
        Vous pouvez bien sûr ajuster ou combiner différemment ces deux critères.
        """
        # Combine les cartes du joueur et les community cards
        cards = player.cards + self.community_cards

        # Si on est au pré-flop, renvoyer 0.0 car on ne peut pas les informations de force de main preflop sont données dans d'autres states
        if len(cards) <= 2:
            return 0.0, 0.0

        # --- Potentiel pour la quinte (straight draw) ---
        # Récupérer l'ensemble des valeurs uniques et les trier
        values = sorted(set(card.value for card in cards))
        max_run = 1
        current_run = 1
        for i in range(1, len(values)):
            if values[i] == values[i-1] + 1: # Si la valeur actuelle est égale à la valeur précédente + 1, on incrémente le compteur
                current_run += 1
            else:
                if current_run > max_run:
                    max_run = current_run # On met à jour la longueur de la quinte la plus longue
                current_run = 1
        max_run = max(max_run, current_run) # On a donc la longueur de la quinte la plus longue
        
        if max_run >= 5:
            straight_draw = 1.0 
        elif max_run == 4:
            straight_draw = 0.8
        elif max_run == 3:
            straight_draw = 0.4
        elif max_run == 2:
            straight_draw = 0.1
        else:
            straight_draw = 0.0

        # --- Potentiel pour la couleur (flush draw) ---
        # Récupérer l'ensemble des couleurs uniques et les trier
        suit_counts = Counter(card.suit for card in cards)
        flush_draw = 0.0
        for count in suit_counts.values():
            if count >= 5:
                flush_draw = 1.0  # Couleur faite
            elif count == 4:
                flush_draw = max(flush_draw, 0.8)
            elif count == 3:
                flush_draw = max(flush_draw, 0.4)
            elif count == 2:
                flush_draw = max(flush_draw, 0.1)

        return straight_draw, flush_draw