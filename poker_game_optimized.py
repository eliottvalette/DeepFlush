# poker_game_opitmized.py
"""
Texas Hold'em, No Limit, 6 max.

Cette classe est optimisée pour intiliser une partie de poker en cours.
Dans le but d'effectuer des simulations de jeu pour l'algorithme MCCFR.
"""
import pygame
import random as rd
from typing import List, Dict, Optional, Tuple
from utils.config import DEBUG
from collections import Counter
import numpy as np
from poker_game import Player, PlayerAction, GamePhase, Card, SidePot, HandRank, Button, SCREEN_HEIGHT

class PokerGameOptimized:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
    def __init__(self, state : List[float], hero_cards: List[Tuple[int, str]], hero_seat: int, button_seat_position: int, visible_community_cards: List[Tuple[int, str]], rd_opponents_cards: List[List[Tuple[int, str]]], rd_missing_community_cards: List[Tuple[int, str]]):
        """
        Initialise la partie de poker avec un état plat pour la simulation MCCFR.
        """
        self.num_players = 6
        self.small_blind = 0.5
        self.big_blind = 1
        self.starting_stack = 100
        
        # Main pot (indice 127)
        self.main_pot = state[127] * self.starting_stack
        
        # Faire des copies pour éviter de modifier les listes originales
        self.community_cards = visible_community_cards.copy() if visible_community_cards else []
        self.rd_missing_community_cards = rd_missing_community_cards.copy() if rd_missing_community_cards else []
        self.rd_opponents_cards = rd_opponents_cards.copy() if rd_opponents_cards else []
        self.hero_cards = hero_cards
        self.side_pots = self._create_side_pots()
        
        # Phase de jeu (indices 49-54)
        phase_indices = {0: GamePhase.PREFLOP, 1: GamePhase.FLOP, 2: GamePhase.TURN, 
                        3: GamePhase.RIVER, 4: GamePhase.SHOWDOWN}
        phase_idx = np.argmax(state[47:52]) if any(state[47:52] > 0) else 0
        self.current_phase = phase_indices[phase_idx]
            
        # Position du bouton
        self.button_seat_position = button_seat_position
        self.current_player_seat = hero_seat
        
        # Mise maximale (indice 52)
        self.current_maximum_bet = state[52] * self.starting_stack
        
        # Initialiser les joueurs avec leurs stacks (indices 53-59)
        self.players = self._initialize_simulated_players(state)
        
        # Historique des actions pour pygame
        self.pygame_action_history = {player.name: [] for player in self.players}
        
        self.number_raise_this_game_phase = 0
        self.last_raiser = None
        self.action_buttons = self._create_action_buttons()
        
        # Initialiser les stacks APRÈS avoir créé les joueurs
        # On utilise les stacks déjà initialisés dans _initialize_simulated_players
        self.initial_stacks = {player.name: state[53 + player.seat_position] * self.starting_stack for player in self.players}
        self.net_stack_changes = {player.name: player.stack - self.initial_stacks[player.name] for player in self.players}
        self.final_stacks = {player.name: player.stack for player in self.players}

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
            rd_missing_community_cards_copy = self.rd_missing_community_cards.copy()
            if DEBUG:
                print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5:
                self.community_cards.append(rd_missing_community_cards_copy.pop())
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
                rd_missing_community_cards_copy = self.rd_missing_community_cards.copy()
                self.community_cards.append(rd_missing_community_cards_copy.pop())
            self.handle_showdown()
            return
        
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
        
        # Vérifier que nous avons suffisamment de cartes pour la distribution
        if not self.rd_missing_community_cards:
            if DEBUG:
                print("Attention: Aucune carte communautaire disponible pour la distribution.")
            return
            
        # Faire une copie locale des cartes communautaires pour éviter de les épuiser
        rd_missing_community_cards_copy = self.rd_missing_community_cards.copy()
        
        if self.current_phase == GamePhase.FLOP:
            for _ in range(3):
                self.community_cards.append(rd_missing_community_cards_copy.pop())
        elif self.current_phase in [GamePhase.TURN, GamePhase.RIVER]:
            self.community_cards.append(rd_missing_community_cards_copy.pop())
        
        self.rd_missing_community_cards = rd_missing_community_cards_copy

    def deal_cards(self):
        """
        Distribue deux cartes à chaque joueur actif.
        Réinitialise et mélange le jeu avant la distribution.
        """
        # Deal two cards to each active player
        for _ in range(2):
            for player in self.players:
                if not player.is_active:
                    continue
                if player.seat_position == self.hero_seat:
                    player.cards = self.hero_cards
                else:
                    player.cards = self.rd_opponents_cards.pop()

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
        # Pour la simulation MCCFR, nous utilisons une version simplifiée des boutons
        # sans dépendance à pygame
        from poker_game import PlayerAction
        
        # Créer un dictionnaire de boutons simplifiés
        class SimpleButton:
            def __init__(self):
                self.enabled = False
                
        buttons = {action: SimpleButton() for action in PlayerAction}
        
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
            raise ValueError(f"{player.name} n'était pas censé pouvoir faire une action, Raisons : actif = {player.is_active}, all-in = {player.is_all_in}, folded = {player.has_folded}, phase = {self.current_phase}")
        
        # Mettre à jour l'état des boutons avant de vérifier les actions valides
        self._update_button_states()
        valid_actions = [a for a in PlayerAction if self.action_buttons[a].enabled]
        if not valid_actions:
            raise ValueError(f"Aucune action valide disponible pour {player.name}")
        
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
        Gère la phase de showdown pour le mode simulation MCCFR.
        Version simplifiée qui n'utilise pas le deck.
        """
        if DEBUG:
            print("\n=== DÉBUT SHOWDOWN SIMULATION ===")
        
        self.current_phase = GamePhase.SHOWDOWN
        
        # Désactiver tous les boutons pendant le showdown
        for button in self.action_buttons.values():
            button.enabled = False
        
        # Pour la simulation MCCFR, nous avons seulement besoin de déterminer qui a gagné
        # et de mettre à jour les stacks en conséquence
        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        
        if len(active_players) == 1:
            # Si un seul joueur reste, il remporte tout le pot
            winner = active_players[0]
            winner.stack += self.main_pot
            if DEBUG:
                print(f"Victoire par fold - {winner.name} gagne {self.main_pot:.2f}BB")
            self.main_pot = 0
        else:
            # Dans le cas où plusieurs joueurs sont en jeu, nous évaluons leurs mains
            best_rank = -1
            winners = []
            
            for player in active_players:
                if player.cards:  # S'assurer que le joueur a des cartes
                    hand_rank, _ = self.evaluate_final_hand(player)
                    rank_value = hand_rank.value
                    
                    if rank_value > best_rank:
                        best_rank = rank_value
                        winners = [player]
                    elif rank_value == best_rank:
                        winners.append(player)
            
            # Distribuer le pot entre les gagnants
            if winners:
                share = self.main_pot / len(winners)
                for winner in winners:
                    winner.stack += share
                    if DEBUG:
                        print(f"{winner.name} gagne {share:.2f}BB")
                self.main_pot = 0
            else:
                if DEBUG:
                    print("Aucun gagnant déterminé")
        
        # Calculer les changements nets des stacks
        self.net_stack_changes = {player.name: (player.stack - self.initial_stacks.get(player.name, 0)) 
                                for player in self.players}
        self.final_stacks = {player.name: player.stack for player in self.players}
        
        if DEBUG:
            print("=== FIN SHOWDOWN SIMULATION ===\n")

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
        
    def play_trajectory(self, trajectory_action):
        """
        Simule le déroulement de la fin d'une partie à partir d'une action donnée.
        
        Args:
            trajectory_action (PlayerAction): L'action initiale à jouer, imposée au Hero
            
        Returns:
            float: Le gain final (positif ou négatif) pour le joueur courant
        """
        # Nous sommes au tour du joueur courant (hero)
        hero = self.players[self.current_player_seat]
        hero_initial_stack = hero.stack
        
        # On fait une copie des cartes pour éviter de les consommer dans l'original
        rd_opponents_cards_copy = self.rd_opponents_cards.copy() if self.rd_opponents_cards else []
        
        # Distribuer les cartes aux adversaires
        active_opponents = [p for p in self.players if p.is_active and not p.has_folded and p.seat_position != self.current_player_seat]
        for i, opponent in enumerate(active_opponents):
            if i < len(rd_opponents_cards_copy):
                opponent.cards = rd_opponents_cards_copy[i]
        
        # Mettre à jour les actions valides
        self._update_button_states()
        valid_actions = [a for a in PlayerAction if self.action_buttons[a].enabled]
        
        # Vérifier que des actions valides sont disponibles
        if not valid_actions:
            if DEBUG:
                print(f"Aucune action valide disponible pour {hero.name}. Impossible de continuer la simulation.")
            return 0
            
        # Le héros joue d'abord son action
        if trajectory_action not in valid_actions:
            # Si l'action fournie n'est pas valide, essayer de trouver une action alternative
            if valid_actions:
                if DEBUG:
                    print(f"L'action {trajectory_action} n'est pas valide. Utilisation d'une action alternative parmi: {valid_actions}")
                trajectory_action = rd.choice(valid_actions)
            else:
                if DEBUG:
                    print("Aucune action valide disponible. Fin de la simulation.")
                return 0
        
        bet_amount = None
        if trajectory_action == PlayerAction.ALL_IN:
            bet_amount = hero.stack
            
        # Exécute l'action du héros
        self.process_action(hero, trajectory_action, bet_amount)
        
        # Simulation du jeu jusqu'à la fin de la main
        while self.current_phase != GamePhase.SHOWDOWN:
            # Récupération du joueur actuel
            current_player = self.players[self.current_player_seat]
            
            # Si c'est le tour du héros à nouveau
            if current_player.seat_position == hero.seat_position:
                # Choix aléatoire d'une action valide
                self._update_button_states()
                valid_actions = [a for a in PlayerAction if self.action_buttons[a].enabled]
                if not valid_actions:
                    break  # Si aucune action n'est valide, sortir de la boucle
                
                random_action = rd.choice(valid_actions)
                bet_amount = None
                if random_action == PlayerAction.ALL_IN:
                    bet_amount = current_player.stack
                    
                self.process_action(current_player, random_action, bet_amount)
            else:
                # Pour les adversaires, on utilise une stratégie simplifiée
                self._update_button_states()
                valid_actions = [a for a in PlayerAction if self.action_buttons[a].enabled]
                if not valid_actions:
                    break  # Si aucune action n'est valide, sortir de la boucle
                
                # Stratégie simple pour les adversaires
                if PlayerAction.CHECK in valid_actions:
                    # Check si possible
                    self.process_action(current_player, PlayerAction.CHECK, None)
                elif PlayerAction.CALL in valid_actions and rd.random() < 0.7:
                    # Call avec 70% de probabilité si check n'est pas possible
                    self.process_action(current_player, PlayerAction.CALL, None)
                elif PlayerAction.FOLD in valid_actions:
                    # Fold si on ne peut pas call
                    self.process_action(current_player, PlayerAction.FOLD, None)
                else:
                    # Action aléatoire si aucune des actions précédentes n'est possible
                    random_action = rd.choice(valid_actions)
                    bet_amount = None
                    if random_action == PlayerAction.ALL_IN:
                        bet_amount = current_player.stack
                        
                    self.process_action(current_player, random_action, bet_amount)
        
        # À ce stade, nous avons atteint le showdown, calculer le gain du héros
        hero_final_stack = hero.stack
        payoff = hero_final_stack - hero_initial_stack
        
        return payoff

    def _initialize_simulated_players(self, state):
        """
        Initialise 6 joueurs simulés pour une partie MCCFR.
        """
        players = []
        
        # Extraction des stacks (indices 53-59)
        stacks = [state[53+i] * self.starting_stack for i in range(6)]
        
        # Extraction de l'état actif des joueurs (indices 65-71)
        active_states = [state[65+i] > 0 for i in range(6)]
        
        # Extraction des mises actuelles (indices 59-65)
        current_bets = [state[59+i] * self.starting_stack for i in range(6)]
        
        # Créer les joueurs
        for i in range(6):
            player = Player(
                name=f"Player_{i}",
                agent=None,
                stack=stacks[i],
                seat_position=i
            )
            player.is_active = active_states[i]
            player.has_folded = not active_states[i]
            player.current_player_bet = current_bets[i]
            player.total_bet = current_bets[i]
            player.cards = []
            players.append(player)
            
            player.role_position = (player.seat_position - self.button_seat_position - 1) % 6
        
        return players