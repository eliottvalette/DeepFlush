# poker_game_optimized.py
from poker_game import PlayerAction, PokerGame, HandRank, Card, Player, GamePhase
import random as rd
import numpy as np
from typing import List, Tuple, Dict

class PokerGameOptimized:
    def __init__(self, game: PokerGame):
        """
        Initialise la simulation optimisée à partir de l'état simplifié du jeu classique.
        On s’appuie sur get_simple_state() pour récupérer les informations essentielles.
        """
        self.simple_state = game.get_simple_state()
        self.current_phase = self.simple_state['phase']  # ex: "preflop"
        self.pot = self.simple_state['pot']
        self.current_maximum_bet = self.simple_state['current_maximum_bet']
        self.hero_cards = self.simple_state['hero_cards']
        self.community_cards = self.simple_state['community_cards'].copy()
        self.players_info = self.simple_state['players_info']  # liste de dicts pour chaque joueur
        self.num_active_players = self.simple_state['num_active_players']
        self.contributions = {player['name']: player.get('current_bet', 0) for player in self.players_info}
        self.bet_unit = self.current_maximum_bet if self.current_maximum_bet > 0 else 1

    def get_valid_actions(self, player_info: Dict) -> List[PlayerAction]: # TODO : comparer avec _update_button_states
        """
        Retourne la liste des actions valides pour un joueur donné dans l'état de simulation.
        Si le joueur a déjà foldé ou est all‑in, aucune action n'est possible.
        """
        if player_info['has_folded'] or player_info['is_all_in'] or self.current_phase == GamePhase.SHOWDOWN:
            return [] # Aucune action valide pour ces joueurs
        
        # ---- Activer tous les boutons par défaut ----
        valid_actions = [PlayerAction.FOLD, PlayerAction.CHECK,PlayerAction.CALL, PlayerAction.RAISE,PlayerAction.ALL_IN]
        
        current_bet = player_info['current_bet']
        stack = player_info['stack']
        
        # ---- CHECK ----
        if current_bet < self.current_maximum_bet: # Si le joueur n'a pas égalisé la mise maximale, il ne peut pas check
            valid_actions.remove(PlayerAction.CHECK)

        # ---- FOLD ----
        if PlayerAction.CHECK in valid_actions: # Si le joueur peut check, il ne peut pas fold
            valid_actions.remove(PlayerAction.FOLD)

        # ---- CALL ----
        # Désactiver call si pas de mise à suivre ou pas assez de jetons
        if current_bet == self.current_maximum_bet: # Si le joueur a égalisé la mise maximale, il ne peut pas call
            valid_actions.remove(PlayerAction.CALL)
        elif stack < (self.current_maximum_bet - current_bet): # Si le joueur n'a pas assez de jetons pour suivre la mise maximale, il ne peut pas call
            valid_actions.remove(PlayerAction.CALL)
            # Activer all-in si le joueur a des jetons mais pas assez pour call
            if stack > 0:
                self.action_buttons[PlayerAction.ALL_IN].enabled = True
            valid_actions.remove(PlayerAction.RAISE)

        # ---- RAISE ----
        # Calculer le raise minimal
        if self.current_maximum_bet == 0:
            min_raise = self.big_blind
        else:
            min_raise = (self.current_maximum_bet - current_bet) * 2

        # Désactiver l'action raise si le joueur n'a pas suffisamment de jetons pour couvrir le raise minimum
        if stack < min_raise:
            valid_actions.remove(PlayerAction.RAISE)

        # Désactiver raise si déjà 4 relances dans le tour
        if self.number_raise_this_game_phase >= 4:
            valid_actions.remove(PlayerAction.RAISE)

        # ---- ALL-IN ----
        # All-in disponible si le joueur a des jetons, qu'il soit le premier à agir ou non
        if stack <= 0 :
            valid_actions.remove(PlayerAction.ALL_IN)
        
        return valid_actions

    def simulate_action(self, player_info: Dict, action: PlayerAction) -> None:
        """
        Applique l'effet de l'action choisie par un joueur sur l'état de simulation.
        On met à jour sa contribution, son stack et la taille du pot.
        Pour simplifier, le montant de la mise est fixé à self.bet_unit (pour la relance).
        """
        name = player_info['name']
        current_bet = player_info.get('current_bet', 0)
        stack = player_info.get('stack', 0)
        if action == PlayerAction.FOLD:
            player_info['has_folded'] = True
        elif action == PlayerAction.CHECK:
            # Aucune modification.
            pass
        elif action == PlayerAction.CALL:
            call_amount = self.current_maximum_bet - current_bet
            call_amount = min(call_amount, stack)
            player_info['stack'] = stack - call_amount
            player_info['current_bet'] = current_bet + call_amount
            self.contributions[name] += call_amount
            self.pot += call_amount
            if player_info['stack'] == 0:
                player_info['is_all_in'] = True
        elif action == PlayerAction.RAISE:
            # On relance d'un montant fixe : le joueur doit d'abord caller puis ajouter self.bet_unit.
            call_amount = self.current_maximum_bet - current_bet
            raise_amount = self.bet_unit
            total_bet = call_amount + raise_amount
            total_bet = min(total_bet, stack)
            player_info['stack'] = stack - total_bet
            player_info['current_bet'] = current_bet + total_bet
            self.contributions[name] += total_bet
            self.pot += total_bet
            if player_info['current_bet'] > self.current_maximum_bet:
                self.current_maximum_bet = player_info['current_bet']
            if player_info['stack'] == 0:
                player_info['is_all_in'] = True
        elif action == PlayerAction.ALL_IN:
            all_in_amount = stack
            player_info['stack'] = 0
            player_info['current_bet'] = current_bet + all_in_amount
            self.contributions[name] += all_in_amount
            self.pot += all_in_amount
            if player_info['current_bet'] > self.current_maximum_bet:
                self.current_maximum_bet = player_info['current_bet']
            player_info['is_all_in'] = True

    def betting_round(self, hero_trajectory_action: PlayerAction) -> None:
        """
        Simule un round de pari dans lequel chaque joueur actif agit une fois.
        Pour le joueur cible (hero), on force l'action passée en paramètre ;
        pour les autres, l'action est choisie aléatoirement parmi les actions valides.
        La mise à jour du pot et des contributions est effectuée.
        """
        for player_info in self.players_info:
            # Si le joueur est déjà foldé ou all‑in, on ne fait rien.
            if player_info.get('has_folded', False) or player_info.get('is_all_in', False):
                continue
            valid = self.get_valid_actions(player_info)
            if not valid: # Aucune action valid on passe au suivant (le joueur a fold ou est all-in)
                continue

            # On considère que le hero est le premier joueur de players_info.
            is_hero = (player_info['name'] == self.simple_state['hero_name'])
            if is_hero:
                chosen_action = rd.choice(valid) # TODO
            else:
                chosen_action = rd.choice(valid) # TODO
            self.simulate_action(player_info, chosen_action)

    def advance_phase(self, rd_missing_community_cards: List[Tuple[int, str]]) -> None:
        """
        Fait évoluer la phase du jeu en complétant le board selon la phase.
        Par exemple, de preflop au flop (ajout de 3 cartes), puis turn et river.
        """
        phase_order = ["preflop", "flop", "turn", "river", "showdown"]
        if self.current_phase not in phase_order:
            self.current_phase = "preflop"
        if self.current_phase == "preflop":
            needed = 3 - len(self.community_cards)
            if needed > 0:
                self.community_cards.extend(rd_missing_community_cards[:needed])
            self.current_phase = "flop"
        elif self.current_phase == "flop":
            if len(self.community_cards) < 4:
                self.community_cards.extend(rd_missing_community_cards[:1])
            self.current_phase = "turn"
        elif self.current_phase == "turn":
            if len(self.community_cards) < 5:
                self.community_cards.extend(rd_missing_community_cards[:1])
            self.current_phase = "river"
        elif self.current_phase == "river":
            self.current_phase = "showdown"

    def simulate_hand(self, hero_trajectory_action: PlayerAction, rd_missing_community_cards: List[Tuple[int, str]]) -> None:
        """
        Simule la suite de la main depuis l'état actuel jusqu'au showdown.
        À chaque round de pari, le hero (premier joueur) utilisera l'action fournie (pour la première itération)
        puis pour les rounds suivants les actions seront tirées aléatoirement.
        """
        current_hero_action = hero_trajectory_action
        while self.current_phase != "showdown":
            self.betting_round(current_hero_action)
            self.advance_phase(rd_missing_community_cards)
            # Pour les rounds suivants, on ne force plus une action particulière pour le hero.
            current_hero_action = rd.choice(list(PlayerAction))

    def evaluate_showdown(self) -> Dict[str, float]:
        """
        À l'issue du showdown, évalue les mains de tous les joueurs encore en lice
        et répartit le pot entre les gagnants.
        
        Pour cela, on recrée des instances de Player (de la classe du jeu classique)
        en assignant au hero ses cartes connues et aux adversaires des mains aléatoires
        (tirées du deck restant).
        Le payoff de chaque joueur est calculé comme : (part du pot remportée) - (contribution).
        
        Retourne un dictionnaire {nom_du_joueur: payoff}.
        """
        # Pour évaluer, on crée une liste de joueurs simulés.
        simulated_players = []
        # On considère que le hero est le premier joueur de players_info.
        for i, info in enumerate(self.players_info):
            if info.get('has_folded', False):
                continue  # Les joueurs foldés n'entrent pas en showdown
            # Création d'une instance de Player pour l'évaluation.
            sim_player = Player(agent=None, name=info['name'], stack=info['stack'], seat_position=i)
            if i == 0:
                # Pour le hero, on affecte les cartes connues.
                sim_player.cards = [Card(value, suit) for value, suit in self.hero_cards]
            else:
                # Pour les adversaires, si aucune main n'est assignée, on attribue 2 cartes aléatoires
                # parmi le deck complet en retirant les cartes déjà visibles.
                full_deck = [(v, s) for v in range(2, 15) for s in ["♠", "♥", "♦", "♣"]]
                known = self.hero_cards + self.community_cards
                remaining = [card for card in full_deck if card not in known]
                opp_cards = rd.sample(remaining, 2)
                sim_player.cards = [Card(value, suit) for value, suit in opp_cards]
            simulated_players.append(sim_player)
        # Pour évaluer, nous utilisons la méthode evaluate_final_hand du jeu classique.
        game_dummy = PokerGame([])  # On crée un "dummy" dont on fixera le board
        game_dummy.community_cards = [Card(value, suit) for value, suit in self.community_cards]
        hand_evals = {}
        for player in simulated_players:
            try:
                eval_result = game_dummy.evaluate_final_hand(player)
            except Exception as e:
                eval_result = (HandRank.HIGH_CARD, [0])
            hand_evals[player.name] = eval_result
        # Détermination des gagnants : on compare (rang de main, puis kickers)
        best_eval = None
        winners = []
        for name, eval_result in hand_evals.items():
            key = (eval_result[0].value, tuple(eval_result[1]))
            if best_eval is None or key > best_eval:
                best_eval = key
                winners = [name]
            elif key == best_eval:
                winners.append(name)
        # Calcul du payoff pour chaque joueur : payoff = (part du pot si gagnant) - contribution
        payoffs = {}
        num_winners = len(winners)
        for info in self.players_info:
            name = info['name']
            contribution = self.contributions.get(name, 0)
            if name in winners:
                share = self.pot / num_winners
                payoffs[name] = share - contribution
            else:
                payoffs[name] = -contribution
        return payoffs

    def play_trajectory(self, trajectory_action: PlayerAction, rd_opponents_cards: List[Tuple[int, str]], rd_missing_community_cards: List[Tuple[int, str]]) -> float:
        """
        Simule une trajectoire en faisant jouer la suite de la main (en choisissant aléatoirement
        parmi les actions valides) jusqu'au showdown, puis évalue le payoff pour le joueur cible (hero).
        
        Args:
            trajectory_action (PlayerAction): L'action initiale à appliquer pour le hero.
            rd_opponents_cards (List[Tuple[int, str]]): Les mains aléatoires (tirées) pour les adversaires.
            rd_missing_community_cards (List[Tuple[int, str]]): Les cartes manquantes pour compléter le board.
        
        Returns:
            float: Le payoff simulé pour le hero.
        """
        # Simuler la suite de la main.
        self.simulate_hand(trajectory_action, rd_opponents_cards, rd_missing_community_cards)
        # À l'issue du showdown, évaluer et calculer le payoff.
        payoffs = self.evaluate_showdown()
        # On suppose que le hero est le premier joueur (players_info[0]).
        hero_name = self.players_info[0]['name']
        return payoffs.get(hero_name, 0.0)
