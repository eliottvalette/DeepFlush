# poker_game_optimized.py
from poker_game import PlayerAction, PokerGame, HandRank, Card, Player, GamePhase
import random as rd
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import json

class PokerGameOptimized:
    def __init__(self, flat_state_and_count: Tuple[List, int]):
        """
        Initialise la simulation optimisée à partir de l'état simplifié vectorisé du jeu classique.
        
        Args:
            flat_state_and_count (Tuple[List, int]): Tuple contenant le vecteur d'état plat et le nombre de joueurs actifs
        """
        flat_state, num_active_players = flat_state_and_count
        
        # Extraction des informations du vecteur plat
        idx = 0
        
        # Info du héros
        hero_idx = flat_state[idx]
        idx += 1
        
        # Cartes du héros
        hero_cards = []
        for _ in range(2):
            value = flat_state[idx]
            suit = ["♠", "♥", "♦", "♣"][int(flat_state[idx + 1])]
            hero_cards.append((value, suit))
            idx += 2
        
        # Cartes communes
        community_cards = []
        for _ in range(5):
            value = flat_state[idx]
            suit_idx = flat_state[idx + 1]
            if value != -1 and suit_idx != -1:
                suit = ["♠", "♥", "♦", "♣"][int(suit_idx)]
                community_cards.append((value, suit))
            idx += 2
        
        # Phase et informations générales
        phase_value = flat_state[idx]
        idx += 1
        pot = flat_state[idx]
        idx += 1
        current_maximum_bet = flat_state[idx]
        idx += 1
        big_blind = flat_state[idx]
        idx += 1
        small_blind = flat_state[idx]
        idx += 1
        
        # Reconstruction des informations des joueurs
        players_info = []
        for _ in range(num_active_players):
            player_info = {
                'name': f'Player_{int(flat_state[idx])}',
                'player_stack': flat_state[idx + 1],
                'current_player_bet': flat_state[idx + 2],
                'is_active': bool(flat_state[idx + 3]),
                'has_folded': bool(flat_state[idx + 4]),
                'is_all_in': bool(flat_state[idx + 5]),
                'role_position': int(flat_state[idx + 6]),
                'has_acted': bool(flat_state[idx + 7]),
                'seat_position': int(flat_state[idx + 8])
            }
            players_info.append(player_info)
            idx += 9
        
        # Trier les joueurs par position
        players_info.sort(key=lambda x: x['role_position'])
        
        # Reconstruction du dictionnaire d'état
        self.simple_state = {
            'hero_name': f'Player_{hero_idx}',
            'hero_index': next(i for i, p in enumerate(players_info) if p['name'] == f'Player_{hero_idx}'),
            'hero_cards': hero_cards,
            'community_cards': community_cards,
            'phase': GamePhase(phase_value),
            'pot': pot,
            'current_maximum_bet': current_maximum_bet,
            'players_info': players_info,
            'num_active_players': num_active_players,
            'big_blind': big_blind,
            'small_blind': small_blind
        }
        
        # Initialisation des autres attributs comme avant
        self.current_phase = self.simple_state['phase']
        self.pot = self.simple_state['pot']
        self.big_blind = self.simple_state['big_blind']
        self.current_maximum_bet = self.simple_state['current_maximum_bet']
        self.current_player_idx = self.simple_state['hero_index']
        self.hero_cards = self.simple_state['hero_cards']
        self.community_cards = self.simple_state['community_cards'].copy()
        self.players_info = self.simple_state['players_info']
        self.num_active_players = self.simple_state['num_active_players']
        self.contributions = {player['name']: player['current_player_bet'] for player in self.players_info}
        self.number_raise_this_game_phase = 0
        self.hero_first_action = True
        self.num_players = len(self.players_info)
        self.rd_opponents_cards = []
        self.rd_missing_community_cards = []

    def _get_remaining_deck(self, known_cards: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """
        Retourne la liste des cartes restantes dans le deck.
        
        Args:
            known_cards: Liste des cartes déjà visibles (cartes du héros + cartes communes)
            
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

    def get_valid_actions(self, player_info: Dict) -> List[PlayerAction]:
        """
        Retourne la liste des actions valides pour un joueur donné dans l'état de simulation.
        Si le joueur a déjà foldé ou est all‑in, aucune action n'est possible.
        """

        if player_info['has_folded'] or self.current_phase == GamePhase.SHOWDOWN:
            raise ValueError(f"Le joueur {player_info['name']} a déjà foldé ou est au showdown - on ne devrait pas etre en train de check ses actions valides")

        # Si le joueur est all-in, aucune action n'est possible
        if player_info['is_all_in'] :
            return []
        
        # ---- Activer tous les boutons par défaut ----
        valid_actions = [
            PlayerAction.FOLD, 
            PlayerAction.CHECK,
            PlayerAction.CALL, 
            PlayerAction.RAISE,
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
            PlayerAction.RAISE_3X_POT,
            PlayerAction.ALL_IN
        ]
        
        current_player_bet = player_info['current_player_bet']
        player_stack = player_info['player_stack']
        
        # ---- CHECK ----
        if current_player_bet < self.current_maximum_bet: # Si le joueur n'a pas égalisé la mise maximale, il ne peut pas check
            valid_actions.remove(PlayerAction.CHECK)

        # ---- FOLD ----
        if PlayerAction.CHECK in valid_actions: # Si le joueur peut check, il ne peut pas fold
            valid_actions.remove(PlayerAction.FOLD)

        # ---- CALL ----
        # Désactiver call si pas de mise à suivre ou pas assez de jetons
        if current_player_bet == self.current_maximum_bet: # Si le joueur a égalisé la mise maximale, il ne peut pas call
            valid_actions.remove(PlayerAction.CALL)
        elif player_stack < (self.current_maximum_bet - current_player_bet): # Si le joueur n'a pas assez de jetons pour suivre la mise maximale, il ne peut pas call
            valid_actions.remove(PlayerAction.CALL)
            valid_actions.remove(PlayerAction.RAISE)
            # Remove all pot-based raises as well
            pot_based_actions = [
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
            for action in pot_based_actions:
                try:
                    valid_actions.remove(action)
                except ValueError:
                    pass

        # ---- RAISE ----
        # Calculer le raise minimal
        if self.current_maximum_bet == 0:
            min_raise = self.simple_state['big_blind']
        else:
            min_raise = (self.current_maximum_bet - current_player_bet) * 2

        # Désactiver l'action raise si le joueur n'a pas suffisamment de jetons pour couvrir le raise minimum
        if player_stack < min_raise or self.number_raise_this_game_phase >= 4:
            try:
                valid_actions.remove(PlayerAction.RAISE)
            except ValueError:
                pass

            # Also remove all pot-based raises
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

            for action, percentage in raise_percentages.items():
                # Calculate required raise for this pot-based action
                pot_raise = self.pot * percentage
                if pot_raise < min_raise or player_stack < pot_raise:
                    try:
                        valid_actions.remove(action)
                    except ValueError:
                        pass

        # ---- ALL-IN ----
        # All-in disponible si le joueur a des jetons, qu'il soit le premier à agir ou non
        if player_stack <= 0 :
            valid_actions.remove(PlayerAction.ALL_IN)
        
        return valid_actions

    def simulate_action(self, player_info: Dict, action: PlayerAction) -> None:
        """
        Applique l'effet de l'action choisie par un joueur sur l'état de simulation.
        On met à jour sa contribution, son stack et la taille du pot.
        Pour simplifier, le montant de la mise est fixé à self.bet_unit (pour la relance).
        """

        name = player_info['name']
        current_player_bet = player_info['current_player_bet']
        player_stack = player_info['player_stack']

        if action == PlayerAction.FOLD:
            print(f"{name} a fold")
            player_info['has_folded'] = True

        elif action == PlayerAction.CHECK:
            print(f"{name} a check")
            # Add validation to prevent invalid CHECK actions
            if player_info['current_player_bet'] < self.current_maximum_bet:
                raise ValueError(f"Invalid CHECK action - player must match current bet of {self.current_maximum_bet}")
            # No modification for check
            pass

        elif action == PlayerAction.CALL:
            print(f"{name} a call")
            call_amount = self.current_maximum_bet - current_player_bet
            if call_amount > player_stack: 
                raise ValueError(f"{name} n'a pas assez de jetons pour suivre la mise maximale, il n'aurait pas du avoir le droit de call")
            
            player_info['player_stack'] = player_stack - call_amount
            player_info['current_player_bet'] = current_player_bet + call_amount
            self.contributions[name] += call_amount
            self.pot += call_amount
            if player_info['player_stack'] == 0:
                player_info['is_all_in'] = True

        elif action == PlayerAction.RAISE:
            print(f"{name} a raise")
            call_amount = self.current_maximum_bet - current_player_bet
            if self.current_maximum_bet == 0:
                raise_amount = self.big_blind
            else:
                raise_amount = (self.current_maximum_bet - player_info['current_player_bet']) * 2
            total_bet = call_amount + raise_amount
            total_bet = min(total_bet, player_stack)
            player_info['player_stack'] = player_stack - total_bet
            player_info['current_player_bet'] = current_player_bet + total_bet
            self.contributions[name] += total_bet
            self.number_raise_this_game_phase += 1
            self.pot += total_bet
            if player_info['current_player_bet'] > self.current_maximum_bet:
                self.current_maximum_bet = player_info['current_player_bet']
            if player_info['player_stack'] == 0:
                player_info['is_all_in'] = True

        # Add handling for pot-based raise actions
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
            print(f"{name} a raise (pot-based)")
            # Map actions to their pot percentages
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
            raise_amount = self.pot * percentage
            
            # Ensure minimum raise amount
            if self.current_maximum_bet == 0:
                min_raise = self.big_blind
            else:
                min_raise = (self.current_maximum_bet - current_player_bet) * 2
            
            raise_amount = max(raise_amount, min_raise)
            total_bet = min(raise_amount, player_stack)  # Cap at player's stack
            
            player_info['player_stack'] = player_stack - total_bet
            player_info['current_player_bet'] = current_player_bet + total_bet
            self.contributions[name] += total_bet
            self.number_raise_this_game_phase += 1
            self.pot += total_bet
            if player_info['current_player_bet'] > self.current_maximum_bet:
                self.current_maximum_bet = player_info['current_player_bet']
            if player_info['player_stack'] == 0:
                player_info['is_all_in'] = True

        elif action == PlayerAction.ALL_IN:
            print(f"{name} a all-in")
            # Calculer le montant réel de l'all-in en tenant compte des mises déjà faites
            remaining_stack = player_info['player_stack']
            all_in_amount = remaining_stack
            
            player_info['player_stack'] = 0
            player_info['current_player_bet'] += all_in_amount  # Ajouter à la mise existante
            self.contributions[name] += all_in_amount
            self.pot += all_in_amount
            
            if player_info['current_player_bet'] > self.current_maximum_bet:
                self.current_maximum_bet = player_info['current_player_bet']
            player_info['is_all_in'] = True
            print(f"{name} a all-in {all_in_amount}BB (mise totale: {player_info['current_player_bet']}BB)")
        else :
            raise ValueError(f"Action invalide: {action}")
        player_info['has_acted'] = True

    def betting_round(self, hero_trajectory_action: PlayerAction) -> Tuple[bool, bool]:
        print(f"\n--- Début betting_round - phase : {self.current_phase} ---")
        print(f"Index joueur actuel: {self.current_player_idx}")
        
        # Réinitialiser has_acted pour les joueurs actifs non all-in
        for player_info in self.players_info:
            if not (player_info['has_folded'] or player_info['is_all_in']):
                player_info['has_acted'] = False

        # Trouver le premier joueur actif
        in_loop = 0
        while self.players_info[self.current_player_idx]['has_folded'] or self.players_info[self.current_player_idx]['is_all_in']:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            in_loop += 1
            if in_loop > 10:
                print("Etat des joueurs avant le round :")
                for p in self.players_info:
                    print(f"{p['name']}: fold={p['has_folded']}, all_in={p['is_all_in']}, acted={p['has_acted']}, bet={p['current_player_bet']}")
                raise ValueError("ATTENTION: Boucle infinie dans betting_round! before betting_round while loop")
        
        iteration = 0
        while iteration < 20:
            # Vérifier si la phase est déja terminée
            phase_completed, instant_finish = self.check_phase_completion()
            if phase_completed :
                print(f"Phase terminée - sortie de la boucle")
                break
            if instant_finish:
                print("Showdown instantané - sortie de la boucle")
                break
            
            # Faire agir le joueur courant
            player_info = self.players_info[self.current_player_idx]
            print(f"Joueur actuel: {player_info['name']}")

            valid_actions = self.get_valid_actions(player_info)
            print(f"Actions valides: {[a.value for a in valid_actions]}")

            if not valid_actions:
                print("player_info :", player_info)
                raise ValueError("WARNING: Aucune action valide pour le joueur courant!")
            
            is_hero = (player_info['name'] == self.simple_state['hero_name'])

            print('is_hero :', is_hero)

            if is_hero:
                if self.hero_first_action:
                    chosen_action = hero_trajectory_action
                    self.hero_first_action = False
                else:
                    chosen_action = rd.choice(valid_actions)
            else:
                chosen_action = rd.choice(valid_actions)

            print(f"Action choisie: {chosen_action}")
            self.simulate_action(player_info, chosen_action)

            # Vérifier si la phase est terminée
            phase_completed, instant_finish = self.check_phase_completion()
            if phase_completed :
                print(f"Phase terminée - sortie de la boucle")
                break
            if instant_finish:
                print("Showdown instantané - sortie de la boucle")
                break

            # Passer au joueur suivant
            in_loop = 0
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            while self.players_info[self.current_player_idx]['has_folded'] or self.players_info[self.current_player_idx]['is_all_in']:
                self.current_player_idx = (self.current_player_idx + 1) % self.num_players
                in_loop += 1
                if in_loop > 10:
                    print("Etat des joueurs avant le round :")
                    for p in self.players_info:
                        print(f"{p['name']}: fold={p['has_folded']}, all_in={p['is_all_in']}, acted={p['has_acted']}, bet={p['current_player_bet']}")
                    raise ValueError("ATTENTION: Boucle infinie dans betting_round!")

            iteration += 1
            if iteration >= 20:
                print('current_player_idx :', self.current_player_idx)
                for p in self.players_info:
                            print(f"{p['name']}: fold={p['has_folded']}, all_in={p['is_all_in']}, acted={p['has_acted']}, bet={p['current_player_bet']}")
                raise ValueError("ATTENTION: Nombre maximum d'itérations atteint dans betting_round!")

        return phase_completed, instant_finish

    def check_phase_completion(self):
        """
        Vérifie si le tour d'enchères est terminé.
        """
        # Récupérer les joueurs actifs
        active_players = [player_info for player_info in self.players_info if not ((player_info['has_folded']) or (player_info['is_all_in']))]
        non_folded_players = [player_info for player_info in self.players_info if not player_info['has_folded']]
        folded_players = [player_info for player_info in self.players_info if player_info['has_folded']]
        
        # Tout le monde a fold, signaler une erreur car la partie est censée être terminée avant que le dernier joueur agisse car c'était le dernier joueur en jeu
        if len(folded_players) == self.num_players:
            for p in folded_players:
                print(f"{p['name']}: fold={p['has_folded']}, all_in={p['is_all_in']}, acted={p['has_acted']}, bet={p['current_player_bet']}")
            raise ValueError("ATTENTION: Tout le monde a foldé!")
        
        # Tout le monde a fold ou all-in
        if len(active_players) == 0:
            print("Tout le monde a foldé ou all-in - showdown instantané")
            return True, True
        
        # Si un seul joueur reste
        if len(non_folded_players) == 1:
            print("Un seul joueur reste - showdown instantané")
            return True, True
        
        # Vérifier que tous les joueurs ont agi et égalisé la mise maximale
        for player_info in active_players:
            # Si le joueur n'a pas encore agi dans la phase, le tour n'est pas terminé
            if not player_info['has_acted']:
                return False, False

            # Si le joueur n'a pas égalisé la mise maximale et n'est pas all-in, le tour n'est pas terminé
            if player_info['current_player_bet'] < self.current_maximum_bet and not player_info['is_all_in']:
                return False, False
        
        # Nouvelle condition: vérifier si tous les joueurs non foldés ont égalisé la mise maximale
        all_bets_equal = True
        max_bet = max(p['current_player_bet'] for p in non_folded_players)
        for player_info in non_folded_players:
            if not player_info['is_all_in'] and player_info['current_player_bet'] != max_bet:
                all_bets_equal = False
                break
        
        if not all_bets_equal:
            return False, False
        
        # Atteindre cette partie du code signifie que la phase est terminée
        print(f"Phase terminée, tous les joueurs ont agi et égalisé la mise maximale")
        phase_completed = True
        if self.current_phase == GamePhase.RIVER:
            print("River complete - going to showdown")
            return phase_completed, False
        
        return phase_completed, False

    def advance_phase(self) -> None:
        """
        Fait évoluer la phase du jeu en complétant le board selon la phase.
        """
        print("\n--- Début advance_phase ---")
        print(f"Phase actuelle: {self.current_phase}")
        print(f"Cartes communes actuelles: {self.community_cards}")
        print(f"Cartes restantes à ajouter: {self.rd_missing_community_cards}")

        # Progression des phases et distribution des cartes
        if self.current_phase == GamePhase.PREFLOP:
            # Au flop, on ajoute 3 cartes
            self.community_cards.extend(self.rd_missing_community_cards[:3])
            del self.rd_missing_community_cards[:3]
            self.current_phase = GamePhase.FLOP
        elif self.current_phase == GamePhase.FLOP:
            # Au turn, on ajoute 1 carte
            self.community_cards.append(self.rd_missing_community_cards[0])
            del self.rd_missing_community_cards[0]
            self.current_phase = GamePhase.TURN
        elif self.current_phase == GamePhase.TURN:
            # À la river, on ajoute 1 carte
            self.community_cards.append(self.rd_missing_community_cards[0])
            del self.rd_missing_community_cards[0]
            self.current_phase = GamePhase.RIVER
        elif self.current_phase == GamePhase.RIVER:
            self.current_phase = GamePhase.SHOWDOWN

        print(f"Nouvelle phase: {self.current_phase}")
        print(f"Nouvelles cartes communes: {self.community_cards}")

        # Réinitialiser les mises pour la nouvelle phase
        self.current_maximum_bet = 0
        for player_info in self.players_info:
            player_info['current_player_bet'] = 0
            if not player_info["has_folded"] and not player_info["is_all_in"]:
                player_info["has_acted"] = False

        # Réinitialiser le compteur de raises pour la nouvelle phase
        self.number_raise_this_game_phase = 0
        
        # Trouver le premier joueur actif après la SB pour les phases post-flop
        if self.current_phase != GamePhase.PREFLOP:
            # Commencer par l'index 0 (SB) et chercher le premier joueur actif
            for i in range(self.num_players):
                player = self.players_info[i]
                if not (player['has_folded'] or player['is_all_in']):
                    self.current_player_idx = i
                    break

        print("--- Fin advance_phase ---\n")

    def simulate_hand(self, hero_trajectory_action: PlayerAction, valid_actions: List[PlayerAction]) -> None:
        """
        Simule la suite de la main depuis l'état actuel jusqu'au showdown.
        """
        print("\n=== Début simulate_hand ===")
        print(f"Phase initiale: {self.current_phase}")
        
        # Convert GamePhase enum to string value for JSON serialization
        json_safe_state = self.simple_state.copy()
        if 'phase' in json_safe_state:
            json_safe_state['phase'] = str(json_safe_state['phase'])
        
        print(f"Action de la trajectoire: {hero_trajectory_action} parmi les actions à parcourir {[action.value for action in valid_actions]}")
        
        # Ne pas simuler si déjà au showdown
        if self.current_phase == GamePhase.SHOWDOWN:
            print("Déjà au showdown - pas de simulation nécessaire")
            return False
        
        iteration = 0
        instant_finish = False
        while self.current_phase != GamePhase.SHOWDOWN:
            print(f"\nItération {iteration} de simulate_hand")            
            phase_completed, instant_finish = self.betting_round(hero_trajectory_action)
            print(f"--- Phase completed: {phase_completed}, Instant win: {instant_finish} ---")
            
            if instant_finish :
                print("Instant showdown détecté - sortie totale de la boucle")
                break
            elif phase_completed:
                print(f"Phase {self.current_phase} terminée - passage à la phase suivante")
                self.advance_phase()
                print(f"Nouvelle phase: {self.current_phase}")
            else :
                raise ValueError("ATTENTION: Phase non terminée dans la boucle principale de simulate_hand")
            
        print("=== Fin simulate_hand ===\n")
        return instant_finish

    def evaluate_showdown(self, instant_finish: bool) -> Dict[str, float]:
        """
        Évalue les mains des joueurs et distribue le pot.
        
        Args:
            instant_finish (bool): True si un seul joueur reste en jeu (les autres ont fold)
            
        Returns:
            Dict[str, float]: Dictionnaire {nom_joueur: gain/perte}
        """
        # Vérifier les joueurs restants
        remaining_players = [info for info in self.players_info if not info['has_folded']]
        
        # Si personne n'est resté dans le coup, retourner des pertes pour tous
        if len(remaining_players) == 0:
            raise ValueError("Erreur: Aucun joueur restant au showdown!")
        
        # Victoire par fold
        if len(remaining_players) == 1 or instant_finish:
            winner = remaining_players[0]
            payoffs = {}
            for info in self.players_info:
                name = info['name']
                contribution = self.contributions[name]
                if name == winner['name']:
                    payoffs[name] = self.pot - contribution  # Gagne le pot moins sa contribution
                else:
                    payoffs[name] = -contribution  # Perd sa contribution
            return payoffs

        # Évaluation normale des mains
        # Créer une liste des joueurs encore en jeu avec leurs cartes
        active_players = []
        opponent_idx = 0
        
        for player_info in self.players_info:
            if player_info['has_folded']:
                continue
            
            # Créer un dictionnaire avec les informations nécessaires pour l'évaluation
            player_data = {
                'name': player_info['name'],
                'cards': []
            }
            
            # Assigner les cartes selon qu'il s'agit du hero ou d'un adversaire
            if player_info['name'] == self.simple_state['hero_name']:
                player_data['cards'] = self.hero_cards
            else:
                player_data['cards'] = self.rd_opponents_cards[opponent_idx]
                opponent_idx += 1
                
            active_players.append(player_data)

        # Évaluer la main de chaque joueur
        hand_ranks = {}
        for player_data in active_players:
            hand_eval = self._evaluate_hand(player_data['cards'] + self.community_cards)
            hand_ranks[player_data['name']] = hand_eval

        # Déterminer le(s) gagnant(s) en comparant les rangs puis les kickers
        best_eval = None
        winners = []
        for name, hand_eval in hand_ranks.items():
            current_key = (hand_eval[0].value, tuple(hand_eval[1]))  # (HandRank.value, kickers)
            if best_eval is None or current_key > best_eval:
                best_eval = current_key
                winners = [name]
            elif current_key == best_eval:
                winners.append(name)
        
        # Calculer les gains/pertes
        payoffs = {}
        share = self.pot / len(winners)  # Partage équitable du pot entre les gagnants
        
        for player_info in self.players_info:
            name = player_info['name']
            contribution = self.contributions[name]
            if name in winners:
                payoffs[name] = share - contribution
            else:
                payoffs[name] = -contribution

        return payoffs

    def play_trajectory(self, trajectory_action: PlayerAction, rd_opponents_cards: List[List[Tuple[int, str]]], rd_missing_community_cards: List[Tuple[int, str]], valid_actions: List[PlayerAction]) -> float:
        """
        Simule une trajectoire complète et retourne le payoff pour le héros.
        """
        self.rd_missing_community_cards = rd_missing_community_cards.copy()
        self.rd_opponents_cards = rd_opponents_cards.copy()

        # Si déjà au showdown, on relève une erreur
        if self.current_phase == GamePhase.SHOWDOWN:
            raise ValueError("Erreur: Déjà au showdown - pas de simulation possible")
        
        # Simuler la main
        instant_finish = self.simulate_hand(trajectory_action, valid_actions)
        
        # À l'issue du showdown, évaluer et calculer le payoff
        payoffs = self.evaluate_showdown(instant_finish)
        print(f"Payoffs: {payoffs}")
        return payoffs[self.simple_state['hero_name']]

    def _evaluate_hand(self, cards: List[Tuple[int, str]]) -> Tuple[HandRank, List[int]]:
        """
        Évalue la meilleure main possible d'un joueur.
        
        Args:
            cards: Liste de tuples (valeur, couleur) représentant les cartes
            
        Returns:
            Tuple[HandRank, List[int]]: (rang de la main, liste des valeurs importantes)
        """
        if not cards:
            raise ValueError("Cannot evaluate hand - no cards provided")
        
        # Extraire les valeurs et couleurs
        values = [card[0] for card in cards]
        suits = [card[1] for card in cards]
        
        # Vérifie si une couleur est possible (5+ cartes de même couleur)
        suit_counts = Counter(suits)
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 5), None)
        
        # Si une couleur est possible, on vérifie d'abord les mains les plus fortes
        if flush_suit:
            # Trie les cartes de la couleur par valeur décroissante
            flush_cards = sorted([card[0] for card in cards if card[1] == flush_suit], reverse=True)
            
            # Vérifie si on a une quinte flush
            for i in range(len(flush_cards) - 4):
                if flush_cards[i] - flush_cards[i+4] == 4:
                    # Si la plus haute carte est un As, c'est une quinte flush royale
                    if flush_cards[i] == 14 and flush_cards[i+4] == 10:
                        return (HandRank.ROYAL_FLUSH, [14])
                    # Sinon c'est une quinte flush normale
                    return (HandRank.STRAIGHT_FLUSH, [flush_cards[i]])
            
            # Vérifie la quinte flush basse (As-5)
            if set([14,2,3,4,5]).issubset(set(flush_cards)):
                return (HandRank.STRAIGHT_FLUSH, [5])
        
        # Compte les occurrences de chaque valeur
        value_counts = Counter(values)
        
        # Vérifie le carré
        if 4 in value_counts.values():
            quads = [v for v, count in value_counts.items() if count == 4][0]
            kicker = max(v for v in values if v != quads)
            return (HandRank.FOUR_OF_A_KIND, [quads, kicker])
        
        # Vérifie le full house
        if 3 in value_counts.values():
            trips = sorted([v for v, count in value_counts.items() if count >= 3], reverse=True)
            pairs = []
            for value, count in value_counts.items():
                if count >= 2:
                    if count >= 3 and value != trips[0]:
                        pairs.append(value)
                    elif count == 2:
                        pairs.append(value)
            
            if pairs:
                return (HandRank.FULL_HOUSE, [trips[0], max(pairs)])
        
        # Vérifie la couleur simple
        if flush_suit:
            flush_values = sorted([card[0] for card in cards if card[1] == flush_suit], reverse=True)
            return (HandRank.FLUSH, flush_values[:5])
        
        # Vérifie la quinte
        unique_values = sorted(set(values), reverse=True)
        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i+4] == 4:
                return (HandRank.STRAIGHT, [unique_values[i]])
        
        # Vérifie la quinte basse (As-5)
        if set([14,2,3,4,5]).issubset(set(values)):
            return (HandRank.STRAIGHT, [5])
        
        # Vérifie le brelan
        if 3 in value_counts.values():
            trips = max(v for v, count in value_counts.items() if count >= 3)
            kickers = sorted([v for v in values if v != trips], reverse=True)[:2]
            return (HandRank.THREE_OF_A_KIND, [trips] + kickers)
        
        # Vérifie la double paire
        pairs = sorted([v for v, count in value_counts.items() if count >= 2], reverse=True)
        if len(pairs) >= 2:
            kickers = [v for v in values if v not in pairs[:2]]
            kicker = max(kickers) if kickers else 0
            return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
        
        # Vérifie la paire simple
        if pairs:
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, [pairs[0]] + kickers)
        
        # Carte haute
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])
