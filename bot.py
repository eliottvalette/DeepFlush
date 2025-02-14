import numpy as np
import random
import itertools
from collections import Counter, namedtuple
from poker_game import PlayerAction

# Le bot à pour but de simuler un joueur de poker qui joue de manière "optimale".
# Il s'appuie sur une simulation Monte Carlo pour évaluer la force de sa main
# et prendre des décisions en fonction de l'état du jeu. 

class PokerBot:
    """
    Hardcoded Poker Bot that selects actions based on predefined rules.
    All training related methods simply pass.
    The decision process has been adapted for the new state representation (dimension 116).
    
    Personal cards are encoded in indices 0-9 (two cards × 5 dims each).
    Community cards are stored in indices 10-34 (5 cards × 5 dims each).
    The hand info (e.g. one-hot of hand rank, kicker, etc.) occupies indices 35-46.
    The phase is encoded at indices 47-51 (one-hot: 0=Preflop, 1=Flop, 2=Turn, 3=River, 4=Showdown).
    Player activity (6 dims) are from indices 65-70 and actions available (5 dims) from indices 77-81.
    """
    def __init__(self, name, show_cards=True):
        self.name = name
        self.show_cards = show_cards

    def get_action(self, state_seq, epsilon=0, valid_actions=None):
        """
        Uses a hardcoded rule–based decision to select an action.
        
        Args:
            state_seq (list): Sequence of state vectors, each of dimension 116.
            epsilon: (ignored) For compatibility with AI agents.
            valid_actions (iterable): Collection of valid PlayerAction values.
            
        Returns:
            tuple: (PlayerAction, action_mask) where:
                - PlayerAction is the chosen action enum
                - action_mask is a vector of 1s for valid actions, 0s for invalid
        """
        # We only need the most recent state
        state = state_seq[-1]
        
        # Ensure state is a numpy array
        state = np.array(state)
        
        # --- Extract personal cards (each card uses 5 dimensions) ---
        card1_enc = state[0:5]
        card2_enc = state[5:10]
        card1 = self.extract_card(card1_enc)
        card2 = self.extract_card(card2_enc)
        
        # --- Extract community cards (5 cards, each 5 dims) ---
        community_cards = []
        for i in range(5):
            start = 10 + i * 5
            end = start + 5
            card_enc = state[start:end]
            if np.all(card_enc == -1):  # All values are -1 -> card not dealt yet
                continue
            card = self.extract_card(card_enc)
            if card is not None:
                community_cards.append(card)
        
        # --- Determine game phase from indices 47:52 ---
        phase_vector = state[47:52]
        phase = int(np.argmax(phase_vector))  # 0=Preflop, 1=Flop, 2=Turn, 3=River, 4=Showdown
        
        # --- Count active players from indices 65:71 ---
        active_players = np.sum(state[65:71] == 1)
        num_opponents = max(active_players - 1, 0)
        
        # --- Evaluate hand strength ---
        if phase == 0:  # Preflop: use fast heuristic
            strength = self._evaluate_preflop_strength(card1, card2)
        else:
            # For postflop, use Monte Carlo simulation
            strength = self.evaluate_hand_strength([card1, card2], community_cards, num_opponents, num_simulations=50)
        
        # --- Get decision thresholds based on phase and number of opponents ---
        all_in_th, raise_th, call_th, check_th = self.get_thresholds(phase, num_opponents)
        
        # --- Decision logic ---
        chosen_index = 0  # Default: FOLD
        if strength >= all_in_th:
            chosen_index = 4  # ALL-IN
        elif strength >= raise_th and (valid_actions is None or PlayerAction.RAISE in valid_actions):
            chosen_index = 3  # RAISE
        elif strength >= call_th and (valid_actions is None or PlayerAction.CALL in valid_actions):
            chosen_index = 2  # CALL
        elif strength >= check_th and (valid_actions is None or PlayerAction.CHECK in valid_actions):
            chosen_index = 1  # CHECK
        else:
            chosen_index = 0  # FOLD

        # Map index to action
        action_map = {
            0: PlayerAction.FOLD,
            1: PlayerAction.CHECK,
            2: PlayerAction.CALL,
            3: PlayerAction.RAISE,
            4: PlayerAction.ALL_IN
        }
        
        chosen_action = action_map[chosen_index]
        
        # Ensure chosen action is valid
        if valid_actions is not None and chosen_action not in valid_actions:
            for action in valid_actions:
                chosen_action = action
                chosen_index = list(action_map.values()).index(action)
                break
        
        # Build action mask from valid_actions
        action_mask = np.zeros(5)
        if valid_actions:
            for action in valid_actions:
                idx = list(action_map.values()).index(action)
                action_mask[idx] = 1
        
        print(f"[{self.name}] Phase: {phase}, Hand strength: {strength:.3f}")
        print(f"Thresholds: ALL_IN:{all_in_th:.3f}, RAISE:{raise_th:.3f}, CALL:{call_th:.3f}, CHECK:{check_th:.3f}")
        print(f"Chosen action: {chosen_action.value}")
        
        return chosen_action, action_mask

    def train_model(self, *args, **kwargs):
        """
        Hardcoded bot does not train but returns empty metrics for compatibility.
        
        Returns:
            dict: Empty metrics dictionary with default values.
        """
        return {
            'entropy_loss': 0.0,
            'value_loss': 0.0,
            'invalid_action_penalty': 0.0,
            'std': 0.0,
            'learning_rate': 0.0,
            'loss': 0.0
        }

    def remember(self, *args, **kwargs):
        """Hardcoded bot does not store experiences."""
        pass

    def extract_card(self, card_encoding):
        """
        Extracts card value and suit from a 5-dimensional encoding.
        
        Args:
            card_encoding (array-like): A 5-element vector.
                Index 0: normalized value ((card.value - 2) / 14)
                Indices 1-4: one-hot encoding of suit (♠, ♥, ♦, ♣)
        
        Returns:
            tuple: (value, suit) where value is 2-14 and suit is 0-3,
                   or None if the card is missing/invalid
        """
        if np.all(card_encoding == -1):  # Card not dealt yet
            return None
        
        # Denormalize value: encoded = (value - 2) / 14
        # So value = encoded * 14 + 2
        value = int(round(card_encoding[0] * 14 + 2))
        
        # Extract suit from one-hot encoding
        suit_vector = card_encoding[1:5]
        if 1 in suit_vector:
            suit = int(np.where(suit_vector == 1)[0][0])
        else:
            return None
        
        return (value, suit)

    def format_cards(self, cards):
        """
        Formats a list of cards into a human-readable string.
        
        Args:
            cards (list of tuple): Each tuple is (value, suit).
            
        Returns:
            str: String with cards formatted (e.g. "A♠ K♥").
        """
        value_str = {11: "J", 12: "Q", 13: "K", 14: "A"}
        suit_str = {0: "♠", 1: "♥", 2: "♦", 3: "♣"}
        formatted = []
        for val, s in cards:
            v = value_str.get(val, str(val))
            s_char = suit_str.get(s, "?")
            formatted.append(f"{v}{s_char}")
        return " ".join(formatted)

    def _evaluate_preflop_strength(self, card1, card2):
        """
        Computes a simple heuristic for preflop strength.
        
        For a pair: bonus + (card value / 28), otherwise 
        average the card values (normalized) plus bonuses for suitedness and connectivity.
        
        Args:
            card1, card2: Each is a tuple (value, suit).
        
        Returns:
            float: Strength between 0 and 1.
        """
        if card1 is None or card2 is None:
            return 0.0
        if card1[0] == card2[0]:
            strength = 0.6 + card1[0] / 28.0
        else:
            base = (card1[0] + card2[0]) / 28.0
            if card1[1] == card2[1]:
                base += 0.1
            if abs(card1[0] - card2[0]) == 1:
                base += 0.05
            strength = base
        return min(strength, 1.0)

    def evaluate_5card_hand(self, hand):
        """
        Evaluates a 5-card hand and returns a tuple ranking.
        
        The tuple format allows for direct comparison between hands.
        Categories:
            8: Straight Flush
            7: Four of a Kind
            6: Full House
            5: Flush
            4: Straight
            3: Three of a Kind
            2: Two Pair
            1: One Pair
            0: High Card
        
        Args:
            hand (list of tuple): List of 5 cards.
        
        Returns:
            tuple: Hand ranking tuple.
        """
        values = sorted([v for v, s in hand], reverse=True)
        suits = [s for v, s in hand]
        counts = Counter(values)
        counts_values = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        is_flush = (len(set(suits)) == 1)
        unique_values = sorted(set(values), reverse=True)
        is_straight = False
        straight_high = None
        if len(unique_values) >= 5:
            if unique_values[0] - unique_values[-1] == 4 and len(unique_values) == 5:
                is_straight = True
                straight_high = unique_values[0]
            elif set([14, 2, 3, 4, 5]).issubset(set(values)):
                is_straight = True
                straight_high = 5
        if is_flush and is_straight:
            return (8, straight_high)
        if counts_values[0][1] == 4:
            kicker = [v for v in values if v != counts_values[0][0]][0]
            return (7, counts_values[0][0], kicker)
        if counts_values[0][1] == 3 and counts_values[1][1] >= 2:
            return (6, counts_values[0][0], counts_values[1][0])
        if is_flush:
            return (5, values)
        if is_straight:
            return (4, straight_high)
        if counts_values[0][1] == 3:
            kickers = sorted([v for v in values if v != counts_values[0][0]], reverse=True)
            return (3, counts_values[0][0], kickers)
        if counts_values[0][1] == 2 and counts_values[1][1] == 2:
            remaining = [v for v in values if v != counts_values[0][0] and v != counts_values[1][0]]
            kicker = max(remaining) if remaining else 0
            high_pair = max(counts_values[0][0], counts_values[1][0])
            low_pair = min(counts_values[0][0], counts_values[1][0])
            return (2, high_pair, low_pair, kicker)
        if counts_values[0][1] == 2:
            kickers = sorted([v for v in values if v != counts_values[0][0]], reverse=True)
            return (1, counts_values[0][0], kickers)
        return (0, values)

    def best_hand(self, cards):
        """
        From a list of cards, selects the best possible 5-card hand.
        
        Args:
            cards (list of tuple): List of cards.
        
        Returns:
            tuple: Best hand ranking tuple.
        """
        best = None
        for combo in itertools.combinations(cards, 5):
            rank = self.evaluate_5card_hand(list(combo))
            if best is None or rank > best:
                best = rank
        return best

    def simulate_hand(self, hole_cards, board_cards, num_opponents, num_simulations=100, return_board=False):
        """
        Simulates a complete hand by filling in unknown board cards and dealing random opponent hands.
        
        Args:
            hole_cards (list of tuple): Your hole cards.
            board_cards (list of tuple): Revealed community cards.
            num_opponents (int): Number of opponents.
            num_simulations (int): Number of simulations.
            return_board (bool): If True, also return the final board and opponents' hands.
        
        Returns:
            int: 1 for win, 0 for tie, -1 for loss.
            If return_board is True, also returns (full_board, opponents_hole_cards).
        """
        full_deck = [(v, s) for v in range(2, 15) for s in range(4)]
        known = set(hole_cards + board_cards)
        deck = [card for card in full_deck if card not in known]
        random.shuffle(deck)
        
        missing = 5 - len(board_cards)
        board_completion = deck[:missing]
        full_board = board_cards + board_completion
        deck = deck[missing:]
        
        opponents_best = []
        opponents_hole_cards = []
        for _ in range(num_opponents):
            opp_hole = deck[:2]
            deck = deck[2:]
            opponents_hole_cards.append(opp_hole)
            opp_hand = opp_hole + full_board
            opponents_best.append(self.best_hand(opp_hand))
        
        player_best = self.best_hand(hole_cards + full_board)
        tie_flag = False
        result = None
        for opp in opponents_best:
            if opp > player_best:
                result = -1
                break
            elif opp == player_best:
                tie_flag = True
        if result is None:
            result = 0 if tie_flag else 1
        if return_board:
            return result, full_board, opponents_hole_cards
        return result

    def monte_carlo_equity(self, hole_cards, board_cards, num_opponents, num_simulations=100):
        """
        Estimates the hand equity using Monte Carlo simulations.
        
        Args:
            hole_cards (list of tuple): Your hole cards.
            board_cards (list of tuple): Revealed community cards.
            num_opponents (int): Number of opponents.
            num_simulations (int): Number of simulations to run.
        
        Returns:
            float: Estimated equity (between 0 and 1).
        """
        wins, ties, total = 0, 0, 0
        for _ in range(num_simulations):
            result = self.simulate_hand(hole_cards, board_cards, num_opponents, num_simulations=1)
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            total += 1
        equity = (wins + 0.5 * ties) / total if total > 0 else 0.0
        return equity

    def evaluate_hand_strength(self, hole_cards, board_cards, num_opponents, num_simulations=100):
        """
        Computes the strength of your hand in the current situation via Monte Carlo simulation.
        
        Args:
            hole_cards (list of tuple): Your hole cards.
            board_cards (list of tuple): Revealed community cards.
            num_opponents (int): Number of opponents.
            num_simulations (int): Number of simulations.
        
        Returns:
            float: Hand equity between 0 and 1.
        """
        return self.monte_carlo_equity(hole_cards, board_cards, num_opponents, num_simulations)

    def get_thresholds(self, phase, num_opponents):
        """
        Returns threshold values for decision making based on the game phase and opponent count.
        
        For Preflop:
          - The more opponents, the more conservative thresholds become.
        For Postflop:
          - Very low thresholds are used in later phases.
        
        Args:
            phase (int): 0=Preflop, 1=Flop, 2=Turn, 3=River, 4=Showdown.
            num_opponents (int): Number of opponents.
        
        Returns:
            tuple: (all_in_threshold, raise_threshold, call_threshold, check_threshold)
        """
        effective = min(num_opponents, 5)
        if phase == 0:
            delta = (effective - 1) * 0.03
            base_all_in = 0.80
            base_raise  = 0.70
            base_call   = 0.60
            base_check  = 0.40
            final_all_in = min(base_all_in + delta, 1.0)
            final_raise  = min(base_raise + delta, 1.0)
            final_call   = min(base_call + delta, 1.0)
            final_check  = min(base_check + delta, 1.0)
        else:
            delta = (effective - 1) * 0.01
            if phase == 1:
                phase_offset = 0.0
            elif phase in [2, 3]:
                phase_offset = -0.03
            else:
                phase_offset = 0.0
            base_all_in = 0.45
            base_raise  = 0.35
            base_call   = 0.25
            base_check  = 0.15
            final_all_in = max(0.0, min(base_all_in - delta + phase_offset, 1.0))
            final_raise  = max(0.0, min(base_raise - delta + phase_offset, 1.0))
            final_call   = max(0.0, min(base_call - delta + phase_offset, 1.0))
            final_check  = max(0.0, min(base_check - delta + phase_offset, 1.0))
        return final_all_in, final_raise, final_call, final_check