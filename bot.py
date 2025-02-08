import numpy as np
import random
import itertools
from collections import Counter
from poker_game import PlayerAction

# Le bot à pour but de simuler un joueur de poker qui joue de manière "optimale".
# Il s'appuie sur une simulation Monte Carlo pour évaluer la force de sa main
# et prendre des décisions en fonction de l'état du jeu. 

def extract_card(card_encoding):
    """
    Extrait la valeur et la couleur d'une carte à partir de son encodage one-hot.
    
    Args:
        card_encoding (array-like): Tableau de taille 17 représentant une carte.
            - Les 13 premières positions (indices 0-12) correspondent à l'encodage one-hot de la valeur:
              [2, 3, 4, 5, 6, 7, 8, 9, 10, Valet, Dame, Roi, As]
            - Les 4 positions suivantes (indices 13-16) encodent la couleur:
              [♠, ♥, ♦, ♣]
            
    Exemple:
        [0,0,1,0,0,0,0,0,0,0,0,0,0, 1,0,0,0] -> (4, 0) représente le 4 de ♠
        [0,0,0,0,0,0,0,0,0,0,0,1,0, 0,0,1,0] -> (13, 2) représente le Roi de ♦
            
    Returns:
        tuple: (valeur, indice_couleur)
            - valeur (int): Entier entre 2 et 14 où:
                2-10 représentent leur valeur nominale
                11 = Valet, 12 = Dame, 13 = Roi, 14 = As
            - indice_couleur (int): 0=♠, 1=♥, 2=♦, 3=♣
            
        Si la carte n'existe pas (encodage nul), retourne (None, None)
    """
    card_encoding = np.array(card_encoding)
    if np.sum(card_encoding[:13]) == 0:
        return None, None
    value = int(np.argmax(card_encoding[:13]) + 2)
    suit = int(np.argmax(card_encoding[13:17]))
    return value, suit

def hardcoded_poker_bot(state, valid_actions=None):
    """
    Bot de poker basé sur des règles qui sélectionne une action en fonction de l'état du jeu.
    
    Args:
        state (array-like): Vecteur d'état contenant:
            - Indices 0-16: Encodage de la 1ère carte personnelle
            - Indices 17-33: Encodage de la 2ème carte personnelle  
            - Indices 34-118: Encodage des cartes communes (5 x 17 positions)
            - 4 derniers indices: Paramètres contextuels
                - phase (int): 0=preflop, 1=flop, 2=turn, 3=river
                - position (int): 0=précoce, 1=milieu, 2=tardive
                - stack_ratio (float): Ratio stack/pot (ex: 2.5 = stack 2.5x plus grand que le pot)
                - num_active_opponents (int): Nombre d'adversaires encore en jeu
        valid_actions (set): Set des actions valides pour ce joueur
    
    Returns:
        np.array: Vecteur one-hot de taille 5 indiquant l'action choisie:
            [1,0,0,0,0] -> FOLD  (se coucher)
            [0,1,0,0,0] -> CHECK (parole/suivre gratuit)
            [0,0,1,0,0] -> CALL  (suivre la mise)
            [0,0,0,1,0] -> RAISE (relancer)
            [0,0,0,0,1] -> ALL-IN (tout miser)
            
    Exemple:
        Pour un état avec:
        - Main: As♠ Roi♥
        - Flop: 7♣ 7♦ As♣
        - Position tardive
        - Stack ratio de 3.0
        - 2 adversaires
        
        Le bot pourrait choisir RAISE car:
        - Double paire forte (As et 7)
        - Position avantageuse
        - Stack confortable
        - Peu d'adversaires
    """
    state = np.array(state)
    # Si aucune action n'est autorisée, retourner None
    if valid_actions is not None and len(valid_actions) == 0:
         return None
    
    # Extraction des cartes personnelles (2 cartes)
    card1_enc = state[0:17]
    card2_enc = state[17:34]
    card1_value, card1_suit = extract_card(card1_enc)
    card2_value, card2_suit = extract_card(card2_enc)
    
    valeurs = []
    couleurs = []
    
    # Ajout des cartes personnelles
    if card1_value is not None:
        valeurs.append(card1_value)
        couleurs.append(card1_suit)
    if card2_value is not None:
        valeurs.append(card2_value)
        couleurs.append(card2_suit)
        
    # Extraction des cartes communes en fonction de la phase
    if len(state) >= (34 + 5*17 + 4):
        phase = int(state[-4])                # 0: preflop, 1: flop, 2: turn, 3: river
        position = int(state[-3])             # 0: précoce, 1: milieu, 2: tardive
        stack_ratio = state[-2]               # Ratio du stack par rapport à une référence (ex: pot)
        num_active_opponents = int(state[-1]) # Nombre d'adversaires encore actifs
    else:
        # Valeurs par défaut si les informations contextuelles ne sont pas fournies
        phase = 1
        position = 1
        stack_ratio = 1.0
        num_active_opponents = 5

    if phase == 0:
        n_comm = 0  # Preflop : aucune carte commune n'est distribuée
    elif phase == 1:
        n_comm = 3  # Flop
    elif phase == 2:
        n_comm = 4  # Flop + Turn
    elif phase == 3:
        n_comm = 5  # Flop + Turn + River
    else:
        n_comm = 0

    for i in range(n_comm):
        start_idx = 34 + i * 17
        end_idx = start_idx + 17
        if end_idx > len(state):
            break
        card_enc = state[start_idx:end_idx]
        if np.sum(card_enc[:13]) == 0:
            continue  # Carte non distribuée
        val, col = extract_card(card_enc)
        if val is not None:
            valeurs.append(val)
            couleurs.append(col)
    
    # Extraction des paramètres contextuels (si présents) à la fin du vecteur d'état :
    # On suppose que les 4 derniers éléments de l'état correspondent à :
    # [phase, position, stack_ratio, num_active_opponents]
    if len(state) >= (34 + 5*17 + 4):
        phase = int(state[-4])                # 0: preflop, 1: flop, 2: turn, 3: river
        position = int(state[-3])             # 0: précoce, 1: milieu, 2: tardive
        stack_ratio = state[-2]               # Ratio du stack par rapport à une référence (ex: pot)
        num_active_opponents = int(state[-1]) # Nombre d'adversaires encore actifs
    else:
        # Valeurs par défaut si les informations contextuelles ne sont pas fournies
        phase = 1
        position = 1
        stack_ratio = 1.0
        num_active_opponents = 5

    # Définir des seuils de décision de base en fonction de la phase du jeu
    if phase == 0:  # Preflop
        all_in_threshold = 0.90
        raise_threshold = 0.70
        call_threshold  = 0.50
    elif phase == 1:  # Flop
        all_in_threshold = 0.85
        raise_threshold = 0.65
        call_threshold  = 0.40
    elif phase == 2:  # Turn
        all_in_threshold = 0.80
        raise_threshold = 0.60
        call_threshold  = 0.35
    elif phase == 3:  # River
        all_in_threshold = 0.80
        raise_threshold = 0.60
        call_threshold  = 0.40
    else:
        all_in_threshold = 0.85
        raise_threshold = 0.65
        call_threshold  = 0.40

    # Ajustement des seuils selon d'autres paramètres contextuels :
    # - Si le bot est short-stacked, on baisse les seuils pour favoriser des mises plus agressives.
    if stack_ratio < 0.5:
        all_in_threshold -= 0.05
        raise_threshold -= 0.05
        call_threshold  -= 0.05

    # - Si le bot est en position tardive (avantage informationnel), il peut être plus agressif.
    if position == 2:
        raise_threshold -= 0.05
        call_threshold  -= 0.05
    # - En position précoce, il faut être plus prudent.
    elif position == 0:
        raise_threshold += 0.05
        call_threshold  += 0.05

    # - Si le nombre d'adversaires actifs est élevé, il faut être plus sélectif car la compétition est plus forte.
    if num_active_opponents > 4:
        all_in_threshold += 0.05
        raise_threshold += 0.05
        call_threshold  += 0.05

    print("\n=== Analyse du Bot ===")
    print(f"Phase: {phase} (0=preflop, 1=flop, 2=turn, 3=river)")
    print(f"Position: {position} (0=précoce, 1=milieu, 2=tardive)")
    print(f"Stack ratio: {stack_ratio:.2f}")
    print(f"Adversaires actifs: {num_active_opponents}")
    
    print("\nSeuils de décision:")
    print(f"- All-in : {all_in_threshold:.2f}")
    print(f"- Raise  : {raise_threshold:.2f}")
    print(f"- Call   : {call_threshold:.2f}")

    # Évaluation de la main
    strength = evaluate_hand_strength(valeurs, couleurs, (card1_value, card1_suit), (card2_value, card2_suit))
    print(f"\nForce de la main (Monte Carlo): {strength:.2f}")
    
    # Détermination de l'action
    if strength >= all_in_threshold:
        action_index = 4  # ALL-IN
        decision = "ALL-IN"
    elif strength >= raise_threshold:
        action_index = 3  # RAISE
        decision = "RAISE"
    elif strength >= call_threshold:
        action_index = 2  # CALL
        decision = "CALL"
    else:
        action_index = 0  # FOLD
        decision = "FOLD"
    
    print(f"Décision initiale: {decision}")

    # Vérification de la validité de l'action
    if valid_actions is not None:
        chosen_action = list(PlayerAction)[action_index]
        if chosen_action not in valid_actions:
            old_decision = decision
            if chosen_action == PlayerAction.CALL and PlayerAction.CHECK in valid_actions:
                action_index = list(PlayerAction).index(PlayerAction.CHECK)
                decision = "CHECK"
            else:
                for a in list(PlayerAction):
                    if a in valid_actions:
                        action_index = list(PlayerAction).index(a)
                        decision = a.name
                        break
            print(f"Action {old_decision} non valide, choix alternatif: {decision}")
            print(f"Actions valides: {[a.name for a in valid_actions]}")

    # Construction du vecteur one-hot correspondant à l'action choisie
    action_vector = np.zeros(5)
    action_vector[action_index] = 1
    return action_vector

# Nouvelle version de evaluate_hand_strength utilisant une simulation Monte Carlo
def evaluate_hand_strength(valeurs, couleurs, card1, card2, num_opponents=5, num_simulations=300):
    """
    Évalue la force d'une main par simulation Monte Carlo.
    
    Args:
        valeurs (list[int]): Liste des valeurs des cartes (2-14)
            Ex: [14, 13, 7, 7, 14] pour As-Roi + Flop: 7♣ 7♦ As♣
        couleurs (list[int]): Liste des indices de couleurs (0-3)
            Ex: [0, 1, 3, 2, 3] pour les couleurs correspondantes
        card1 (tuple): Première carte perso (valeur, couleur)
            Ex: (14, 0) pour As♠
        card2 (tuple): Deuxième carte perso (valeur, couleur)
            Ex: (13, 1) pour Roi♥
        num_opponents (int): Nombre d'adversaires à simuler (défaut: 5)
        num_simulations (int): Nombre de simulations Monte Carlo (défaut: 300)
    
    Returns:
        float: Équité de la main entre 0 et 1
            Ex: 0.75 signifie 75% de chances de gagner/égaliser
    """
    # Construction des cartes connues
    hole_cards = []
    if card1[0] is not None:
        hole_cards.append(card1)
    if card2[0] is not None:
        hole_cards.append(card2)
    
    board_cards = []
    # Les deux premières positions de 'valeurs' correspondent aux cartes personnelles.
    # Les cartes suivantes sont les community cards.
    if len(valeurs) > 2:
        for i in range(2, len(valeurs)):
            if valeurs[i] is not None:
                board_cards.append((valeurs[i], couleurs[i]))
                
    # Calcul de l'équité via simulation Monte Carlo
    equity = monte_carlo_equity(hole_cards, board_cards, num_opponents, num_simulations)
    return equity

def monte_carlo_equity(hole_cards, board_cards, num_opponents, num_simulations):
    """
    Calcule l'équité de la main via simulation Monte Carlo en simulant plusieurs scénarios de jeu.
    
    Args:
        hole_cards (list[tuple]): Liste des cartes personnelles
            Ex: [(14, 0), (13, 1)] pour As♠ Roi♥
        board_cards (list[tuple]): Liste des cartes communes déjà révélées
            Ex: [(7, 3), (7, 2), (14, 3)] pour 7♣ 7♦ As♣
        num_opponents (int): Nombre d'adversaires simulés
            Ex: 3 simulera 3 adversaires avec des mains aléatoires
        num_simulations (int): Nombre d'itérations de simulation
            Ex: 300 générera 300 scénarios différents
    
    Returns:
        float: Probabilité de gagner/égaliser entre 0 et 1
            Ex: 0.65 signifie que la main gagne ou égalise dans 65% des simulations
            
    Note:
        Plus le nombre de simulations est élevé, plus l'estimation est précise,
        mais plus le calcul est long. 300 simulations offrent un bon compromis.
    """
    wins = 0
    ties = 0
    total = 0
    
    print("\nDétails Monte Carlo:")
    print(f"- Cartes personnelles: {format_cards(hole_cards)}")
    print(f"- Cartes communes: {format_cards(board_cards)}")
    print(f"- Nombre d'adversaires: {num_opponents}")
    print(f"- Nombre de simulations: {num_simulations}")
    
    print_indices = list(range(50, num_simulations+1, 50))
    for sim in range(num_simulations):
        if (sim+1) in print_indices:
            sim_result, sim_board, sim_opponents = simulate_hand(hole_cards, board_cards, num_opponents, return_board=True)
            print(f"Simulation #{sim+1}: Cartes communes simulées: {format_cards(sim_board)}")
            for idx, opp_hole in enumerate(sim_opponents):
                print(f"  Adversaire #{idx+1}: {format_cards(opp_hole)}")
        else:
            sim_result = simulate_hand(hole_cards, board_cards, num_opponents)
        if sim_result == 1:
            wins += 1
        elif sim_result == 0:
            ties += 1
        total += 1
    
    equity = (wins + ties * 0.5) / total
    print(f"Résultats:")
    print(f"- Victoires: {wins} ({(wins/total)*100:.1f}%)")
    print(f"- Égalités: {ties} ({(ties/total)*100:.1f}%)")
    print(f"- Défaites: {total-wins-ties} ({((total-wins-ties)/total)*100:.1f}%)")
    print(f"- Équité totale: {equity:.3f}")
    
    return equity

# Fonction utilitaire pour formater l'affichage des cartes
def format_cards(cards):
    values = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    suits = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
    formatted = []
    for value, suit in cards:
        card_value = values.get(value, str(value))
        card_suit = suits.get(suit, '?')
        formatted.append(f"{card_value}{card_suit}")
    return ' '.join(formatted)

def simulate_hand(hole_cards, board_cards, num_opponents, opponent_range_matrix=None, return_board=False):
    """
    Simule une main complète en distribuant aléatoirement les cartes manquantes.
    
    Args:
        hole_cards (list[tuple]): Cartes personnelles du joueur, ex. [(14, 0), (13, 1)] pour As♠ Roi♥.
        board_cards (list[tuple]): Cartes communes déjà révélées, ex. [(7, 3)] pour 7♣ au flop.
        num_opponents (int): Nombre d'adversaires à simuler, ex. 2 pour simuler une table avec 3 joueurs.
        opponent_range_matrix (np.array, optionnel): Matrice 13x13 définissant la range des adversaires.
            Si fournie, les adversaires ne recevront que des mains correspondant à cette range.
        return_board (bool, optionnel): Si True, la fonction retourne également le board complet utilisé dans la simulation.
    
    Returns:
        int: Résultat de la simulation
            1  = Victoire
            0  = Égalité
           -1  = Défaite
    """
    # Construction d'un deck complet en excluant les cartes connues (hole cards et cartes communautaires réelles)
    full_deck = [(v, s) for v in range(2, 15) for s in range(4)]
    known_cards = set(hole_cards + board_cards)
    deck = [card for card in full_deck if card not in known_cards]
    
    # Mélanger le deck
    random.shuffle(deck)
    
    # Compléter le board s'il manque des cartes (le board doit avoir 5 cartes)
    missing_board = 5 - len(board_cards)
    board_completion = deck[:missing_board]
    full_board = board_cards + board_completion
    deck = deck[missing_board:]
    
    # Évaluer la main du joueur (cartes personnelles + board complet)
    player_hand = hole_cards + full_board
    player_best = best_hand(player_hand)
    
    opponents_best = []
    opponents_hole_cards = []  # Liste pour stocker les mains des adversaires
    # Distribution des cartes aux adversaires
    for opponent in range(num_opponents):
        if opponent_range_matrix is not None:
            allowed = get_allowed_adversary_hands(opponent_range_matrix, deck)
            if allowed:
                opp_hole = random.choice(allowed)
                # Retirer les cartes sélectionnées du deck
                deck.remove(opp_hole[0])
                deck.remove(opp_hole[1])
            else:
                opp_hole = deck[:2]
                deck = deck[2:]
        else:
            opp_hole = deck[:2]
            deck = deck[2:]
        opponents_hole_cards.append(opp_hole)
        opp_hand = opp_hole + full_board
        opponents_best.append(best_hand(opp_hand))
    
    # Comparaison : si un adversaire a une main meilleure, on perd.
    tie_flag = False
    result = None
    for opp in opponents_best:
        if opp > player_best:
            result = -1
            break
        elif opp == player_best:
            tie_flag = True
    if result is None:
        if tie_flag:
            result = 0
        else:
            result = 1
    if return_board:
        return result, full_board, opponents_hole_cards
    return result

def best_hand(cards):
    """
    Détermine la meilleure main de poker possible avec les cartes disponibles.
    
    Args:
        cards (list[tuple]): Liste de cartes (valeur, couleur)
            Ex: [(14,0), (13,1), (7,3), (7,2), (14,3), (10,1), (2,0)]
            pour As♠ Roi♥ + 7♣ 7♦ As♣ 10♥ 2♠
    
    Returns:
        tuple: Classement de la meilleure main possible
            Format: (catégorie, valeur1, valeur2, ...)
            Ex: (6, 14, 7) pour un full aux As par les 7
            
    Note:
        Teste toutes les combinaisons de 5 cartes possibles
        et retourne le classement de la meilleure main.
        L'ordre des valeurs dans le tuple permet la comparaison
        directe entre deux mains.
    """
    best = None
    for combo in itertools.combinations(cards, 5):
        rank = evaluate_5card_hand(list(combo))
        if best is None or rank > best:
            best = rank
    return best

def evaluate_5card_hand(hand):
    """
    Évalue une main de 5 cartes exactement et retourne son classement.
    
    Args:
        hand (list[tuple]): Liste de 5 cartes (valeur, couleur)
            Ex: [(14,0), (14,1), (7,3), (7,2), (10,1)]
            pour As♠ As♥ 7♣ 7♦ 10♥
    
    Returns:
        tuple: Classement détaillé de la main
            Format général: (catégorie, valeur1, valeur2, ...)
            
            Formats spécifiques par catégorie:
            8: Quinte Flush  -> (8, hauteur)
                Ex: (8, 14) pour une quinte flush à l'As
            7: Carré        -> (7, valeur_carré, kicker)
                Ex: (7, 7, 10) pour un carré de 7 avec un 10
            6: Full House   -> (6, valeur_brelan, valeur_paire)
                Ex: (6, 14, 7) pour un full aux As par les 7
            5: Couleur      -> (5, [valeurs_triées])
                Ex: (5, [14,10,8,7,2]) pour une couleur As-haut
            4: Suite        -> (4, hauteur)
                Ex: (4, 14) pour une suite à l'As
            3: Brelan       -> (3, valeur_brelan, [kickers])
                Ex: (3, 7, [14,10]) pour un brelan de 7 avec As-10
            2: Double Paire -> (2, haute_paire, basse_paire, kicker)
                Ex: (2, 14, 7, 10) pour As-As 7-7 avec un 10
            1: Paire        -> (1, valeur_paire, [kickers])
                Ex: (1, 14, [10,7,2]) pour une paire d'As
            0: Carte Haute  -> (0, [valeurs_triées])
                Ex: (0, [14,10,7,5,2]) pour As-haut
    """
    values = sorted([v for v, s in hand], reverse=True)
    suits = [s for v, s in hand]
    counts = Counter(values)
    counts_values = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = (len(set(suits)) == 1)
    
    # Détecter une suite
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
    
    # Quinte flush
    if is_flush and is_straight:
        return (8, straight_high)
    # Carré
    if counts_values[0][1] == 4:
        kicker = [v for v in values if v != counts_values[0][0]][0]
        return (7, counts_values[0][0], kicker)
    # Full house
    if counts_values[0][1] == 3 and counts_values[1][1] >= 2:
        return (6, counts_values[0][0], counts_values[1][0])
    # Couleur (flush)
    if is_flush:
        return (5, values)
    # Suite (straight)
    if is_straight:
        return (4, straight_high)
    # Brelan
    if counts_values[0][1] == 3:
        kickers = sorted([v for v in values if v != counts_values[0][0]], reverse=True)
        return (3, counts_values[0][0], kickers)
    # Double paire
    if counts_values[0][1] == 2 and counts_values[1][1] == 2:
        remaining = [v for v in values if v != counts_values[0][0] and v != counts_values[1][0]]
        kicker = max(remaining) if remaining else 0
        high_pair = max(counts_values[0][0], counts_values[1][0])
        low_pair = min(counts_values[0][0], counts_values[1][0])
        return (2, high_pair, low_pair, kicker)
    # Paire
    if counts_values[0][1] == 2:
        kickers = sorted([v for v in values if v != counts_values[0][0]], reverse=True)
        return (1, counts_values[0][0], kickers)
    # Carte haute
    return (0, values)

# ----------------------------------------------------------------
# Section : Gestion des ranges pour la simulation Monte Carlo
# ----------------------------------------------------------------

def get_player_range(state, player_id):
    """
    Retourne la matrice de range 13x13 pour un joueur donné basée sur l'état de la partie.
    
    Args:
        state (dict): État de la partie, qui peut contenir les ranges définies pour chaque joueur.
        player_id (int): Identifiant du joueur.
    
    Returns:
        np.array: Une matrice 13x13 (avec des entiers 0 ou 1) où chaque cellule vaut 1
                  si la main correspondant aux deux rangs (de 2 à As) est dans la range du joueur,
                  sinon 0.
    
    Exemple:
        Si le joueur a une range très large, la fonction renverra une matrice entièrement composée de 1.
    """
    if 'player_actions' in state and player_id in state['player_actions']:
        actions = state['player_actions'][player_id]
        # Priorité des actions : si le joueur a fold, sa range est vide,
        # sinon si raise, on retourne une range restreinte (mains fortes),
        # sinon si call, on retourne une range légèrement plus large.
        if 'fold' in actions:
            return np.zeros((13, 13), dtype=int)
        elif 'raise' in actions:
            # Exemple : pour un raise, on autorise uniquement :
            # - les paires d'au moins 8 (indice >= 6 ; car 2->0,3->1,...,8->6)
            # - les combinations offsuit où la carte haute est d'au moins 10 (indice>=8)
            range_matrix = np.zeros((13, 13), dtype=int)
            for i in range(13):
                for j in range(i+1):
                    if i == j:
                        if i >= 6:
                            range_matrix[i][j] = 1
                    else:
                        if i >= 8:
                            range_matrix[i][j] = 1
            return range_matrix
        elif 'call' in actions:
            # Pour un call, on autorise presque toutes les mains, sauf les très faibles.
            range_matrix = np.ones((13, 13), dtype=int)
            for i in range(13):
                for j in range(i+1):
                    if i == j and i < 2:
                        range_matrix[i][j] = 0
            return range_matrix
    # Par défaut, si aucune action n'est enregistrée, on suppose que toutes les mains sont possibles.
    return np.ones((13, 13), dtype=int)


def get_allowed_adversary_hands(range_matrix, deck):
    """
    Filtre les combinaisons de deux cartes disponibles dans le deck selon la range (matrice 13x13).
    
    Args:
        range_matrix (np.array): Matrice 13x13 avec des 1 pour les mains autorisées.
            Les indices 0 à 12 correspondent aux rangs de 2 à As 
            (indice 0 = 2, indice 12 = As).
        deck (list of tuple): Liste des cartes disponibles dans le deck, chaque carte étant un tuple (valeur, couleur)
            avec valeur comprise entre 2 et 14.
    
    Returns:
        list: Liste de combinaisons (tuple de 2 cartes) présentes dans le deck qui correspondent à une main dans la range.
    
    Note:
        Pour chaque combinaison de deux cartes du deck, on définit une représentation canonique de la main comme suit :
            i = max(c1[0], c2[0]) - 2
            j = min(c1[0], c2[0]) - 2
        Si range_matrix[i][j] == 1, la combinaison est autorisée.
    """
    allowed_hands = []
    for combo in itertools.combinations(deck, 2):
        c1, c2 = combo
        i = max(c1[0], c2[0]) - 2  # Car 2 correspond à l'indice 0
        j = min(c1[0], c2[0]) - 2
        if range_matrix[i][j] == 1:
            allowed_hands.append(combo)
    return allowed_hands