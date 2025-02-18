#!/usr/bin/env python
"""
montecarlo.py

Ce script réalise une expérience Monte Carlo :
 - Il distribue aléatoirement 2 cartes à 6 joueurs et 5 cartes communes.
 - Pour chaque joueur, on évalue la force de sa main (7 cartes combinées)
   à l'aide d'un évaluateur simplifié.
 - On détermine le(s) gagnant(s) et on attribue à chaque joueur vainqueur
   un gain fractionné en cas d'égalité.
 - Pour chacune des mains de départ (définies ici par les deux cartes, sans
   distinguer leur couleur), on enregistre le nombre d'occurrences et les
   victoires obtenues dans un tableau 13×13 (où les lignes/colonnes correspondent
   aux rangs de A à 2).
 - Finalement, on affiche la heatmap du pourcentage de victoire par main de départ.
"""

import random
import itertools
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os  # Assurez-vous que cet import figure en début de fichier

def create_deck():
    """
    Crée un paquet de 52 cartes.
    Chaque carte est représentée par un tuple (valeur, couleur),
    valeur de 2 à 14 (14 pour As) et couleur parmi ♠, ♥, ♦, ♣.
    """
    suits = ["♠", "♥", "♦", "♣"]
    deck = [(rank, suit) for rank in range(2, 15) for suit in suits]
    return deck

def evaluate_hand(cards):
    """
    Évalue une main de 7 cartes et retourne un tuple représentant la force.
    
    La fonction utilise un évaluateur simplifié qui distingue :
      - Straight flush (8)
      - Carré (7)
      - Full house (6)
      - Couleur (5)
      - Quinte (4)
      - Brelan (3)
      - Deux paires (2)
      - Paire (1)
      - Carte haute (0)
      
    Le tuple retourné est construit de façon à pouvoir être comparé (plus grand -> meilleure main).
    """
    ranks = [rank for rank, suit in cards]
    suits = [suit for rank, suit in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    # Vérifier la couleur
    is_flush = any(count >= 5 for count in suit_counts.values())
    flush_suit = None
    if is_flush:
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
    sorted_ranks = sorted(set(ranks), reverse=True)
    
    def check_straight(ranks_list):
        """Détecte une suite de 5 cartes consécutives."""
        if len(ranks_list) < 5:
            return None
        for i in range(len(ranks_list) - 4):
            if ranks_list[i] - ranks_list[i+4] == 4:
                return ranks_list[i]
        # Vérifier la quinte "roue" (A-2-3-4-5)
        if set([14, 5, 4, 3, 2]).issubset(ranks_list):
            return 5
        return None

    # Quinte flush
    straight_flush = None
    if flush_suit:
        flush_cards = [rank for rank, suit in cards if suit == flush_suit]
        flush_unique = sorted(set(flush_cards), reverse=True)
        straight_flush = check_straight(flush_unique)
    
    straight_high = check_straight(sorted_ranks)
    
    # Classification des mains
    if straight_flush:
        return (8, straight_flush)
    elif 4 in rank_counts.values():
        for rank, count in rank_counts.items():
            if count == 4:
                quad = rank
                break
        remaining = max([r for r in ranks if r != quad])
        return (7, quad, remaining)
    elif (3 in rank_counts.values()) and (2 in rank_counts.values() or list(rank_counts.values()).count(3) > 1):
        three = max([rank for rank, count in rank_counts.items() if count >= 3])
        pairs = [rank for rank, count in rank_counts.items() if count >= 2 and rank != three]
        pair = max(pairs) if pairs else three
        return (6, three, pair)
    elif is_flush:
        flush_cards = sorted([rank for rank, suit in cards if suit == flush_suit], reverse=True)[:5]
        return (5, flush_cards)
    elif straight_high:
        return (4, straight_high)
    elif 3 in rank_counts.values():
        three = max([rank for rank, count in rank_counts.items() if count == 3])
        kickers = sorted([r for r in ranks if r != three], reverse=True)[:2]
        return (3, three, kickers)
    elif list(rank_counts.values()).count(2) >= 2:
        pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)[:2]
        remaining = max([r for r in ranks if r not in pairs])
        return (2, pairs, remaining)
    elif 2 in rank_counts.values():
        pair = max([rank for rank, count in rank_counts.items() if count == 2])
        kickers = sorted([r for r in ranks if r != pair], reverse=True)[:3]
        return (1, pair, kickers)
    else:
        return (0, sorted_ranks[:5])

def simulate_hand(num_players):
    """
    Simule une main complète pour num_players :
      - Mélange le paquet, distribue 2 cartes à num_players joueurs et 5 cartes communes.
      - Évalue la main de chaque joueur (7 cartes : main + community)
      - Détermine le(s) gagnant(s) et attribue à chacun une part égale en cas d'égalité.
    
    Retourne :
        - players_hole : la liste des num_players mains (2 cartes chacune)
        - players_best : la liste des évaluations de main pour chaque joueur
        - winners      : liste des indices des joueurs gagnants
        - win_fraction : la part de victoire pour chaque gagnant.
    """
    deck = create_deck()
    random.shuffle(deck)
    # Distribuer 2 cartes par joueur
    players_hole = [[deck.pop(), deck.pop()] for _ in range(num_players)]
    # Distribuer 5 cartes communes
    community = [deck.pop() for _ in range(5)]
    # Évaluer la main de chaque joueur (7 cartes : main + community)
    players_best = []
    for hole in players_hole:
        seven_cards = hole + community
        players_best.append(evaluate_hand(seven_cards))
    best = max(players_best)
    winners = [i for i, hand in enumerate(players_best) if hand == best]
    win_fraction = 1.0 / len(winners)
    return players_hole, players_best, winners, win_fraction

def rank_to_index(rank):
    """
    Mappe une valeur de carte à un indice pour le tableau.
    Pour qu'As (14) => 0, Roi (13) => 1, …, 2 => 12.
    """
    return 14 - rank

def card_to_hand_key(card1, card2):
    """
    Détermine la clé d'une main de départ.
    La clé est un tuple (carte-haute, carte-basse) en se basant uniquement sur la valeur.
    """
    r1, _ = card1
    r2, _ = card2
    high = max(r1, r2)
    low = min(r1, r2)
    return (high, low)

def hand_key_to_name(hand_key):
    """
    Transforme une clé (valeur haute, valeur basse) en notation commune.
    Exemple : (14, 14) -> "AA", (14, 13) -> "AK"
    """
    rank_symbols = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T', 9: '9', 8: '8',
                    7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
    high, low = hand_key
    return f"{rank_symbols[high]}{rank_symbols[low]}"

def run_experiment(num_simulations=10000, num_players=6):
    """
    Exécute de nombreuses simulations et met à jour deux matrices 13×13 :
      - counts_matrix : nombre d'occurrences pour chaque main de départ
      - wins_matrix   : nombre total de victoires (fractionnaires en cas d'égalité)
    
    De plus, collecte les statistiques pour les hand ranks :
      - hand_rank_counts : nombre d'occurrences pour chaque hand rank
      - hand_rank_wins   : nombre total de victoires (fractionnaires) pour chaque hand rank
    
    La matrice counts_matrix est indexée de manière à ce que la ligne 0 corresponde à l'As
    et la ligne 12 au 2.
    """
    # Création des matrices pour les mains suited et offsuit
    suited_counts = np.zeros((13, 13))
    suited_wins   = np.zeros((13, 13))
    offsuit_counts = np.zeros((13, 13))
    offsuit_wins   = np.zeros((13, 13))

    # Initialiser les statistiques pour les hand ranks (indices 0 à 8)
    hand_rank_wins   = np.zeros(9)
    hand_rank_counts = np.zeros(9)

    for sim in range(num_simulations):
        players_hole, players_best, winners, win_fraction = simulate_hand(num_players)
        for i, hole in enumerate(players_hole):
            r1, s1 = hole[0]
            r2, s2 = hole[1]
            if r1 == r2:
                # Les paires : elles vont sur la diagonale
                idx = rank_to_index(r1)
                suited_counts[idx, idx] += 1
                if i in winners:
                    suited_wins[idx, idx] += win_fraction
            else:
                if s1 == s2:
                    # Mains suited : placer dans l'upper triangle
                    high = max(r1, r2)
                    low  = min(r1, r2)
                    row = rank_to_index(high)
                    col = rank_to_index(low)
                    suited_counts[row, col] += 1
                    if i in winners:
                        suited_wins[row, col] += win_fraction
                else:
                    # Mains offsuit : placer dans le lower triangle
                    high = max(r1, r2)
                    low  = min(r1, r2)
                    row = rank_to_index(low)
                    col = rank_to_index(high)
                    offsuit_counts[row, col] += 1
                    if i in winners:
                        offsuit_wins[row, col] += win_fraction

        # Mise à jour des statistiques par hand rank
        for i, best_hand in enumerate(players_best):
            rank_val = best_hand[0]  # Le premier élément représente le type de main (0 à 8)
            hand_rank_counts[rank_val] += 1
            if i in winners:
                hand_rank_wins[rank_val] += win_fraction

    # Combinaison des matrices suited et offsuit dans une matrice agrégée
    aggregate_win_rates = np.zeros((13,13))
    aggregate_counts    = np.zeros((13,13))
    for i in range(13):
        for j in range(13):
            if i == j:
                aggregate_counts[i, j] = suited_counts[i, j]
                if aggregate_counts[i, j] > 0:
                    aggregate_win_rates[i, j] = suited_wins[i, j] / aggregate_counts[i, j]
            elif i < j:
                # Upper triangle : mains suited
                aggregate_counts[i, j] = suited_counts[i, j]
                if aggregate_counts[i, j] > 0:
                    aggregate_win_rates[i, j] = suited_wins[i, j] / aggregate_counts[i, j]
            else:
                # Lower triangle : mains offsuit
                aggregate_counts[i, j] = offsuit_counts[i, j]
                if aggregate_counts[i, j] > 0:
                    aggregate_win_rates[i, j] = offsuit_wins[i, j] / aggregate_counts[i, j]

    with np.errstate(divide='ignore', invalid='ignore'):
        hand_rank_win_rates = np.where(hand_rank_counts > 0, hand_rank_wins / hand_rank_counts, 0)
    return aggregate_win_rates, aggregate_counts, hand_rank_win_rates, hand_rank_counts

def plot_win_rates(win_rates):
    """
    Affiche le tableau 13×13 des pourcentages de victoire en heatmap,
    avec les noms de main (ex: AA, AK, …) et leur proportion de victoire.
    """
    # Labels pour l'axe (As en haut, 2 en bas)
    labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # Afficher l'image (origin='upper' pour que la première ligne corresponde à A)
    im = ax.imshow(win_rates, cmap='viridis', origin='upper')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Carte Basse")
    ax.set_ylabel("Carte Haute")
    ax.set_title("Proportion de victoire par main de départ (simulée)")
    
    # Ajouter les annotations sur chaque cellule
    for i in range(len(labels)):
        for j in range(len(labels)):
            # On récupère la proportion (si aucune main simulée, la cellule restera vide)
            value = win_rates[i, j]
            if value > 0:
                text = f"{value*100:.1f}%"
            else:
                text = ""
            # Récupérer les valeurs de cartes associées (inversion de l'indice)
            high = 14 - i
            low = 14 - j
            hand_name = hand_key_to_name((high, low))
            # Afficher d'abord le pourcentage
            ax.text(j, i, text, ha="center", va="center", color="w", fontsize=8)
            # Afficher en dessous le nom de la main (décalage vertical)
            ax.text(j, i + 0.3, hand_name, ha="center", va="center", color="w", fontsize=8)
    
    fig.colorbar(im, ax=ax)
    plt.show()

def plot_simulation_results(win_rates, hand_rank_win_rates, num_players):
    """
    Crée une figure avec deux subplots :
      - À gauche, une heatmap présentant la proportion de victoire par main de départ.
      - À droite, un bar plot présentant le win rate par hand rank.
    La figure est ensuite sauvegardée au format JPG dans le dossier 'viz_pdf'.
    """
    # S'assurer que le dossier 'viz_pdf' existe
    if not os.path.exists("viz_pdf"):
        os.makedirs("viz_pdf")
    
    # Sous-figure pour la heatmap des mains de départ
    labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    im = ax1.imshow(win_rates, cmap='viridis', origin='upper')
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Carte Basse")
    ax1.set_ylabel("Carte Haute")
    ax1.set_title(f"Proportion de victoire par main de départ (simulée) - Table à {num_players} joueurs")
    # Ajout des annotations pour la heatmap
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = win_rates[i, j]
            text = f"{value*100:.1f}%" if value > 0 else ""
            # Pour les cellules, on doit déterminer correctement l'ordre des cartes :
            if i < j:
                # Upper triangle : mains suited, l'ordre est correct
                high_val = 14 - i
                low_val = 14 - j
                suffix = "s"
            elif i > j:
                # Lower triangle : mains offsuited -> inverser l'ordre pour afficher high-low
                high_val = 14 - j
                low_val = 14 - i
                suffix = "o"
            else:
                high_val = 14 - i
                low_val = 14 - j
                suffix = ""
            hand_name = hand_key_to_name((high_val, low_val)) + suffix
            ax1.text(j, i, text, ha="center", va="center", color="w", fontsize=8)
            ax1.text(j, i + 0.3, hand_name, ha="center", va="center", color="w", fontsize=10)
    fig.colorbar(im, ax=ax1)
    
    # Sous-figure pour le bar plot des win rates par hand rank
    hand_rank_names = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
    x = np.arange(len(hand_rank_names))
    win_rates_percentage = hand_rank_win_rates * 100
    bars = ax2.bar(x, win_rates_percentage, color='skyblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(hand_rank_names, rotation=45, ha='right')
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title(f"Win Rate par Hand Rank (simulée) - Table à {num_players} joueurs")
    ax2.set_ylim(0, 100)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f"{height:.1f}%", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    # Sauvegarde de la figure en JPG dans le dossier 'viz_pdf'
    plt.savefig(f"viz_pdf/simulation_results_{num_players}p.jpg", format="jpg")
    plt.close(fig)

def main():
    num_simulations = 10_000_000  # Nombre d'épisodes pour chacune des simulations
    for num in [6, 5, 4, 3, 2]:
        print(f"Simulation pour une table à {num} joueurs :")
        win_rates, counts_matrix, hand_rank_win_rates, hand_rank_counts = run_experiment(num_simulations, num)
        print(f"Nombre de simulations : {num_simulations} pour {num} joueurs")
        print("Proportion de victoire par main (matrice 13×13) :")
        print(win_rates)
        print("Win rates par Hand Rank :")
        print(hand_rank_win_rates)
        plot_simulation_results(win_rates, hand_rank_win_rates, num)

if __name__ == "__main__":
    main() 