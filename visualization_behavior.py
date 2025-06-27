"""
print('\n[GAME] === État actuel du jeu ===')

# Affichage complet du state pour vérifier toutes les informations (par ex. si les cartes sont suited ou non)
print("\n[GAME] [DEBUG] Contenu complet du state:")
print("[GAME]", state.tolist())
print('[GAME] Longueur totale :', len(state))

# 1. Cartes du joueur
print("\n[GAME] 1. Cartes du joueur:")
cards_suits = []
for i in range(2):
    base_idx = i * 5
    valeur = state[base_idx] * 14 + 2
    suit_vector = state[base_idx+1:base_idx+5].tolist()
    couleur_idx = suit_vector.index(1) if 1 in suit_vector else -1
    couleur = ["♠", "♥", "♦", "♣"][couleur_idx] if couleur_idx != -1 else "?"
    cards_suits.append(couleur)
    print(f"[GAME] Carte {i+1}: {int(valeur)}{couleur}")

# Vérification de si les cartes sont suited
if len(cards_suits) == 2:
    if cards_suits[0] == cards_suits[1]:
        print("[GAME] => Les cartes du joueur sont SUITED.")
    else:
        print("[GAME] => Les cartes du joueur ne sont PAS suited.")
else:
    print("[GAME] => Informations insuffisantes pour déterminer le suited.")

# 2. Cartes communes
print("\n[GAME] 2. Cartes communes:")
for i in range(5):
    base_idx = 10 + (i * 5)
    if state[base_idx] != -1:
        valeur = state[base_idx] * 14 + 2
        suit_vector = state[base_idx+1:base_idx+5].tolist()
        couleur_idx = suit_vector.index(1) if 1 in suit_vector else -1
        couleur = ["♠", "♥", "♦", "♣"][couleur_idx] if couleur_idx != -1 else "?"
        print(f"[GAME] Carte {i+1}: {int(valeur)}{couleur}")

# 3. Information sur la main
print("\n[GAME] 3. Information sur la main:")
hand_rank_idx = state[35:45].tolist().index(1) if 1 in state[35:45].tolist() else -1
print(f"[GAME] Rang de la main: {HandRank(hand_rank_idx).name if hand_rank_idx != -1 else 'Inconnu'}")
print(f"[GAME] Kicker: {int(state[45] * 13 + 2)}")
print(f"[GAME] Rang normalisé: {state[46]:.2f}")

# 4. Phase de jeu
print("\n[GAME] 4. Phase de jeu:")
phase_idx = state[47:52].tolist().index(1) if 1 in state[47:52].tolist() else -1
phases = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
print(f"[GAME] Phase actuelle: {phases[phase_idx] if phase_idx != -1 else 'Inconnu'}")

# 5-8. Informations sur les joueurs
print("\n[GAME] 5-8. Informations sur les joueurs:")
for i in range(3):
    print(f"\n[GAME] Joueur {i+1}:")
    print(f"[GAME] Stack: {state[53+i]:.2f}")
    print(f"[GAME] Mise actuelle: {state[56+i]:.3f}")
    print(f"[GAME] État: {'Actif' if state[59+i] == 1 else 'Inactif/Fold'}")

# 9. Position relative
print("\n[GAME] 9. Position relative:")
pos_idx = state[62:65].tolist().index(1) if 1 in state[62:65].tolist() else -1
positions = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
print(f"[GAME] Position: {positions[pos_idx] if pos_idx != -1 else 'Inconnu'}")

# 10. Actions disponibles
print("\n[GAME] 10. Actions disponibles:")
actions = ["FOLD", "CHECK", "CALL", "RAISE", "raise-25%", "raise-50%", "raise-75%", "raise-100%", "raise-150%", "raise-200%", "raise-300%", "ALL_IN"]
for i, action in enumerate(actions):
    print(f"[GAME] {action}: {'Disponible' if state[65+i] == 1 else 'Indisponible'}")

# 11. Historique des actions
print("\n[GAME] 11. Historique des dernières actions:")
for i in range(3):
    base_idx = 77 + (i * 5)
    action_vector = state[base_idx:base_idx+5]
    action_idx = action_vector.tolist().index(1) if 1 in action_vector.tolist() else -1
    if action_idx != -1:
        print(f"[GAME] Joueur {i+1}: {actions[action_idx]}")
    else:
        print(f"[GAME] Joueur {i+1}: Aucune action")

# 12. Informations stratégiques
print("\n[GAME] 12. Informations stratégiques:")
print(f"[GAME] Probabilité de victoire préflop: {state[92]:.3f}")
print(f"[GAME] Cotes du pot: {state[93]:.3f}")

print('\n[GAME] === Fin de l\'état ===\n')
"""
import os

import numpy as np
import random as rd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from utils.config import STATE_SIZE, GAMMA, ALPHA
from poker_agents import PokerAgent
from poker_game_expresso import Card, PokerGame, HandRank

from typing import List

# --------------------------------------------------
# Debug – impression complète de l'état
# --------------------------------------------------

def debug_full_state(state: np.ndarray):
    """Imprime exhaustivement le vecteur d'état au format lisible."""
    print('\n[GAME] === État actuel du jeu ===')

    # Affichage complet du state
    print("\n[GAME] [DEBUG] Contenu complet du state:")
    print("[GAME]", state.tolist())
    print('[GAME] Longueur totale :', len(state))

    # 1. Cartes du joueur
    print("\n[GAME] 1. Cartes du joueur:")
    cards_suits: List[str] = []
    for i in range(2):
        base_idx = i * 5
        valeur = state[base_idx] * 14 + 2
        suit_vector = state[base_idx+1:base_idx+5].tolist()
        couleur_idx = suit_vector.index(1) if 1 in suit_vector else -1
        couleur = ["♠", "♥", "♦", "♣"][couleur_idx] if couleur_idx != -1 else "?"
        cards_suits.append(couleur)
        print(f"[GAME] Carte {i+1}: {int(valeur)}{couleur}")

    if len(cards_suits) == 2:
        if cards_suits[0] == cards_suits[1]:
            print("[GAME] => Les cartes du joueur sont SUITED.")
        else:
            print("[GAME] => Les cartes du joueur ne sont PAS suited.")

    # 2. Cartes communes
    print("\n[GAME] 2. Cartes communes:")
    for i in range(5):
        base_idx = 10 + (i * 5)
        if state[base_idx] != -1:
            valeur = state[base_idx] * 14 + 2
            suit_vector = state[base_idx+1:base_idx+5].tolist()
            couleur_idx = suit_vector.index(1) if 1 in suit_vector else -1
            couleur = ["♠", "♥", "♦", "♣"][couleur_idx] if couleur_idx != -1 else "?"
            print(f"[GAME] Carte {i+1}: {int(valeur)}{couleur}")

    # 3. Information sur la main
    print("\n[GAME] 3. Information sur la main:")
    hand_rank_idx = state[35:45].tolist().index(1) if 1 in state[35:45].tolist() else -1
    from poker_game_expresso import HandRank  # import tardif pour éviter cycles
    print(f"[GAME] Rang de la main: {HandRank(hand_rank_idx).name if hand_rank_idx != -1 else 'Inconnu'}")
    print(f"[GAME] Kicker: {int(state[45] * 13 + 2)}")
    print(f"[GAME] Rang normalisé: {state[46]:.2f}")

    # 4. Phase de jeu
    print("\n[GAME] 4. Phase de jeu:")
    phase_idx = state[47:52].tolist().index(1) if 1 in state[47:52].tolist() else -1
    phases = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
    print(f"[GAME] Phase actuelle: {phases[phase_idx] if phase_idx != -1 else 'Inconnu'}")

    # 5-8. Informations sur les joueurs
    print("\n[GAME] 5-8. Informations sur les joueurs:")
    for i in range(3):
        print(f"\n[GAME] Joueur {i+1}:")
        print(f"[GAME] Stack: {state[53+i]:.2f}")
        print(f"[GAME] Mise actuelle: {state[56+i]:.3f}")
        print(f"[GAME] État: {'Actif' if state[59+i] == 1 else 'Inactif/Fold'}")

    # 9. Position relative
    print("\n[GAME] 9. Position relative:")
    pos_idx = state[62:65].tolist().index(1) if 1 in state[62:65].tolist() else -1
    positions = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
    print(f"[GAME] Position: {positions[pos_idx] if pos_idx != -1 else 'Inconnu'}")

    # 10. Actions disponibles
    print("\n[GAME] 10. Actions disponibles:")
    actions = ["FOLD", "CHECK", "CALL", "RAISE", "raise-25%", "raise-50%", "raise-75%", "raise-100%", "raise-150%", "raise-200%", "raise-300%", "ALL_IN"]
    for i, action in enumerate(actions):
        print(f"[GAME] {action}: {'Disponible' if state[65+i] == 1 else 'Indisponible'}")

    # 11. Historique des actions
    print("\n[GAME] 11. Historique des dernières actions:")
    for i in range(3):
        base_idx = 77 + (i * 5)
        action_vector = state[base_idx:base_idx+5]
        action_idx = action_vector.tolist().index(1) if 1 in action_vector.tolist() else -1
        if action_idx != -1:
            print(f"[GAME] Joueur {i+1}: {actions[action_idx]}")
        else:
            print(f"[GAME] Joueur {i+1}: Aucune action")

    # 12. Informations stratégiques
    print("\n[GAME] 12. Informations stratégiques:")
    print(f"[GAME] Probabilité de victoire préflop: {state[92]:.3f}")
    print(f"[GAME] Cotes du pot: {state[93]:.3f}")

    print('\n[GAME] === Fin de l\'état ===\n')

# --------------------------------------------------
# Helper utilities
# --------------------------------------------------

def critic_value(agent: PokerAgent, state_vec: np.ndarray) -> float:
    """Run the critic model to obtain V(s) for a single state vector."""
    # Build sequence batch (batch=1, seq_len=10, feat=STATE_SIZE)
    seq = np.zeros((1, 10, STATE_SIZE), dtype=np.float32)
    seq[0, -1, :] = state_vec  # place current state at last position
    tensor = torch.from_numpy(seq).float().to(agent.device)

    agent.critic_model.eval()
    with torch.no_grad():
        _, V = agent.critic_model(tensor)  # returns Q, V
    return V.item()

def rank_label(v: int) -> str:
    mapping = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T"}
    return mapping.get(v, str(v))

def build_state_via_env(env: PokerGame, ranks: tuple[int, int], suited: bool, order: str = "ab") -> np.ndarray:
    """Reset l'environnement, attribue la main voulue au joueur 0 et renvoie le vecteur d'état.
    order peut être "ab" (card1=ranks[0]) ou "ba".
    """
    env.reset()
    # Choix des couleurs
    suit_a = "♥"
    suit_b = suit_a if suited else "♠"
    v1, v2 = ranks if order == "ab" else (ranks[1], ranks[0])
    env.players[0].cards = [Card(suit_a, v1), Card(suit_b, v2)]
    # Mettre à jour le state vector (phase reste preflop, rien d'autre à changer)
    return env.get_state(seat_position=0)

# --------------------------------------------------
# Main routine
# --------------------------------------------------

def main(model_path: str):
    device = torch.device("cpu")

    # Charger l'agent (même modèle que durant l'entraînement)
    agent = PokerAgent(state_size=STATE_SIZE,
                       action_size=12,
                       gamma=GAMMA,
                       learning_rate=ALPHA,
                       device=device,
                       load_model=True,
                       load_path=model_path)

    # Préparer un environnement 3-max expresso
    agent_list = [agent, agent, agent]
    env = PokerGame(agents=agent_list, rendering=False)

    ranks = list(range(14, 1, -1))
    size = len(ranks)
    forward_matrix = np.zeros((size, size))
    reverse_matrix = np.zeros((size, size))
    diff_matrix = np.zeros((size, size))

    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            # Suited dans triangle supérieur, offsuit dans inférieur
            suited_flag = i < j
            forward_state = build_state_via_env(env, (r1, r2), suited_flag, order="ab")
            reverse_state = build_state_via_env(env, (r1, r2), suited_flag, order="ba")

            v_ab = critic_value(agent, forward_state)
            v_ba = critic_value(agent, reverse_state)

            forward_matrix[i, j] = v_ab
            reverse_matrix[i, j] = v_ba
            diff_matrix[i, j] = abs(v_ab - v_ba)

            # Debug print once if verbose requested
            if i == 0 and j == 1:
                debug_full_state(forward_state)

    vmin = min(forward_matrix.min(), reverse_matrix.min())
    vmax = max(forward_matrix.max(), reverse_matrix.max())

    labels = [rank_label(r) for r in ranks]

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    sns.heatmap(forward_matrix, ax=axes[0], xticklabels=labels, yticklabels=labels,
                cmap="RdYlGn", center=(vmin+vmax)/2, vmin=vmin, vmax=vmax)
    axes[0].set_title("V(a,b) – upper: suited, lower: offsuit")

    sns.heatmap(reverse_matrix, ax=axes[1], xticklabels=labels, yticklabels=labels,
                cmap="RdYlGn", center=(vmin+vmax)/2, vmin=vmin, vmax=vmax)
    axes[1].set_title("V(b,a) – upper: suited, lower: offsuit")

    sns.heatmap(diff_matrix, ax=axes[2], xticklabels=labels, yticklabels=labels,
                cmap="magma", vmin=0, vmax=diff_matrix.max())
    axes[2].set_title("|V(a,b) - V(b,a)| asymmetry")

    plt.tight_layout()

    os.makedirs("visualizations_behavior", exist_ok=True)
    plt.savefig("visualizations_behavior/Preflop_V_tri_heatmaps.png")

if __name__ == "__main__":
    main(model_path="saved_models/poker_agent_Player_0.pth") 