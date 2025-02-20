# poker_train.py
import os
import gc
import numpy as np
import random as rd
import pygame
import torch
from visualization import DataCollector
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
from typing import List, Tuple
import json
from utils.config import EPISODES, EPS_DECAY, START_EPS, RENDERING, SAVE_INTERVAL, PLOT_INTERVAL
from utils.renderer import handle_final_rendering, handle_rendering
from utils.helping_funcs import save_models, save_metrics
from poker_MCCFR import MCCFRTrainer

# Compteurs
number_of_hand_per_game = 0

def run_episode(env: PokerGame, epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector, mccfr_trainer: MCCFRTrainer) -> Tuple[List[float], List[dict]]:
    """
    Exécute un épisode complet du jeu de poker.
    
    Args:
        env (PokerGame): L'environnement de jeu
        epsilon (float): Paramètre d'exploration
        rendering (bool): Active/désactive le rendu graphique
        episode (int): Numéro de l'épisode en cours
        render_every (int): Fréquence de mise à jour du rendu
        data_collector (DataCollector): Collecteur de données pour la visualisation

    Returns:
        Tuple[List[float], List[dict]]: Récompenses finales et métriques d'entraînement
    """

    # Nettoyage des variables non utilisées
    if episode % PLOT_INTERVAL + 1 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        torch.mps.empty_cache()

    global number_of_hand_per_game  # Ajout de cette ligne pour référencer et mettre à jour la variable globale

    # Vérification du nombre minimum de joueurs
    players_that_can_play = [p for p in env.players if p.stack > 0]
    if len(players_that_can_play) < 3 or number_of_hand_per_game > 100:  # Évite les heads-up
        env.reset()
        number_of_hand_per_game = 0
        players_that_can_play = [p for p in env.players if p.stack > 0]
    else:
        env.start_new_hand()
        number_of_hand_per_game += 1    

    # Stockage des actions et rewards par joueur
    cumulative_rewards = {player.name: 0 for player in env.players}

    # Initialiser un dictionnaire qui associe à chaque agent (par son nom) la séquence d'états
    state_seq = {player.name: [] for player in env.players}
    initial_state = env.get_state()  # état initial (vecteur de dimension 116)

    # Assurez-vous que chaque joueur a déjà son nom attribué dans l'environnement avant d'initialiser
    for player in env.players:
        state_seq[player.name].append(initial_state)

    # Stocker l'état initial pour la collecte des métriques
    state_info = {
        "player": env.players[env.current_player_seat].name,
        "phase": "pre-game",
        "action": None,
        "stack_changes": env.net_stack_changes,
        "final_stacks": env.final_stacks,
        "num_active_players": len(players_that_can_play),
        "state_vector": initial_state.tolist()
        }
    data_collector.add_state(state_info)
    
    #### Boucle principale du jeu ####
    while env.current_phase != GamePhase.SHOWDOWN:
        # Récupération du joueur actuel et mise à jour des boutons   
        current_player = next(p for p in env.players if p.seat_position == env.current_player_seat)
        env._update_button_states()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]

        # On récupere la sequence d'entat du joueur actuel
        player_state_seq = state_seq[current_player.name]
        
        # Génération du vecteur de probabilités cible avec MCCFR
        target_vector, payoffs = mccfr_trainer.compute_expected_payoffs(player_state_seq[-1], valid_actions)

        # Prédiction avec une inférence de classique du model
        action_chosen, action_mask, action_probs = current_player.agent.get_action(player_state_seq, valid_actions)

        
        # Exécuter l'action dans l'environnement
        next_state, _ = env.step(action_chosen)
        
        # Mise à jour de la séquence : on ajoute le nouvel état à la fin
        if env.current_phase != GamePhase.SHOWDOWN:
            state_seq[current_player.name].append(next_state)
            
            # Sauve l'exp dans un json. On ne stock pas le state durant le showdown car on le fait plus tard et cela créerait un double compte
            current_state = player_state_seq[-1].clone()
            state_info = {
                "player": current_player.name,
                "phase": env.current_phase.value,
                "action": action_chosen.value,
                "stack_changes": env.net_stack_changes,
                "final_stacks": env.final_stacks,
                "num_active_players": len(players_that_can_play),
                "state_vector": current_state.tolist()
                }
            data_collector.add_state(state_info)

            # Stocker l'expérience
            current_player.agent.temp_remember(
                state_seq = player_state_seq.copy(), 
                target_vector = target_vector, 
                valid_action_mask = action_mask
            )
        
        else : # Cas spécifique au joueur qui déclenche le showdown par son action
            # Stocker l'expérience pour l'entrainement du modèle: on enregistre une copie de la séquence courante
            previous_player_state_seq = state_seq[current_player.name].copy()
            penultimate_state = previous_player_state_seq[-1]
            final_state = env.get_final_state(penultimate_state, env.final_stacks)

            # On ajoute le nouvel état à la fin de la séquence (car dans ce cas, c'est un state issu d'une action)
            state_seq[current_player.name].append(next_state)

            # Stocker l'expérience
            current_player.agent.remember(
                state_seq = previous_player_state_seq.copy(), 
                target_vector = target_vector, 
                valid_action_mask = action_mask
            )
        
        # Rendu graphique si activé
        handle_rendering(env, rendering, episode, render_every)

    # Calcul des récompenses finales en utilisant les stacks capturées pré-reset
    print("\n=== Résultats de l'épisode ===")
    # Attribution des récompenses finales
    for player in env.players:
        # ---- Pour la collecte et l'affichage des métriques ----
        # Récupération de l'état final
        player_state_seq = state_seq[player.name]
        penultimate_state = player_state_seq[-1]
        final_state = env.get_final_state(penultimate_state, env.final_stacks)
        # Stocker l'expérience finale pour la collecte des métriques
        current_player_name = player.name
        state_info = {
            "player": current_player_name,
            "phase": GamePhase.SHOWDOWN.value,
            "action": None,
            "stack_changes": env.net_stack_changes,
            "final_stacks": env.final_stacks,
            "num_active_players": len(players_that_can_play),
            "state_vector": final_state.tolist()
        }
        data_collector.add_state(state_info)

    # Affichage des résultats
    print("\nRécompenses finales:")
    for player in env.players:
        print(f"  {player.name}: {cumulative_rewards[player.name]:.3f}")
    print(f"\nFin de l'épisode {episode}")
    print(f"Randomness: {epsilon*100:.3f}%")
    
    # Entraînement et collecte des métriques
    metrics_list = []
    for player in env.players:
        metrics = player.agent.train_model()
        metrics['reward'] = cumulative_rewards[player.name]
        metrics_list.append(metrics)

    # Sauvegarde des données
    data_collector.add_metrics(metrics_list)
    data_collector.save_episode(episode)

    # Gestion du rendu graphique final
    if rendering and (episode % render_every == 0):
        handle_final_rendering(env)

    return cumulative_rewards, metrics_list

def main_training_loop(agent_list: List[PokerAgent], episodes: int = EPISODES, 
                      rendering: bool = RENDERING, render_every: int = 1000):
    """
    Boucle principale d'entraînement des agents.
    
    Args:
        agent_list (List[PokerAgent]): Liste des agents à entraîner
        episodes (int): Nombre total d'épisodes d'entraînement
        rendering (bool): Active/désactive le rendu graphique
        render_every (int): Fréquence de mise à jour du rendu graphique
    """
    # Initialisation des historiques et de l'environnement
    metrics_history = {}
    env = PokerGame(agent_list)
    
    # Configuration du collecteur de données
    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY
    )

    # Initialisation du MCCFRTrainer
    mccfr_trainer = MCCFRTrainer(env.players[0])
    try:
        for episode in range(episodes):
            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, 0.05, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            reward_dict, metrics_list = run_episode(
                env, epsilon, rendering, episode, render_every, data_collector, mccfr_trainer
            )
            
            # Enregistrer les métriques pour cet épisode en associant chaque métrique à une clé "agent"
            metrics_history[str(episode)] = metrics_list
            
            # Afficher les informations de l'épisode
            print(f"\nEpisode [{episode + 1}/{episodes}]")
            print(f"Randomness: {epsilon*100:.3f}%")
            for player in env.players:
                print(f"Agent {player.name} reward: {reward_dict[player.name]:.2f}")

        # Save models at end of training
        if episode == episodes - 1:
            save_models(env.players, episode)
            print("Generating visualization...")
            data_collector.force_visualization()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_models(env.players, episode)
        print("Generating visualization...")
        data_collector.force_visualization()
        
    finally:
        if rendering:
            pygame.quit()
        save_metrics(metrics_history, data_collector.output_dir)
