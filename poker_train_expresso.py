# poker_train_expresso.py
import os
import gc
import numpy as np
import random as rd
import pygame
import torch
import time
import copy
from visualization import DataCollector
from poker_agents import PokerAgent
from poker_game_expresso import PokerGame, GamePhase, PlayerAction
from typing import List, Tuple
import json
from utils.config import EPISODES, EPS_DECAY, START_EPS, DEBUG, SAVE_INTERVAL, PLOT_INTERVAL, MC_SIMULATIONS
from utils.renderer import handle_final_rendering, handle_rendering
from utils.helping_funcs import save_models
from poker_MCCFR_expresso import MCCFRTrainer

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

    global number_of_hand_per_game  # Ajout de cette ligne pour référencer et mettre à jour la variable globale

     # Nettoyage des caches
    if episode % 100 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        torch.mps.empty_cache()

    # Vérification du nombre minimum de joueurs
    players_that_can_play = [p for p in env.players if p.stack > 0]
    if len(players_that_can_play) < 2 or number_of_hand_per_game > 1:  # Évite les heads-up
        env.reset()
        number_of_hand_per_game = 0
        players_that_can_play = [p for p in env.players if p.stack > 0]
    else:
        env.start_new_hand()
        number_of_hand_per_game += 1    

    # Initialiser un dictionnaire qui associe à chaque agent son stack initial
    initial_stacks = {player.name: player.stack for player in env.players}

    # Initialiser un dictionnaire qui associe à chaque agent (par son nom) la séquence d'états
    state_seq = {player.name: [] for player in env.players}
    initial_state = env.get_state(seat_position = env.current_player_seat)  # état initial (vecteur de dimension 116)

    # Assurez-vous que chaque joueur a déjà son nom attribué dans l'environnement avant d'initialiser
    for player in env.players:
        state_seq[player.name].append(initial_state)

    # Stocker l'état initial pour la collecte des métriques
    state_info = {
        "player": env.players[env.current_player_seat].name,
        "phase": "pre-game",
        "action": None,
        "final_stacks": env.final_stacks,
        "num_active_players": len(players_that_can_play),
        "state_vector": initial_state.tolist()
        }
    data_collector.add_state(state_info)
    
    # Buffer to collect experiences until final reward is known
    experiences = []
    # Mapping from PlayerAction to action index
    action_to_idx = {
        PlayerAction.FOLD: 0,
        PlayerAction.CHECK: 1,
        PlayerAction.CALL: 2,
        PlayerAction.RAISE: 3,
        PlayerAction.RAISE_25_POT: 4,
        PlayerAction.RAISE_50_POT: 5,
        PlayerAction.RAISE_75_POT: 6,
        PlayerAction.RAISE_100_POT: 7,
        PlayerAction.RAISE_150_POT: 8,
        PlayerAction.RAISE_2X_POT: 9,
        PlayerAction.RAISE_3X_POT: 10,
        PlayerAction.ALL_IN: 11
    }

    #### Boucle principale du jeu ####
    while env.current_phase != GamePhase.SHOWDOWN:
        # Récupération du joueur actuel et mise à jour des boutons   
        current_player = next(p for p in env.players if p.seat_position == env.current_player_seat)
        env._update_button_states()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]

        # On récupère la sequence d'états du joueur actuel
        player_state_seq = state_seq[current_player.name]

        # Génération du vecteur de probabilités cible avec MCCFR à partir de l'état simplifié du jeu
        state = env.get_state(seat_position = current_player.seat_position)
        if DEBUG:
            print(f"state : {state}")
            print(f"\ncurrent_player.name : {current_player.name}, current_phase : {env.current_phase}, chosen_action : {chosen_action}\n")
            print(f"current_player_bet : {current_player.current_player_bet}, current_maximum_bet : {env.current_maximum_bet}, stack : {current_player.stack}")
        
        target_vector, payoffs = mccfr_trainer.compute_expected_payoffs_and_target_vector(
            valid_actions = valid_actions, 
            state = state, 
            hero_seat = current_player.seat_position,
            button_seat = env.button_seat_position,
            hero_cards = current_player.cards,
            visible_community_cards = env.community_cards,
            num_active_players = len(players_that_can_play),
            initial_stacks = initial_stacks.copy(),
            state_seq = copy.deepcopy(state_seq)
        )

        if DEBUG:
            print(f"hero_name : {current_player.name}\ntarget_vector : {target_vector}\n payoffs : {payoffs.values()}")

        # Prédiction avec une inférence classique du modèle
        chosen_action, action_mask, action_probs = current_player.agent.get_action(state = player_state_seq, valid_actions = valid_actions, target_vector = target_vector, epsilon = epsilon)

        review_state = env.get_state(seat_position = current_player.seat_position)
        if state.tolist() != review_state.tolist():
            raise ValueError(f"state != review_state => {state.tolist()} != {review_state.tolist()}")

        # Exécuter l'action dans l'environnement
        next_state = env.step(chosen_action)
        
        # Mise à jour de la séquence : on ajoute le nouvel état à la fin
        if env.current_phase != GamePhase.SHOWDOWN:
            state_seq[current_player.name].append(next_state)
            
            # Sauve l'exp dans un json. On ne stocke pas le state durant le showdown car on le fait plus tard et cela créerait un double compte
            current_state = player_state_seq[-1]
            state_info = {
                "player": current_player.name,
                "phase": env.current_phase.value,
                "action": chosen_action.value,
                "final_stacks": env.final_stacks,
                "num_active_players": len(players_that_can_play),
                "state_vector": current_state.tolist(),
                "target_vector": target_vector.tolist(),
                }
            data_collector.add_state(state_info)

            # Buffer experience (state_seq, action_index, mask) for later reward assignment
            experiences.append((current_player.agent, player_state_seq.copy(), action_to_idx[chosen_action], action_mask, target_vector, env.current_phase))
        
        else : # Cas spécifique au joueur qui déclenche le showdown par son action
            # Stocker l'expérience pour l'entrainement du modèle: on enregistre une copie de la séquence courante
            previous_player_state_seq = state_seq[current_player.name].copy()
            penultimate_state = previous_player_state_seq[-1]
            final_state = env.get_final_state(penultimate_state, env.final_stacks)

            # On ajoute le nouvel état à la fin de la séquence (car dans ce cas, c'est un state issu d'une action)
            state_seq[current_player.name].append(next_state)

            # Stocker l'expérience
            experiences.append((current_player.agent, previous_player_state_seq.copy(), action_to_idx[chosen_action], action_mask, target_vector, env.current_phase))
        
        # Rendu graphique si activé
        handle_rendering(env, rendering, episode, render_every)

    # Calcul des récompenses finales en utilisant les stacks capturées pré-reset
    print(f"\n=== Résultats de l'épisode [{episode + 1}/{EPISODES}] ===")
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
            "state_vector": final_state.tolist(),
            "target_vector": target_vector.tolist(),
        }
        data_collector.add_state(state_info)
    
    # Store buffered experiences with final rewards weighted by the phase
    phase_weights = {
        GamePhase.PREFLOP: 0.05,
        GamePhase.FLOP: 0.1,
        GamePhase.TURN: 0.3,
        GamePhase.RIVER: 0.5,
        GamePhase.SHOWDOWN: 1.0
    }
    for agent, state_sequence, action_idx, valid_mask, target_vector, phase in experiences:
        reward = env.net_stack_changes[agent.name] * phase_weights[phase]
        agent.remember(state_seq=state_sequence, action_index=action_idx, valid_action_mask=valid_mask, reward=reward, target_vector=target_vector)

    metrics_list = []
    for player in env.players:
        metrics = player.agent.train_model()
        metrics_list.append(metrics)

    # Sauvegarde des données
    data_collector.add_metrics(metrics_list)
    data_collector.save_episode(episode)

    # Gestion du rendu graphique final
    if rendering and (episode % render_every == 0):
        handle_final_rendering(env)

def main_training_loop(agent_list: List[PokerAgent], episodes: int, rendering: bool, render_every: int):
    """
    Boucle principale d'entraînement des agents.
    
    Args:
        agent_list (List[PokerAgent]): Liste des agents à entraîner
        episodes (int): Nombre total d'épisodes d'entraînement
        rendering (bool): Active/désactive le rendu graphique
        render_every (int): Fréquence de mise à jour du rendu graphique
    """
    # Initialisation des historiques et de l'environnement
    env = PokerGame(agents = agent_list, rendering=rendering)
    
    # Configuration du collecteur de données
    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY
    )

    # Initialisation du MCCFRTrainer
    mccfr_trainer = MCCFRTrainer(num_simulations = MC_SIMULATIONS, agent_list= agent_list)
    
    try:
        for episode in range(episodes):
            start_time = time.time()

            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, 0.05, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            run_episode(env, epsilon, rendering, episode, render_every, data_collector, mccfr_trainer)
            
            # Afficher les informations de l'épisode
            print(f"\nEpisode [{episode + 1}/{episodes}]")
            print(f"Randomness: {epsilon*100:.3f}%")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            
            # Periodic memory cleanup every 100 episodes
            if episode % 100 == 0:
                print("Performing periodic memory cleanup...")
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            
        # Save models at end of training
        if episode == episodes - 1:
            save_models(env.players, episode)
            print("Generating visualization...")
            data_collector.force_visualization()

    except Exception as e:
        print(f"An error occurred: {e}")
        save_models(env.players, episode)
        print("Generating visualization...")
        data_collector.force_visualization()
    finally:
        save_models(env.players, episode)
        print("Generating visualization...")
        data_collector.force_visualization()
        del env
        del mccfr_trainer
        gc.collect()
