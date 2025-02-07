# poker_train.py
import os
import logging
import numpy as np
import random as rd
import pygame
import torch
import time
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg qui ne nécessite pas de GUI
from visualization import TrainingVisualizer, plot_winning_stats
from typing import List, Tuple

# Hyperparamètres
EPISODES = 4000
GAMMA = 0.9985
ALPHA = 0.001
EPS_DECAY = 0.9999
START_EPS = 1.0
STATE_SIZE = 201
RENDERING = False
FPS = 0.5

WINDOW_SIZE = 50  
PLOT_UPDATE_INTERVAL = 10  
SAVE_INTERVAL = 500 # Sauvegarder les graphiques tous les X épisodes

def set_seed(seed=42):
    """
    Définit les graines aléatoires pour garantir la reproductibilité des résultats.
    
    Args:
        seed (int): La graine à utiliser pour l'initialisation des générateurs aléatoires
    """
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_episode(env: PokerGame, agent_list: List[PokerAgent], epsilon: float, rendering: bool, episode: int, render_every: int):
    """
    Exécute un épisode complet du jeu de poker.
    
    Args:
        env (PokerGame): L'environnement du jeu de poker
        agent_list (List[PokerAgent]): Liste des agents participant à la partie
        epsilon (float): Paramètre d'exploration pour la politique epsilon-greedy
        rendering (bool): Active ou désactive le rendu graphique
        episode (int): Numéro de l'épisode en cours
        render_every (int): Fréquence de mise à jour du rendu graphique
    
    Returns:
        tuple: Contient (récompenses finales, liste des gagnants, actions prises,
               force des mains, métriques d'entraînement)
    """

    active_players = [p for p in env.players if p.stack > 0]
    if len(active_players) < 2:
        env.reset()
    else:
        env.start_new_hand()
    
    # Synchroniser les noms des joueurs et leur statut humain avec les agents
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human
    
    # Initialiser les récompenses cumulatives pour chaque joueur
    cumulative_rewards = [0] * len(agent_list)
    initial_stacks = [player.stack for player in env.players]

    # Modifier les actions_taken pour suivre par agent
    actions_taken = {f"Agent {i+1}": [] for i in range(len(agent_list))}
    hand_strengths = [0] * len(agent_list)  # Initialize with zeros for all players

    # Boucle principale du jeu
    while not env.current_phase == GamePhase.SHOWDOWN:
        
        # Ajouter une limite de sécurité pour éviter les boucles infinies
        if len(actions_taken[f"Agent {env.current_player_seat + 1}"]) > 100:
            raise Exception(f"Agent {env.current_player_seat + 1} a pris plus de 100 actions")

        # Récupérer le joueur actuel et l'agent correspondant
        current_player = env.players[env.current_player_seat]
        current_agent = agent_list[env.current_player_seat]
        
        # Obtenir l'état actuel du jeu et les actions valides
        state = env.get_state()
        env._update_button_states()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        
        if len(valid_actions) == 0:
            print(f"\nBug pour {current_player.name} ===")
            print(f"Joueur actif : {current_player.is_active}")
            print(f"Phase actuelle : {env.current_phase}")
            print(f"Actions valides : {valid_actions}")
            print(f"A folded : {current_player.has_folded}")
            print(f"A all-in : {current_player.is_all_in}")
            print(f"Pot actuel : {env.phase_pot}BB")
            print(f"Mise maximale actuelle : {env.current_maximum_bet}BB")
            print(f"Stack du joueur avant action : {current_player.stack}BB")
            print(f"Mise actuelle du joueur : {current_player.current_player_bet}BB")
            raise Exception(f"Agent {env.current_player_seat + 1} n'a plus d'actions valides et il lui a pourtant été demandé de jouer")
        
        # Calculate and store hand strength for current player
        strength = env._evaluate_hand_strength(current_player)
        hand_strengths[env.current_player_seat] = strength  # Store strength for current player
        
        # Obtenir l'action choisie par l'agent et la pénalité associée
        action_chosen = current_agent.get_action(state, epsilon, valid_actions)
        
        # Exécuter l'action et obtenir le nouvel état et la récompense
        next_state, reward = env.step(action_chosen)
        cumulative_rewards[env.current_player_seat] += reward
        
        # Stocker l'expérience dans la mémoire de l'agent
        current_agent.remember(state, action_chosen, reward, next_state, 
                             env.current_phase == GamePhase.SHOWDOWN)
        # Stocker l'action pour l'agent spécifique
        actions_taken[f"Agent {env.current_player_seat + 1}"].append(action_chosen)
            
        # Gérer l'affichage si le rendering est activé
        if rendering and (episode % render_every == 0):
            env._draw()
            pygame.display.flip()
            env.clock.tick(FPS)
            
            # Attendre que le temps d'affichage du gagnant soit écoulé
            if env.pygame_winner_info:
                current_time = pygame.time.get_ticks()
                while current_time - env.pygame_winner_display_start < env.pygame_winner_display_duration:
                    # Continuer à gérer les événements pygame pendant l'attente
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                    # Redessiner et mettre à jour l'affichage
                    env._draw()
                    pygame.display.flip()
                    env.clock.tick(FPS)
                    current_time = pygame.time.get_ticks()
    
    # Calculer les récompenses finales en fonction des changements de stack et du statut de victoire
    final_stacks = [player.stack for player in env.players]
    stack_changes = [np.clip((final - initial) / env.starting_stack, -1.0, 1.0) 
                    for final, initial in zip(final_stacks, initial_stacks)]
    
    # Déterminer les gagnants (joueurs avec le stack le plus élevé et qui n'ont pas fold)
    in_game_players = [p for p in env.players if p.is_active and not p.has_folded]
    if len(in_game_players) == 1:
        # Si un seul joueur actif, il est le gagnant
        winning_list = [1 if (p.is_active and not p.has_folded) else 0 for p in env.players]
    else:
        # Sinon, comparer les stacks des joueurs actifs
        max_stack = max(p.stack for p in in_game_players)
        winning_list = [1 if (p.is_active and not p.has_folded and p.stack == max_stack) else 0 for p in env.players]

    final_rewards = [r + s for r, s in zip(cumulative_rewards, stack_changes)]

    # Calculer la récompense finale en utilisant la force de la main
    for i, agent in enumerate(agent_list):
        env.current_player_seat = i
        terminal_state = env.get_state()
        is_winner = winning_list[i]
        
        # Calcul de la récompense finale :
        # - Pour un gagnant : récompense = (gain_relatif^0.5) * (1.1 - force_main) * 5
        #   → Récompense plus élevée pour gagner avec une main faible
        #   → Le facteur 1.1 assure une récompense positive même avec une main forte
        # - Pour un perdant : récompense = -(perte_relative^0.5) * force_main * 5
        #   → Pénalité plus forte pour perdre avec une main forte
        # La racine carrée atténue l'impact des grands gains/pertes
        if is_winner:
            final_reward = (stack_changes[i] ** 1/2) * (1.1 - hand_strengths[i]) * 5
        else:
            final_reward = -(abs(stack_changes[i]) ** 1/2) * hand_strengths[i] * 5

        agent.remember(terminal_state, None, final_reward, None, True)

    # Afficher les récompenses finales avec un format plus lisible
    print("Récompenses finales:")
    for i, reward in enumerate(final_rewards):
        print(f"  Joueur {i+1}: {reward:.3f}")
    
    # Entraîner les agents et collecter les métriques
    metrics_list = []
    for agent in agent_list:
        metrics = agent.train_model()
        metrics_list.append(metrics)

    # Afficher l'état final si le rendu est activé
    if rendering and (episode % render_every == 0):
        env._draw()
        pygame.display.flip()
        # Attendre que le temps d'affichage du gagnant soit écoulé
        if env.pygame_winner_info:
            current_time = pygame.time.get_ticks()
            while current_time - env.pygame_winner_display_start < env.pygame_winner_display_duration:
                # Continuer à gérer les événements pygame pendant l'attente
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                # Redessiner et mettre à jour l'affichage
                env._draw()
                pygame.display.flip()
                env.clock.tick(FPS)
                current_time = pygame.time.get_ticks()

    return final_rewards, winning_list, actions_taken, hand_strengths, metrics_list

def main_training_loop(agent_list, episodes=EPISODES, rendering=RENDERING, render_every=1000):
    """
    Boucle principale d'entraînement des agents.
    
    Args:
        agent_list (List[PokerAgent]): Liste des agents à entraîner
        episodes (int): Nombre total d'épisodes d'entraînement
        rendering (bool): Active ou désactive le rendu graphique
        render_every (int): Fréquence de mise à jour du rendu graphique
    """
    # Initialiser les historiques
    rewards_history = {}
    winning_history = {}
    
    # Initialiser l'environnement avec la liste des noms des joueurs
    list_names = [agent.name for agent in agent_list]
    env = PokerGame(list_names)
    
    # Initialiser le visualiseur
    visualizer = TrainingVisualizer(save_interval=SAVE_INTERVAL)

    # Créer l'environnement de jeu
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human
    
    try:
        for episode in range(episodes):
            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, 0.01, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            reward_list, winning_list, actions_taken, hand_strengths, metrics_list = run_episode(
                env, agent_list, epsilon, rendering, episode, render_every
            )
            
            # Mettre à jour les historiques
            rewards_history = update_rewards_history(rewards_history, reward_list, agent_list)
            winning_history = update_winning_history(winning_history, winning_list, agent_list)
            
            # Mettre à jour le visualiseur avec toutes les données incluant les métriques
            if episode % PLOT_UPDATE_INTERVAL == 0:
                visualizer.update_plots(episode, reward_list, winning_list, 
                                     actions_taken, hand_strengths, metrics_list)
            
            # Afficher les informations de l'épisode
            print(f"\nEpisode [{episode + 1}/{episodes}]")
            print(f"Randomness: {epsilon*100:.3f}%")
            for i, reward in enumerate(reward_list):
                print(f"Agent {i+1} reward: {reward:.2f}")

        # Sauvegarder les modèles entraînés et les graphiques finaux
        if episode == episodes - 1:
            print("\nSauvegarde des modèles...")
            for agent in agent_list:
                torch.save(agent.model.state_dict(), 
                         f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
            print("Modèles sauvegardés avec succès!")
            visualizer.save_counter = visualizer.save_interval  # Forcer une mise à jour
            visualizer.update_plots(episode, reward_list, winning_list, 
                                 actions_taken, hand_strengths, metrics_list)
            plot_winning_stats(winning_history, save_path="viz_pdf/poker_wins.jpg")
            print("Graphiques de visualisation sauvegardés avec succès!")

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        plot_winning_stats(winning_history, save_path="viz_pdf/poker_wins.jpg")
        print("Graphiques de visualisation sauvegardés avec succès!")

        print("\nSauvegarde des modèles...")
        for agent in agent_list:
            torch.save(agent.model.state_dict(), 
                     f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
    finally:
        if rendering:
            pygame.quit()
        print("\nEntraînement terminé")

def update_rewards_history(rewards_history: dict, reward_list: list, agent_list: list) -> dict:
    """
    Met à jour l'historique des récompenses pour chaque agent.
    
    Args:
        rewards_history (dict): Historique actuel des récompenses
        reward_list (list): Liste des nouvelles récompenses
        agent_list (list): Liste des agents
    
    Returns:
        dict: Historique des récompenses mis à jour
    """
    for i, agent in enumerate(agent_list):
        if agent.name not in rewards_history:
            rewards_history[agent.name] = []
        rewards_history[agent.name].append(reward_list[i])
    return rewards_history

def update_winning_history(winning_history: dict, winning_list: list, agent_list: list) -> dict:
    """
    Met à jour l'historique des victoires pour chaque agent.
    
    Args:
        winning_history (dict): Historique actuel des victoires
        winning_list (list): Liste des nouveaux résultats de victoire
        agent_list (list): Liste des agents
    
    Returns:
        dict: Historique des victoires mis à jour
    """
    for i, agent in enumerate(agent_list):
        if agent.name not in winning_history:
            winning_history[agent.name] = []
        winning_history[agent.name].append(winning_list[i])
    return winning_history
