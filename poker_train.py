# poker_train.py
import os
import logging
import numpy as np
import random as rd
import pygame
import torch
import time
from visualization import DataCollector
from poker_agents import PokerAgent
from poker_game import PokerGame, GamePhase, PlayerAction, HandRank
from typing import List, Tuple
import json
import glob

# Hyperparamètres
EPISODES = 10_000
GAMMA = 0.9985
ALPHA = 0.001
EPS_DECAY = 0.9994
START_EPS = 0.999
STATE_SIZE = 116

# Paramètres de visualisation
RENDERING = False      # Active/désactive l'affichage graphique
FPS = 3                # Images par seconde pour le rendu

# Intervalles de sauvegarde
SAVE_INTERVAL = 250    # Fréquence de sauvegarde des modèles
PLOT_INTERVAL = 500    # Fréquence de mise à jour des graphiques

# Compteurs
number_of_hand_per_game = 0

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

def run_episode(env: PokerGame, agent_list: List[PokerAgent], epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector) -> Tuple[List[float], List[dict]]:
    """
    Exécute un épisode complet du jeu de poker.
    
    Args:
        env (PokerGame): L'environnement de jeu
        agent_list (List[PokerAgent]): Liste des agents participants
        epsilon (float): Paramètre d'exploration
        rendering (bool): Active/désactive le rendu graphique
        episode (int): Numéro de l'épisode en cours
        render_every (int): Fréquence de mise à jour du rendu
        data_collector (DataCollector): Collecteur de données pour la visualisation

    Returns:
        Tuple[List[float], List[dict]]: Récompenses finales et métriques d'entraînement
    """
    global number_of_hand_per_game  # Added this to reference and update the global variable

    # Vérification du nombre minimum de joueurs
    players_that_can_play = [p for p in env.players if p.stack > 0]
    if len(players_that_can_play) < 3 or number_of_hand_per_game > 100:  # Évite les heads-up
        env.reset()
        number_of_hand_per_game = 0
    else:
        env.start_new_hand()
        number_of_hand_per_game += 1
    
    # Synchroniser les noms et statuts humains entre l'environnement et les agents
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human

    cumulative_rewards = [0] * len(agent_list)
    initial_stacks = [player.stack for player in env.players]

    # Stockage des actions par agent
    actions_taken = {f"Agent {i+1}": [] for i in range(len(agent_list))}

    # --- IMPORTANT : INITIALISER LA SÉQUENCE D'ÉTATS ---
    # Au lieu d'un simple vecteur, on construit ici une séquence d'états, chaque séquence est spécifique à un agent
    state_seq = [[] for _ in range(len(agent_list))]
    initial_state = env.get_state()  # état initial (vecteur de dimension 116)
    for i, agent in enumerate(agent_list):
        state_seq[i].append(initial_state)
    
    # Boucle principale du jeu
    while env.current_phase != GamePhase.SHOWDOWN:
        # Pour éviter les boucles infinies
        if len(actions_taken[f"Agent {env.current_player_seat + 1}"]) > 25:
            raise Exception(f"Agent {env.current_player_seat + 1} a pris plus de 25 actions")
        
        current_player = env.players[env.current_player_seat]
        current_agent = agent_list[env.current_player_seat]
        
        env._update_button_states()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        if len(valid_actions) == 0:
            raise Exception(f"Agent {env.current_player_seat + 1} n'a plus d'actions valides et il lui a pourtant été demandé de jouer")

        # --- Utiliser la séquence d'états accumulée comme entrée ---
        player_state_seq = state_seq[env.current_player_seat]
        
        # Récupérer l'action à partir du modèle en lui passant la sequence des états précédents
        print('Number of states in sequence:', len(player_state_seq))
        print('Shape of each state:', player_state_seq[0].shape)
        
        # Récupérer l'action à partir du modèle en lui passant la sequence des états précédents
        action_chosen, action_mask = current_agent.get_action(player_state_seq, epsilon, valid_actions)
        
        # Exécuter l'action dans l'environnement
        next_state, reward = env.step(action_chosen)
        cumulative_rewards[env.current_player_seat] += reward
        
        # Mise à jour de la séquence : on ajoute le nouvel état à la fin
        state_seq[env.current_player_seat].append(next_state)
        next_player_state_seq = state_seq[env.current_player_seat].copy()
        
        # Stocker l'expérience : on enregistre une copie de la séquence courante
        current_agent.remember(
            player_state_seq.copy(), 
            action_chosen, 
            reward, 
            next_player_state_seq,
            env.current_phase == GamePhase.SHOWDOWN,
            action_mask
        )
        actions_taken[f"Agent {env.current_player_seat + 1}"].append(action_chosen)
        
        # Stocker l'état actuel pour la collecte des métriques
        current_state = player_state_seq[-1].clone()
        state_info = {
            "player": current_player.name,
            "phase": env.current_phase.value,
            "action": action_chosen.value if action_chosen else None,
            "state_vector": current_state.tolist()  # Utilisation de l'état pré-action
        }
        data_collector.add_state(state_info)
        
        # Rendu graphique si activé
        if rendering and (episode % render_every == 0):
            env._draw()
            pygame.display.flip()
            env.clock.tick(FPS)
            if env.pygame_winner_info:
                current_time = pygame.time.get_ticks()
                while current_time - env.pygame_winner_display_start < env.pygame_winner_display_duration:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                    env._draw()
                    pygame.display.flip()
                    env.clock.tick(FPS)
                    current_time = pygame.time.get_ticks()

    # Calcul des récompenses finales
    print("\n=== Résultats de l'épisode ===")
    
    # Calculer les changements de stack pour chaque joueur
    current_in_game_players_mask = [p.is_active for p in env.players] 
    final_stacks = [player.stack for player in env.players]
    stack_changes = [
        np.clip((final - initial) / env.starting_stack, -1.0, 1.0)
        for final, initial in zip(final_stacks, initial_stacks)
    ]
    
    # Déterminer les gagnants
    remaining_players = [p for p in env.players if p.is_active and not p.has_folded]
    
    if len(remaining_players) == 1:
        # Cas où un seul joueur reste (les autres ont fold)
        winning_list = [1 if (p.is_active and not p.has_folded) else 0 for p in env.players]
    else:
        # Cas normal - déterminer les gagnants par les changements de stack
        winning_list = [1 if change > 0 else 0 for change in stack_changes] # TODO: Arréter de dire que les inactifs (dès le debut de la main) sont des perdants (affecte uniquement les plots et non l'entrainement des modèles)
        
        # Vérification de cohérence - au moins un gagnant doit exister
        if sum(winning_list) == 0 and remaining_players:
            print("Warning: Aucun gagnant détecté malgré des joueurs actifs")
            max_stack = max(p.stack for p in remaining_players)
            winning_list = [
                1 if (p.is_active and not p.has_folded and p.stack == max_stack) else 0 
                for p in env.players
            ]

    # Attribution des récompenses finales
    for i, agent in enumerate(agent_list):
        if not current_in_game_players_mask[i]:
            continue  # Ignorer les joueurs inactifs
            
        env.current_player_seat = i
        player_state_seq = state_seq[i]
        terminal_state = player_state_seq.copy()
        is_winner = winning_list[i]

        # Calcul de la récompense finale
        if is_winner:
            if stack_changes[i] < 0:
                raise Exception(f"Agent {i+1} a gagné avec un stack change négatif: {stack_changes[i]}")
            final_reward = (stack_changes[i] ** 0.5) * 5
        else:
            if stack_changes[i] > 0:
                raise Exception(f"Agent {i+1} a perdu avec un stack change positif: {stack_changes[i]}")
            final_reward = -(abs(stack_changes[i]) ** 0.5) * 5
            
            # Pénalité supplémentaire si le joueur est presque ruiné
            if env.players[i].stack <= 10:
                final_reward -= 2
                
        # Enregistrer l'expérience finale
        agent.remember(terminal_state, None, final_reward, None, True, [1, 1, 1, 1, 1])
        cumulative_rewards[i] += final_reward

    # Affichage des résultats
    print("\nRécompenses finales:")
    for i, reward in enumerate(cumulative_rewards):
        print(f"  Joueur {i+1}: {reward:.3f}")
    print(f"\nFin de l'épisode {episode}")
    print(f"Randomness: {epsilon*100:.3f}%")
    
    # Entraînement et collecte des métriques
    metrics_list = []
    for agent in agent_list:
        metrics = agent.train_model()
        metrics['reward'] = cumulative_rewards[agent_list.index(agent)]
        metrics_list.append(metrics)

    # Sauvegarde des données
    data_collector.add_metrics(metrics_list)
    data_collector.save_episode(episode)

    # Gestion du rendu graphique final
    if rendering and (episode % render_every == 0):
        _handle_final_rendering(env)

    return cumulative_rewards, metrics_list

def _handle_final_rendering(env):
    """
    Gère l'affichage final de l'épisode, incluant l'information du gagnant.
    
    Args:
        env (PokerGame): L'environnement de jeu
    """
    env._draw()
    pygame.display.flip()
    
    if env.pygame_winner_info:
        current_time = pygame.time.get_ticks()
        while current_time - env.pygame_winner_display_start < env.pygame_winner_display_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            env._draw()
            pygame.display.flip()
            env.clock.tick(FPS)
            current_time = pygame.time.get_ticks()

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
    list_names = [agent.name for agent in agent_list]
    env = PokerGame(list_names)
    
    # Configuration du collecteur de données
    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY
    )

    # Créer l'environnement de jeu
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human
    
    try:
        for episode in range(episodes):
            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, 0.05, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            reward_list, metrics_list = run_episode(
                env, agent_list, epsilon, rendering, episode, render_every, data_collector
            )
            
            # Enregistrer les métriques pour cet épisode en associant chaque métrique à une clé "agent"
            metrics_with_agent = []
            for i, metric in enumerate(metrics_list):
                metrics_with_agent.append({"agent": agent_list[i].name, **metric})
            metrics_history[str(episode)] = metrics_with_agent
            
            # Afficher les informations de l'épisode
            print(f"\nEpisode [{episode + 1}/{episodes}]")
            print(f"Randomness: {epsilon*100:.3f}%")
            for i, reward in enumerate(reward_list):
                print(f"Agent {i+1} reward: {reward:.2f}")
            
            # Afficher les métriques de chaque agent
            print("Métriques:")
            for i, metrics in enumerate(metrics_with_agent):
                metric_str = f"  Agent {i+1}:"
                # Print all available metrics
                for key, value in metrics.items():
                    if key != 'agent':  # Skip the agent name
                        try:
                            metric_str += f" {key} = {float(value):.6f},"
                        except (ValueError, TypeError):
                            metric_str += f" {key} = {value},"
                print(metric_str.rstrip(','))  # Remove trailing comma

        # Sauvegarder les modèles entraînés
        if episode == episodes - 1:
            print("\nSauvegarde des modèles...")
            for agent in agent_list:
                torch.save(agent.model.state_dict(), 
                         f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")
            print("Modèles sauvegardés avec succès!")

            print("Génération de la vizualiation...")
            data_collector.force_visualization()

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        print("\nSauvegarde des modèles...")
        for agent in agent_list:
            torch.save(agent.model.state_dict(), 
                     f"saved_models/poker_agent_{agent.name}_epoch_{episode+1}.pth")    
        print("Génération de la vizualiation...")
        data_collector.force_visualization()
        
    finally:
        if rendering:
            pygame.quit()
        # Sauvegarder l'historique des métriques dans le dossier viz_json
        metrics_path = os.path.join(data_collector.output_dir, "metrics_history.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        print(f"\nEntraînement terminé et métriques sauvegardées dans '{metrics_path}'")
