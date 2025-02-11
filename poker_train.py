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
START_EPS = 0.5
STATE_SIZE = 201
RENDERING = False
FPS = 1

SAVE_INTERVAL = 250
PLOT_INTERVAL = 500

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

def run_episode(env: PokerGame, agent_list: List[PokerAgent], epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector):
    """
    Exécute un épisode complet du jeu de poker en utilisant une séquence d'états.
    Chaque agent reçoit en entrée la séquence complète des états depuis le début de l'épisode.
    
    Returns:
        tuple: (récompenses finales, liste des gagnants, actions prises,
                force des mains, métriques d'entraînement)
    """

    active_players = [p for p in env.players if p.stack > 0]
    if len(active_players) < 2:
        env.reset()
    else:
        env.start_new_hand()
    
    # Synchroniser les noms et statuts humains entre l'environnement et les agents
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human

    cumulative_rewards = [0] * len(agent_list)
    initial_stacks = [player.stack for player in env.players]

    # Stockage des actions par agent
    actions_taken = {f"Agent {i+1}": [] for i in range(len(agent_list))}
    hand_strengths = [0] * len(agent_list)

    # --- IMPORTANT : INITIALISER LA SÉQUENCE D'ÉTATS ---
    # Au lieu d'un simple vecteur, on construit ici une séquence d'états, chaque séquence est spécifique à un agent
    state_seq = [[] for _ in range(len(agent_list))]
    initial_state = env.get_state()  # état initial (vecteur de dimension 201)
    for i, agent in enumerate(agent_list):
        state_seq[i].append(initial_state)
    
    # Boucle principale du jeu
    while env.current_phase != GamePhase.SHOWDOWN:
        # Pour éviter les boucles infinies
        if len(actions_taken[f"Agent {env.current_player_seat + 1}"]) > 100:
            raise Exception(f"Agent {env.current_player_seat + 1} a pris plus de 100 actions")
        
        current_player = env.players[env.current_player_seat]
        current_agent = agent_list[env.current_player_seat]
        
        env._update_button_states()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        if len(valid_actions) == 0:
            raise Exception(f"Agent {env.current_player_seat + 1} n'a plus d'actions valides et il lui a pourtant été demandé de jouer")
        
        # Calculer et stocker la force de la main pour le joueur courant
        strength = env._evaluate_hand_strength(current_player)
        hand_strengths[env.current_player_seat] = strength

        # --- Utiliser la séquence d'états accumulée comme entrée ---
        player_state_seq = state_seq[env.current_player_seat]
        # Capture de l'état avant l'action, c'est celui que le modèle reçoit
        state_for_decision = player_state_seq[-1].copy()
        action_chosen = current_agent.get_action(player_state_seq, epsilon, valid_actions)
        
        # Exécuter l'action dans l'environnement
        next_state, reward = env.step(action_chosen)
        cumulative_rewards[env.current_player_seat] += reward
        
        # Mise à jour de la séquence : on ajoute le nouvel état à la fin
        state_seq[env.current_player_seat].append(next_state)
        
        # Stocker l'expérience : on enregistre une copie de la séquence courante
        current_agent.remember(player_state_seq.copy(), action_chosen, reward, next_state, env.current_phase == GamePhase.SHOWDOWN)
        actions_taken[f"Agent {env.current_player_seat + 1}"].append(action_chosen)
        
        # Stocker l'état utilisé pour prendre l'action avec des informations supplémentaires
        state_info = {
            "player": current_player.name,
            "phase": env.current_phase.value,
            "action": action_chosen.value if action_chosen else None,
            "state_vector": state_for_decision.tolist()  # Utilisation de l'état pré-action
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

    # Calcul des récompenses finales basées sur les changements de stack
    final_stacks = [player.stack for player in env.players]
    stack_changes = [np.clip((final - initial) / env.starting_stack, -1.0, 1.0)
                     for final, initial in zip(final_stacks, initial_stacks)]
    
    in_game_players = [p for p in env.players if p.is_active and not p.has_folded]
    if len(in_game_players) == 1:
        winning_list = [1 if (p.is_active and not p.has_folded) else 0 for p in env.players]
    else:
        # Determine winners based on who gained chips this hand
        winning_list = [1 if change > 0 else 0 for change in stack_changes]
        
        # Sanity check - at least one winner should exist if there are active players
        if sum(winning_list) == 0 and len(in_game_players) > 0:
            print("Warning: No winners detected despite active players")
            # Fall back to original logic
            max_stack = max(p.stack for p in in_game_players)
            winning_list = [1 if (p.is_active and not p.has_folded and p.stack == max_stack) else 0 for p in env.players]

    final_rewards = [r + s for r, s in zip(cumulative_rewards, stack_changes)]

    # Calcul de la récompense finale en fonction de la force de la main pour chaque agent
    for i, agent in enumerate(agent_list):
        env.current_player_seat = i
        # Utiliser la séquence finale comme terminal_state
        player_state_seq = state_seq[i]
        terminal_state = player_state_seq.copy()
        is_winner = winning_list[i]
        if is_winner:
            if stack_changes[i] < 0:
                raise Exception(f"Agent {i+1} a gagné avec un stack change négatif, ce stack change est de {stack_changes[i]}")
            final_reward = (stack_changes[i] ** 0.5) * (1.1 - hand_strengths[i]) * 5 + 0.5
        else:
            final_reward = -(abs(stack_changes[i]) ** 0.5) * hand_strengths[i] * 5 - 0.5
        agent.remember(terminal_state, None, final_reward, None, True)

    print("Récompenses finales:")
    for i, reward in enumerate(final_rewards):
        print(f"  Joueur {i+1}: {reward:.3f}")
    print(f"\nFin de l'épisode {episode}")
    print(f"Randomness: {epsilon*100:.3f}% ")
    
    # Entraîner les agents et récupérer les métriques
    metrics_list = []
    for agent in agent_list:
        metrics = agent.train_model()
        metrics['reward'] = final_rewards[agent_list.index(agent)]
        metrics_list.append(metrics)

    # Sauvegarder les données de l'épisode et les métriques
    data_collector.add_metrics(metrics_list)
    data_collector.save_episode(episode)

    if rendering and (episode % render_every == 0):
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

    # Collecter les hand ranks finaux et les résultats
    final_hand_ranks = []
    for i, player in enumerate(env.players):
        if not player.cards:  # Si le joueur n'a pas de cartes
            final_hand_ranks.append((HandRank.HIGH_CARD, False))
        elif player.has_folded:
            hand_rank, _ = env.evaluate_current_hand(player)
            final_hand_ranks.append((hand_rank, False))
        else:
            hand_rank, _ = env.evaluate_final_hand(player)
            final_hand_ranks.append((hand_rank, winning_list[i] == 1))

    return final_rewards, metrics_list


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
    metrics_history = {}  # Historique pour les métriques d'entraînement
    
    # Initialiser l'environnement avec la liste des noms des joueurs
    list_names = [agent.name for agent in agent_list]
    env = PokerGame(list_names)
    
    # Initialiser le collecteur de données et supprimer les JSON existants dans viz_json
    data_collector = DataCollector(save_interval=SAVE_INTERVAL, plot_interval=PLOT_INTERVAL, start_epsilon=START_EPS, epsilon_decay=EPS_DECAY)

    # Créer l'environnement de jeu
    for i, agent in enumerate(agent_list):
        env.players[i].name = agent.name
        env.players[i].is_human = agent.is_human
    
    try:
        for episode in range(episodes):
            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, 0.01, START_EPS)
            
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
