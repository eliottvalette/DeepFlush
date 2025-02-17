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

# Hyperparamètres
EPISODES = 10_000
GAMMA = 0.9985
ALPHA = 0.001
EPS_DECAY = 0.99995
START_EPS = 0.5
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
         
    # Set MPS seed if available
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_episode(env: PokerGame, epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector) -> Tuple[List[float], List[dict]]:
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
    actions_taken = {player.name: [] for player in env.players}

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
    
    # Boucle principale du jeu
    while env.current_phase != GamePhase.SHOWDOWN:
        # Pour éviter les boucles infinies
        if len(actions_taken[env.players[env.current_player_seat].name]) > 20:
            raise Exception(f"Agent {env.current_player_seat + 1} a pris plus de 20 actions")
        
        current_player = next(p for p in env.players if p.seat_position == env.current_player_seat)
        
        env._update_button_states()
        valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
        if len(valid_actions) == 0:
            raise Exception(f"Agent {env.current_player_seat + 1} n'a plus d'actions valides et il lui a pourtant été demandé de jouer")

        # --- Utiliser la séquence d'états accumulée comme entrée ---
        player_state_seq = state_seq[current_player.name]
        
        # Récupérer l'action à partir du modèle en lui passant la sequence des états précédents
        action_chosen, action_mask = current_player.agent.get_action(player_state_seq, epsilon, valid_actions)
        
        # Exécuter l'action dans l'environnement
        next_state, reward = env.step(action_chosen)
        cumulative_rewards[current_player.name] += reward
        
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

            # Stocker l'expérience pour l'entrainement du modèle: on enregistre une copie de la séquence courante
            next_player_state_seq = state_seq[current_player.name].copy()
            actions_taken[env.players[env.current_player_seat].name].append(action_chosen)
            # Stocker l'expérience
            current_player.agent.remember(
                player_state_seq.copy(), 
                action_chosen, 
                reward, 
                next_player_state_seq,
                False,
                action_mask
            )
        
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

    # Calcul des récompenses finales en utilisant les stacks capturées pré-reset
    print("\n=== Résultats de l'épisode ===")
    # Attribution des récompenses finales
    for player in env.ordered_players:
        if env.net_stack_changes[player.name] > 0:
            stack_change_normalized = env.net_stack_changes[player.name] / env.starting_stack
            final_reward = (stack_change_normalized ** 0.5) * 5
        elif env.net_stack_changes[player.name] < 0:
            stack_change_normalized = env.net_stack_changes[player.name] / env.starting_stack
            final_reward = -(abs(stack_change_normalized) ** 0.5) * 5
            if env.final_stacks[player.name] <= 2:
                final_reward -= 2
        else:
            final_reward = 0

        # Créer et utiliser l'état final
        player_state_seq = state_seq[player.name]
        penultimate_state = player_state_seq[-1]
        final_state = env.get_final_state(penultimate_state, env.final_stacks)
        state_seq[player.name].append(final_state)
        full_player_state_seq = state_seq[player.name].copy()
        
        # Utiliser directement l'agent du joueur
        player.agent.remember(full_player_state_seq, None, final_reward, None, True, [1, 1, 1, 1, 1])
        cumulative_rewards[player.name] += final_reward

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
    env = PokerGame(agent_list)
    
    # Configuration du collecteur de données
    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY
    )
    
    try:
        for episode in range(episodes):
            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, 0.05, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            reward_dict, metrics_list = run_episode(
                env, epsilon, rendering, episode, render_every, data_collector
            )
            
            # Enregistrer les métriques pour cet épisode en associant chaque métrique à une clé "agent"
            metrics_history[str(episode)] = metrics_list
            
            # Afficher les informations de l'épisode
            print(f"\nEpisode [{episode + 1}/{episodes}]")
            print(f"Randomness: {epsilon*100:.3f}%")
            for player in env.players:
                print(f"Agent {player.name} reward: {reward_dict[player.name]:.2f}")

        # Créer le dossier saved_models s'il n'existe pas
        os.makedirs("saved_models", exist_ok=True)
        
        # Sauvegarder les modèles entraînés
        if episode == episodes - 1:
            print("\nSauvegarde des modèles...")
            if not os.path.exists("saved_models"):  # Création du dossier si nécessaire
                os.makedirs("saved_models")
            for player in env.players:
                # Ne sauvegarder que les agents qui ont un modèle (pas les bots)
                if hasattr(player.agent, 'model'):
                    torch.save(player.agent.model.state_dict(), 
                             f"saved_models/poker_agent_{player.name}_epoch_{episode+1}.pth")
            print("Modèles sauvegardés avec succès!")

            print("Génération de la vizualiation...")
            data_collector.force_visualization()

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        print("\nSauvegarde des modèles...")
        if not os.path.exists("saved_models"):  # Création du dossier si nécessaire
            os.makedirs("saved_models")
        for player in env.players:
            # Ne sauvegarder que les agents qui ont un modèle (pas les bots)
            if hasattr(player.agent, 'model'):
                torch.save(player.agent.model.state_dict(), 
                         f"saved_models/poker_agent_{player.name}_epoch_{episode+1}.pth")    
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
