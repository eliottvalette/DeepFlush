# poker_train.py
import numpy as np 
import random as rd
import pygame
import time
from poker_game import PokerGame, GamePhase, PlayerAction
from bot import hardcoded_poker_bot
import matplotlib
matplotlib.use('Agg')

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)

def create_bot_players():
    """
    Crée une liste de 6 joueurs dont 5 bots et 1 humain.
    
    Returns:
        list: Liste des joueurs avec leurs paramètres
    """
    player_list = []
    
    # Créer le joueur humain
    player_list.append({
        'name': 'Human',
        'is_human': True,
        'bot_function': None
    })
    
    # Créer 5 bots
    for i in range(5):
        player_list.append({
            'name': f'Bot_{i+1}',
            'is_human': False,
            'bot_function': hardcoded_poker_bot
        })
    
    return player_list

def run_test_games(player_list, env):
    """
    Exécute des parties de test avec les bots.
    
    Args:
        player_list (list): Liste des joueurs (humain + bots)
        env (PokerGame): Environnement du jeu
    """
    running = True
    
    while running:
        # Gérer les événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    env.start_new_hand()
                if event.key == pygame.K_r:
                    env.reset()
            
            # Gérer les entrées du joueur humain 
            current_player = env.players[env.current_player_seat]
            current_player_info = player_list[env.current_player_seat]
            
            if current_player.is_human:
                env.handle_input(event)
            else:
                print(f"Current player: {current_player.name}")
                state = env.get_state()
                env._update_button_states()
                valid_actions = [a for a in PlayerAction if env.action_buttons[a].enabled]
                print('valid_actions', valid_actions)
                
                print(f"Bot {current_player.name} is playing")
                # Utiliser le bot hardcodé pour choisir l'action parmi les actions autorisées
                action_vector = current_player_info['bot_function'](state, valid_actions)
                # Si aucune action n'est retournée, on passe au tour suivant
                if action_vector is None:
                    continue
                action_index = np.argmax(action_vector)
                action_chosen = list(PlayerAction)[action_index]
                
                # Ajouter un délai pour voir l'action du bot
                time.sleep(1)
                
                # Exécuter l'action choisie
                env.process_action(current_player, action_chosen)
        
        # Mettre à jour l'affichage
        env._draw()
        pygame.display.flip()
        env.clock.tick(30)  # 30 FPS
    
    pygame.quit()

if __name__ == "__main__":
    # Créer les joueurs (1 humain + 5 bots)
    player_list = create_bot_players()
    
    # Extraire les noms des joueurs pour initialiser l'environnement
    list_names = [player['name'] for player in player_list]
    
    # Initialiser l'environnement avec les noms
    env = PokerGame(list_names)
    
    # Synchroniser les types des joueurs
    for i, player_info in enumerate(player_list):
        env.players[i].is_human = player_info['is_human']
    
    # Lancer les parties de test
    run_test_games(player_list, env)
