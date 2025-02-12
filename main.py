# main.py
from poker_train import main_training_loop, set_seed, EPISODES, GAMMA, ALPHA, STATE_SIZE, RENDERING
from poker_agents import PokerAgent
import torch

# Définir la graine pour la reproductibilité
set_seed(42)

# Définir le device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Forcer l'utilisation du CPU


# Liste des noms des joueurs
list_names = ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5", "Player_6"]

agent_list = []

# Créer les agents pour 6 joueurs
for i in range(6):
    agent = PokerAgent(
        state_size=STATE_SIZE,
        device=device,
        action_size=5,  # [fold, check, call, raise, all-in]
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=True,
        load_path=f"saved_models/poker_agent_{list_names[i]}.pth"
    )
    agent.name = list_names[i]
    agent.is_human = True  # True pour voir leurs cartes
    agent_list.append(agent)

# Démarrer l'entraînement
main_training_loop(agent_list, episodes=EPISODES, rendering=RENDERING, render_every=1000)