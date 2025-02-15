# main.py
from poker_train import main_training_loop, set_seed, EPISODES, GAMMA, ALPHA, STATE_SIZE, RENDERING
from poker_agents import PokerAgent
from bot import PokerBot
import torch

# Définir la graine pour la reproductibilité
set_seed(43)

# Définir le device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Forcer l'utilisation du CPU

# Liste des noms des joueurs
list_names = ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5", "Player_6"]

agent_list = []

# Créer 6 agents IA
for i in range(6):
    agent = PokerAgent(
        state_size=STATE_SIZE,
        device=device,
        action_size=5,
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=False,
        load_path=f"saved_models/poker_agent_{list_names[i]}.pth",
        show_cards=True
    )
    agent.name = list_names[i]
    agent_list.append(agent)

for agent in agent_list:
    print('agent :', agent.name)

# Faire en sorte que les agents IA partagent le même modèle (Changer entre True et False pour activer ou désactiver le partage de modèle)
if False:
    shared_model = agent_list[0].model
    for agent in agent_list[:4]:  # Seulement pour les 4 premiers agents qui sont des IA
        agent.model = shared_model

# Démarrer l'entraînement
main_training_loop(agent_list, episodes=EPISODES, rendering=RENDERING, render_every=1)