# main.py
from poker_train_expresso import main_training_loop
from utils.config import set_seed, EPISODES, GAMMA, ALPHA, STATE_SIZE, RENDERING
from poker_agents import PokerAgent
from poker_bot import PokerBot
import torch
from collections import deque
import gc


# Définir la graine pour la reproductibilité
set_seed(43)

# Définir le device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Forcer l'utilisation du CPU

# Liste des noms des joueurs
list_names = ["Player_0", "Player_1", "Player_2"]

agent_list = []

# Créer 6 agents IA
for i in range(3):
    agent = PokerAgent(
        state_size=STATE_SIZE,
        device=device,
        action_size=12,
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=False, # if i >= 3 else True,
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
    shared_memory = agent_list[0].memory
    for agent in agent_list:  # Seulement pour les 4 premiers agents qui sont des IA
        agent.model = shared_model
        agent.memory = shared_memory

# Test avec un seul Joueur avec un modèle et tous les autres jouent au hasard (ils gardent leur mémoire vide et donc n'apprennent jamais)
elif False:
    for agent in agent_list[3:]:
        agent.memory = deque(maxlen=0)

# Démarrer l'entraînement
main_training_loop(agent_list, episodes=EPISODES, rendering=RENDERING, render_every=1)