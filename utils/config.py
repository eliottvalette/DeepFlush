# config.py
import random as rd
import numpy as np
import torch

EPISODES = 10_000
GAMMA = 0.99
ALPHA = 0.0003
EPS_DECAY = 0.9999
START_EPS = 0.9
STATE_SIZE = 190

# Hyperparamètres du modèle Transformer
MODEL_INPUT_DIM = 190           # Inchangé (dépend de la représentation d'état)
MODEL_OUTPUT_DIM = 16           # Inchangé (nombre d'actions possibles)
MODEL_NHEAD = 2                # Augmenté pour capturer plus de relations
MODEL_NUM_LAYERS = 2           # Augmenté pour une meilleure capacité de modélisation
MODEL_DIM_FEEDFORWARD = 256    # Augmenté pour plus de capacité
MODEL_D_MODEL = 32            # Augmenté pour une meilleure représentation

# Paramètres de visualisation
RENDERING = False
FPS = 3

# Sauvegarde
SAVE_INTERVAL = 250
PLOT_INTERVAL = 500

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
