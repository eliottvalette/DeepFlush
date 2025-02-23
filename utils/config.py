# config.py
import random as rd
import numpy as np
import torch

EPISODES = 10_000
GAMMA = 0.9985
ALPHA = 0.001
EPS_DECAY = 0.9996
START_EPS = 0.8
STATE_SIZE = 190

# Hyperparamètres du modèle Transformer
MODEL_INPUT_DIM = 190           # Dimension d'entrée pour le modèle
MODEL_OUTPUT_DIM = 16           # Dimension de sortie pour le modèle (nombre d'actions)
MODEL_NHEAD = 4                 # Nombre de têtes d'attention
MODEL_NUM_LAYERS = 4            # Nombre de couches dans l'encodeur Transformer
MODEL_DIM_FEEDFORWARD = 512     # Dimension de la couche feedforward
MODEL_D_MODEL = 64              # Dimension latente (d_model) du Transformer

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
