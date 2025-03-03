# config.py
import random as rd
import numpy as np
import torch

EPISODES = 2_000
GAMMA = 0.9985
ALPHA = 0.001
EPS_DECAY = 0.9992
START_EPS = 0.9
STATE_SIZE = 127

# Paramètres de visualisation
DEBUG = False
RENDERING = False
FPS = 3

# Sauvegarde
SAVE_INTERVAL = 250
PLOT_INTERVAL = 500

# Nombre de Simulations
MC_SIMULATIONS = 100

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
