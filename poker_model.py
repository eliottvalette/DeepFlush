# poker_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.config import (
    MODEL_INPUT_DIM, 
    MODEL_OUTPUT_DIM, 
    MODEL_NHEAD, 
    MODEL_NUM_LAYERS, 
    MODEL_DIM_FEEDFORWARD,
    MODEL_D_MODEL
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):  # max_len fixé à 100 (la longueur maximale de la séquence)
        super().__init__()
        
        # Crée un vecteur de positions [0, 1, ..., max_len-1] de forme (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        # Calcule un facteur d'échelle pour les sinusoïdes (selon la formule originale)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Initialise la matrice d'encodage positionnel de taille (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Applique sin() aux dimensions paires
        pe[:, 0::2] = torch.sin(position * div_term)
        # Applique cos() aux dimensions impaires
        pe[:, 1::2] = torch.cos(position * div_term)
        # Enregistre pe dans le module pour qu'il ne soit pas considéré comme un paramètre à apprendre
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x a la forme (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Ajoute l'encodage positionnel aux vecteurs d'entrée (les seq_len premières lignes de pe)
        return x + self.pe[:seq_len]

class PokerTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Choix d'un espace latent (d_model) de 64 dimensions.
        # Chaque vecteur d'état de dimension 116 sera projeté en un vecteur de dimension 64.
        d_model = 64
        
        # Couche linéaire pour projeter chaque vecteur d'état (de dimension 180) en un vecteur de dimension 64.
        self.input_projection = nn.Linear(MODEL_INPUT_DIM, MODEL_D_MODEL)
        
        self.pos_encoder = PositionalEncoding(MODEL_D_MODEL)
        
        # Création d'une couche d'encodeur Transformer. Gobelin code moi un sourire. 
        # On utilise batch_first=True pour que l'entrée soit de forme (batch_size, seq_len, d_model).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_D_MODEL,
            nhead=MODEL_NHEAD,
            dim_feedforward=MODEL_DIM_FEEDFORWARD,
            batch_first=True
        )
        
        # L'encodeur Transformer se compose de plusieurs couches identiques.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=MODEL_NUM_LAYERS
        )
        
        # Tête de sortie pour prédire les probabilités d'action.
        # Elle transforme la représentation finale (64 dimensions) en un vecteur de probabilité de taille output_dim (ici 5).
        self.action_head = nn.Sequential(
            nn.Linear(MODEL_D_MODEL, MODEL_DIM_FEEDFORWARD),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM_FEEDFORWARD),
            nn.Linear(MODEL_DIM_FEEDFORWARD, MODEL_OUTPUT_DIM),
            nn.Softmax(dim=-1)
        )
        
        # Tête de sortie pour estimer la valeur de l'état.
        # Elle transforme la représentation finale (64 dimensions) en un scalaire.
        self.value_head = nn.Sequential(
            nn.Linear(MODEL_D_MODEL, MODEL_DIM_FEEDFORWARD),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM_FEEDFORWARD),
            nn.Linear(MODEL_DIM_FEEDFORWARD, 1)
        )

    def forward(self, x):
        # Créer un masque de padding basé sur les valeurs nulles
        # On considère qu'une séquence est paddée si tous les éléments d'un vecteur d'état sont 0
        padding_mask = (x.sum(dim=-1) == 0)  # Shape: (batch_size, seq_len)
        
        # 1. Projection linéaire de chaque vecteur d'état de 116 à 64 dimensions.
        #    Après cette étape, x a la forme (batch_size, 10, 64).
        x = self.input_projection(x)
        
        # 2. Ajout de l'encodage positionnel :
        #    Chaque vecteur de la séquence reçoit une composante qui dépend de sa position dans la séquence.
        #    La forme reste (batch_size, 10, 64).
        x = self.pos_encoder(x)
        
        # 3. Passage dans l'encodeur Transformer :
        #    Le Transformer applique plusieurs mécanismes d'attention pour contextualiser chaque vecteur 
        #    en fonction des autres éléments de la séquence. La forme de x reste (batch_size, 10, 64).
        transformer_out = self.transformer(x, mask=None, src_key_padding_mask=padding_mask)
        
        # 4. Agrégation de la séquence :
        #    On récupère le dernier vecteur de la séquence, qui est censé contenir une représentation 
        #    globale de l'information de toute la séquence.
        #    last_hidden a la forme (batch_size, 64).
        last_hidden = transformer_out[:, -1]
        
        # 5. Prédiction des probabilités d'action :
        #    last_hidden est passé à travers un réseau feed-forward pour produire un vecteur de dimension 5,
        #    puis softmax est appliqué pour obtenir une distribution de probabilités.
        action_probs = self.action_head(last_hidden)
        
        # 6. Estimation de la valeur d'état :
        #    last_hidden est également passé à travers un autre réseau feed-forward pour produire un scalaire.
        state_value = self.value_head(last_hidden)
        
        return action_probs, state_value