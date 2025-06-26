# poker_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):  # max_len fixé à 10 (la longueur maximale de la séquence)
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

class PokerTransformerActorModel(nn.Module):
    """
    Calcule la politique π_θ(a | s).

    Le réseau construit un vecteur de caractéristiques partagé h = shared_layers(state),
    puis produit des logits pour toutes les combinaisons d'actions possibles.

    La sortie est une distribution catégorielle sur toutes les combinaisons d'actions possibles.
    """
    def __init__(self, input_dim=142, output_dim=5, nhead=4, num_layers=4, dim_feedforward=512):
        super().__init__()
        
        # Choix d'un espace latent (d_model) de 64 dimensions.
        # Chaque vecteur d'état de dimension 142 sera projeté en un vecteur de dimension 64.
        d_model = 64
        
        # Couche linéaire pour projeter chaque vecteur d'état (de dimension 142) en un vecteur de dimension 64.
        # L'entrée est de forme (batch_size, seq_len, 142) et la sortie (batch_size, seq_len, 64).
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Ajout d'un encodage positionnel pour injecter une notion d'ordre dans la séquence. Aucune modification de la dimension.
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Création d'une couche d'encodeur Transformer.
        # On utilise batch_first=True pour que l'entrée soit de forme (batch_size, seq_len, d_model).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        # L'encodeur Transformer se compose de plusieurs couches identiques.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Tête de sortie pour prédire les probabilités d'action.
        # Elle transforme la représentation finale (64 dimensions) en un vecteur de probabilité de taille output_dim (ici 5).
        self.action_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Créer un masque de padding basé sur les valeurs nulles
        # On considère qu'une séquence est paddée si tous les éléments d'un vecteur d'état sont 0
        padding_mask = (x.sum(dim=-1) == 0)  # Shape: (batch_size, seq_len)
        
        # 1. Projection linéaire de chaque vecteur d'état de 142 à 64 dimensions.
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

        return action_probs
    
class PokerTransformerCriticModel(nn.Module):
    """
    Réseau Q duel pour les actions composites :
        • branche partagée  → h
        • tête V(s)         → (batch,1)
        • tête A(s,a)       → (batch, num_actions) - une pour chaque action
        • Q(s,a)=V+A-mean(A)
    """
    def __init__(self, input_dim=142, output_dim=5, nhead=4, num_layers=4, dim_feedforward=512):
        super().__init__()
        # Choix d'un espace latent (d_model) de 64 dimensions.
        # Chaque vecteur d'état de dimension 142 sera projeté en un vecteur de dimension 64.
        d_model = 64
        
        # Couche linéaire pour projeter chaque vecteur d'état (de dimension 142) en un vecteur de dimension 64.
        # L'entrée est de forme (batch_size, seq_len, 142) et la sortie (batch_size, seq_len, 64).
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Ajout d'un encodage positionnel pour injecter une notion d'ordre dans la séquence. Aucune modification de la dimension.
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Création d'une couche d'encodeur Transformer.
        # On utilise batch_first=True pour que l'entrée soit de forme (batch_size, seq_len, d_model).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        # L'encodeur Transformer se compose de plusieurs couches identiques.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Tête de sortie pour prédire les probabilités d'action.
        # Elle transforme la représentation finale (64 dimensions) en un vecteur de probabilité de taille output_dim (ici 5).
        self.V_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, 1)
        )

        self.A_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, x):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        A(s,a) estimates the advantage of each action combination, it's an estimation of how much each action combination is favorable compared to the other combinations.

        So Q(s,a) is a function that estimates the future expected rewards of action combination a in state s.
        To do so, it takes that current value of the state V(s), add the advantage of the action combination A(s,a) to it and substract the mean of the advantages to normalize it. 
        """
        # Créer un masque de padding basé sur les valeurs nulles
        # On considère qu'une séquence est paddée si tous les éléments d'un vecteur d'état sont 0
        padding_mask = (x.sum(dim=-1) == 0)  # Shape: (batch_size, seq_len)
        
        # 1. Projection linéaire de chaque vecteur d'état de 142 à 64 dimensions.
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
        
        # 5. Prédiction de la valeur d'état :
        #    last_hidden est passé à travers un réseau feed-forward pour produire un scalaire.
        V = self.V_head(last_hidden)

        # 6. Prédiction des Q-values :
        #    last_hidden est passé à travers un réseau feed-forward pour produire un vecteur de dimension output_dim (ici 5).
        A = self.A_head(last_hidden)

        # 7. Calcul de la Q-value :
        #    Q(s,a)=V+A-mean(A)
        Q = V + A - A.mean(dim=-1, keepdim=True)

        return Q




