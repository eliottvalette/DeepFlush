# poker_small_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import random as rd

class PokerSmallModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        
        # Couches communes pour traiter l'état
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Tête pour les probabilités d'action (politique)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        
        # Tête pour la valeur d'état
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Paramètres:
        x: Le dernier état de la séquence d'états [batch_size, input_dim]
        
        Retourne:
        action_probs: Probabilités d'action [batch_size, output_dim]
        state_value: Valeur d'état [batch_size, 1]
        """
        # take the last state of the sequence
        if len(x.shape) == 3:  # Si la forme est [batch_size, seq_len, input_dim]
            x = x[:, -1, :]  # Prendre le dernier état de la séquence pour chaque élément du batch
        
        # Traitement de l'état à travers les couches communes
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.15)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.15)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.15)
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        
        # Calcul des probabilités d'action (utilise softmax pour normaliser)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        # Calcul de la valeur d'état
        state_value = self.value_head(x)
        
        return action_probs, state_value
        