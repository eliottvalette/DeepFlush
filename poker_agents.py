import torch
import torch.nn as nn
import torch.optim as optim
from poker_model import PokerTransformerModel
from poker_game import PlayerAction
from collections import deque
import random
import numpy as np
import os
import torch.nn.functional as F

class PokerAgent:
    """Agent de poker utilisant un réseau de neurones Actor-Critic pour l'apprentissage par renforcement"""
    
    def __init__(self, device,state_size, action_size, gamma, learning_rate, entropy_coeff=0.01, value_loss_coeff=0.5, load_model=False, load_path=None, show_cards=False):
        """
        Initialisation de l'agent
        :param state_size: Taille du vecteur d'état
        :param action_size: Nombre d'actions possibles
        :param gamma: Facteur d'actualisation pour les récompenses futures
        :param learning_rate: Taux d'apprentissage
        :param entropy_coeff: Coefficient pour la régularisation par entropie
        :param value_loss_coeff: Coefficient pour la fonction de valeur
        :param load_model: Si True, charge un modèle existant
        :param load_path: Chemin vers le modèle à charger
        """
        # Vérification des paramètres
        if state_size <= 0 or action_size <= 0:
            raise ValueError(f"state_size et action_size doivent être positifs (reçu: {state_size}, {action_size})")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma doit être entre 0 et 1 (reçu: {gamma})")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate doit être positif (reçu: {learning_rate})")
        
        if load_model and (load_path is None or not isinstance(load_path, str)):
            raise ValueError("load_path doit être spécifié et être une chaîne de caractères quand load_model est True")

        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        # Utilisation du modèle Transformer qui attend une séquence d'inputs
        self.model = PokerTransformerModel(input_dim=state_size, output_dim=action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=100)  # Buffer de replay
        self.temp_memory = [] # Buffer temporaire pour les transitions de l'agent, avant update en backpropagation de la final reward

        if load_model:
            self.load(load_path)

        self.show_cards = show_cards
        self.name = 'unknown_agent'

    def load(self, load_path):
        """
        Charge un modèle sauvegardé
        """
        if not isinstance(load_path, str):
            raise TypeError(f"load_path doit être une chaîne de caractères (reçu: {type(load_path)})")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Le fichier {load_path} n'existe pas")
        
        try:
            self.model.load_state_dict(torch.load(load_path))
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")

    def get_action(self, state, valid_actions, epsilon=0.0):
        """
        Sélectionne une action selon la politique epsilon-greedy.
        Ici, 'state' est une séquence de vecteurs (shape: [n, 116]).

        Retourne l'action choisie et une éventuelle pénalité si l'action était invalide.
        """
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"state doit être une liste ou un numpy array (reçu: {type(state)})")
        if not valid_actions:
            raise ValueError("valid_actions ne peut pas être vide")
        
        # Vérifier que toutes les actions sont valides
        for action in valid_actions:
            if not isinstance(action, PlayerAction):
                raise TypeError(f"Toutes les actions doivent être de type PlayerAction (reçu: {type(action)})")

        # Mapping des actions
        action_map = {
            PlayerAction.FOLD: 0,
            PlayerAction.CHECK: 1,
            PlayerAction.CALL: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.RAISE_25_POT: 4,
            PlayerAction.RAISE_33_POT: 5,
            PlayerAction.RAISE_50_POT: 6,
            PlayerAction.RAISE_66_POT: 7,
            PlayerAction.RAISE_75_POT: 8,
            PlayerAction.RAISE_100_POT: 9,
            PlayerAction.RAISE_125_POT: 10,
            PlayerAction.RAISE_150_POT: 11,
            PlayerAction.RAISE_175_POT: 12,
            PlayerAction.RAISE_2X_POT: 13,
            PlayerAction.RAISE_3X_POT: 14,
            PlayerAction.ALL_IN: 15
        }
        valid_indices = [action_map[a] for a in valid_actions]

        # Création d'un masque pour ne considérer que les actions valides
        valid_action_mask = torch.zeros((1, self.action_size), device=self.device)
        for idx in valid_indices:
            valid_action_mask[0, idx] = 1

        # Implement epsilon-greedy
        if random.random() < epsilon:  # With probability epsilon, choose random action
            chosen_index = random.choice(valid_indices)
            # Create uniform distribution for reporting
            action_probs = torch.zeros((1, self.action_size), device=self.device)
            action_probs[0, valid_indices] = 1.0/len(valid_indices)
        else:
            # Use the model to choose action (existing code)
            state_tensor = torch.stack(state).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                action_probs, _ = self.model(state_tensor)
            masked_probs = action_probs * valid_action_mask
            
            if masked_probs.sum().item() == 0:
                chosen_index = random.choice(valid_indices)
            else:
                masked_probs = masked_probs / masked_probs.sum()
                chosen_index = torch.argmax(masked_probs).item()

        reverse_action_map = {v: k for k, v in action_map.items()}
        return reverse_action_map[chosen_index], valid_action_mask, action_probs

    def remember(self, state_seq, target_vector, valid_action_mask):
        """
        Stocke une transition dans la mémoire de replay, cette transition sera utilisée pour l'entrainement du model
        """
        self.memory.append((state_seq, target_vector, valid_action_mask))

    def train_model(self):
        """
        Entraîne le modèle sur un batch de transitions.
        Les états sont des séquences (shape: [n, 116]).
        """
        if len(self.memory) < 32:
            print('Not enough data to train :', len(self.memory))
            return {
                'policy_loss': None,
                'value_loss': None,
                'total_loss': None,
                'invalid_action_loss': None,
                'mean_predicted_value': None,
                'mean_target_value': None,
                'mean_action_prob': None
            }

        try:
            batch = random.sample(self.memory, 32)
            state_sequences, target_vectors, valid_action_masks = zip(*batch)

            # Conversion et uniformisation des valid_action_masks en tensors
            valid_action_masks_tensor = torch.stack([
                # S'assurer qu'il est de dimension 1
                torch.tensor(mask, device=self.device).squeeze()
                for mask in valid_action_masks
            ]).float()  # Shape: (batch_size, action_size)

            # Fixer la longueur de séquence à 10 (padding/tronquage)
            max_seq_len = 10

            # Création du tenseur pour les états
            padded_states = torch.zeros((len(state_sequences), max_seq_len, self.state_size), device=self.device)
            for i, state_sequence in enumerate(state_sequences):
                # Si la séquence est trop longue, on ne garde que les max_seq_len derniers états
                if len(state_sequence) > max_seq_len:
                    seq = state_sequence[-max_seq_len:]
                else:
                    seq = state_sequence
                
                # Convertir directement en tensor PyTorch
                seq_tensor = torch.stack(seq).to(self.device)
                
                # Vérifier la forme de la séquence
                if len(seq_tensor.shape) == 1:
                    # Si c'est un vecteur 1D, on le reshape pour avoir la bonne forme (cela peut arriver si l'agent a Fold Preflop la sequence est de longueur 1 et donc la state sequence devient automatiquement un vecteur 1D)
                    seq_tensor = seq_tensor.reshape(1, -1)
                
                # Remplir le tenseur padded_states avec la séquence
                padded_states[i, :len(seq_tensor)] = seq_tensor

            # Conversion des autres données en tensors PyTorch
            target_vector_tensor = torch.FloatTensor(target_vectors).to(self.device)

            # Calculer les probabilités d'action et les valeurs d'état
            action_probs, state_values = self.model(padded_states)
            state_values = state_values.squeeze(-1)

            # Calcul de la MSE entre les probabilités d'action et le vecteur cible
            policy_loss = F.mse_loss(action_probs, target_vector_tensor)
            
            # Calcul de la perte de la valeur (partie critique)
            # On utilise la valeur finale comme cible pour la valeur d'état
            value_loss = F.mse_loss(state_values, target_vector_tensor.max(dim=1)[0])

            # Calcul de la perte, Probs donnée aux actions invalides
            invalid_action_probs = action_probs * (1 - valid_action_masks_tensor)
            invalid_action_loss = F.mse_loss(invalid_action_probs, torch.zeros_like(invalid_action_probs))
            
            # Perte totale
            total_loss = policy_loss + self.value_loss_coeff * value_loss + invalid_action_loss

            # Optimisation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Métriques pour le suivi
            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item(),
                'invalid_action_loss': invalid_action_loss.item(),
                'mean_predicted_value': state_values.mean().item(),
                'mean_target_value': target_vector_tensor.max(dim=1)[0].mean().item(),
                'mean_action_prob': action_probs.mean().item()
            }

            return metrics

        except Exception as e:
            print(f"Erreur pendant l'entraînement: {str(e)}")
            raise e