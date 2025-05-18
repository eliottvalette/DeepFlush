# poker_agents.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from poker_model import PokerTransformerModel
from poker_small_model import PokerSmallModel
from poker_game_expresso import PlayerAction
from utils.config import DEBUG
import torch.nn.functional as F
from collections import deque
import random
import time
import os

class PokerAgent:
    """Agent de poker utilisant un réseau de neurones Actor-Critic pour l'apprentissage par renforcement"""
    
    def __init__(self, device,state_size, action_size, gamma, learning_rate,
                 entropy_coeff=0.1,
                 value_loss_coeff=0.01,
                 invalid_action_loss_coeff=2,
                 policy_loss_coeff=0.4,
                 reward_norm_coeff=0.5,
                 load_model=False, load_path=None, show_cards=False):
        """
        Initialisation de l'agent
        :param state_size: Taille du vecteur d'état
        :param action_size: Nombre d'actions possibles
        :param gamma: Facteur d'actualisation pour les récompenses futures
        :param learning_rate: Taux d'apprentissage
        :param entropy_coeff: Coefficient pour la régularisation par entropie
        :param value_loss_coeff: Coefficient pour la fonction de valeur
        :param invalid_action_loss_coeff: Coefficient pour la perte des actions invalides
        :param policy_loss_coeff: Coefficient pour la perte de la politique
        :param reward_norm_coeff: Coefficient pour la normalisation des récompenses
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
        self.invalid_action_loss_coeff = invalid_action_loss_coeff
        self.policy_loss_coeff = policy_loss_coeff
        self.reward_norm_coeff = reward_norm_coeff

        # Utilisation du modèle Transformer qui attend une séquence d'inputs
        self.model = PokerTransformerModel(input_dim=state_size, output_dim=action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10_000)  # Buffer de replay

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

    def get_action(self, state, valid_actions, target_vector, epsilon=0.0):
        """
        Sélectionne une action selon la politique epsilon-greedy.
        Ici, 'state' est une séquence de vecteurs (shape: [n, 106]).

        Retourne l'action choisie et une éventuelle pénalité si l'action était invalide.
        """
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"state doit être une liste ou un numpy array (reçu: {type(state)})")
        if not valid_actions:
            raise ValueError("valid_actions ne peut pas être vide")
        
        # Mapping des actions
        action_map = {
            PlayerAction.FOLD: 0,
            PlayerAction.CHECK: 1,
            PlayerAction.CALL: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.RAISE_25_POT: 4,
            PlayerAction.RAISE_50_POT: 5,
            PlayerAction.RAISE_75_POT: 6,
            PlayerAction.RAISE_100_POT: 7,
            PlayerAction.RAISE_150_POT: 8,
            PlayerAction.RAISE_2X_POT: 9,
            PlayerAction.RAISE_3X_POT: 10,
            PlayerAction.ALL_IN: 11
        }

        action_map_str = {
            PlayerAction.FOLD.value: 0,
            PlayerAction.CHECK.value: 1,
            PlayerAction.CALL.value: 2,
            PlayerAction.RAISE.value: 3,
            PlayerAction.RAISE_25_POT.value: 4,
            PlayerAction.RAISE_50_POT.value: 5,
            PlayerAction.RAISE_75_POT.value: 6,
            PlayerAction.RAISE_100_POT.value: 7,
            PlayerAction.RAISE_150_POT.value: 8,
            PlayerAction.RAISE_2X_POT.value: 9,
            PlayerAction.RAISE_3X_POT.value: 10,
            PlayerAction.ALL_IN.value: 11
        }

        valid_indices = [action_map_str[a.value] for a in valid_actions]
        if len(valid_indices) == 0:
            raise ValueError(f"No valid actions found for state: {state}")

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
        elif len(self.memory) == 0: # Chose the highest probability action from the target vector
            chosen_index = np.argmax(target_vector)
            action_probs = torch.zeros((1, self.action_size), device=self.device)
            action_probs[0, chosen_index] = 1.0
        else:
            # Convert numpy arrays to PyTorch tensors
            state_tensors = [torch.from_numpy(s).float() for s in state]
            state_tensor = torch.stack(state_tensors).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                action_probs, state_value = self.model(state_tensor)  
                if round(action_probs.sum().item(), 2) != 1:
                    raise ValueError(f"The model predicted a probability of {round(action_probs.sum().item(), 2)} for all valid actions, state: {state}")
            masked_probs = action_probs * valid_action_mask
            
            if masked_probs.sum().item() == 0:
                raise ValueError(f"The model predicted a probability of 0 for all valid actions, state: {state}")
            else:
                masked_probs = masked_probs / masked_probs.sum()
                chosen_index = torch.argmax(masked_probs).item()

        reverse_action_map = {v: k for k, v in action_map.items()}

        return reverse_action_map[chosen_index], valid_action_mask, action_probs

    def remember(self, state_seq, action_index, valid_action_mask, reward, target_vector):
        """
        Stocke une transition dans la mémoire de replay, cette transition sera utilisée pour l'entrainement du model
        """
        # Store sequence of states, chosen action index, valid action mask, and final reward
        self.memory.append((state_seq, action_index, valid_action_mask, reward, target_vector))

    def train_model(self):
        """
        Entraîne le modèle sur un batch de transitions en combinant reward
        et indicateur d'action optimale issu de MCCFR.
        """
        if len(self.memory) < 32:
            if DEBUG:
                print('Not enough data to train :', len(self.memory))
            return {
                'reward_norm_mean': None,
                'invalid_action_loss': None,
                'value_loss': None,
                'policy_loss': None,
                'entropy': None,
                'total_loss': None
            }

        # Sample transitions and unpack including target_vectors
        batch = random.sample(self.memory, 32)
        state_sequences, actions, valid_action_masks, rewards, target_vectors = zip(*batch)

        # Convert valid action masks to tensor
        valid_action_masks_tensor = torch.stack([mask for mask in valid_action_masks]).float().to(self.device)

        # Pad sequences to max length 10
        max_seq_len = 10
        padded_states = torch.zeros((len(state_sequences), max_seq_len, self.state_size), device=self.device)
        for i, state_sequence in enumerate(state_sequences):
            seq = state_sequence[-max_seq_len:] if len(state_sequence) > max_seq_len else state_sequence
            seq_tensors = [torch.from_numpy(s).float() for s in seq]
            seq_tensor = torch.stack(seq_tensors)
            if seq_tensor.dim() == 1:
                seq_tensor = seq_tensor.unsqueeze(0)
            padded_states[i, :seq_tensor.size(0)] = seq_tensor

        # Convert actions and rewards to tensors
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)

        # Stack target_vectors into tensor
        target_tensors = torch.stack([
            torch.from_numpy(tv).float() if isinstance(tv, np.ndarray) else tv
            for tv in target_vectors
        ]).to(self.device)

        # Forward pass through network
        action_probs, state_values = self.model(padded_states) # action_probs shape: (batch_size, 12), state_values shape: (batch_size, 1)

        # Implement the loss function
        # Normalize rewards for stable training
        reward_norm = rewards_tensor * self.reward_norm_coeff
        # Increase penalty when payoff is -100 (lost all money)
        full_loss_multiplier = 2.0
        full_loss_mask = (rewards_tensor <= -90.0)
        reward_norm[full_loss_mask] *= full_loss_multiplier

        # Value loss: fit state values to normalized rewards
        state_values = state_values.squeeze()
        value_loss = F.mse_loss(state_values, reward_norm)

        # Policy loss: encourage match to MCCFR target distribution
        clamped_probs = action_probs.clamp(min=1e-8)
        policy_loss = - (target_tensors * torch.log(clamped_probs)).sum(dim=1).mean()

        # Entropy bonus: encourage exploration
        entropy = -(action_probs * torch.log(clamped_probs)).sum(dim=1).mean()

        # Invalid action loss: penalize probability mass on invalid actions
        invalid_probs = action_probs * (1 - valid_action_masks_tensor)
        invalid_action_loss = invalid_probs.sum(dim=1).mean()

        # Total loss: weighted sum of components + entropy
        total_loss = (
            - reward_norm.mean()
          + self.policy_loss_coeff   * policy_loss
          + self.value_loss_coeff    * value_loss
          + self.invalid_action_loss_coeff * invalid_action_loss
          - self.entropy_coeff       * entropy
        )

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Metrics for monitoring
        metrics = {
            'reward_norm_mean': reward_norm.mean().item(),
            'invalid_action_loss': invalid_action_loss.item() * self.invalid_action_loss_coeff,
            'value_loss': value_loss.item() * self.value_loss_coeff,
            'policy_loss': policy_loss.item() * self.policy_loss_coeff,
            'entropy': entropy.item() * self.entropy_coeff,
            'total_loss': total_loss.item()
        }

        return metrics

    def cleanup(self):
        """
        Clean up resources and memory
        """
        # Clear memory
        self.memory.clear()
        
        # Cleanup optimizer
        if hasattr(self, 'optimizer'):
            self.optimizer.zero_grad(set_to_none=True)
        
        # Cleanup device cache
        try:
            torch.cuda.empty_cache() 
        except:
            pass
        try:
            torch.mps.empty_cache()
        except:
            pass