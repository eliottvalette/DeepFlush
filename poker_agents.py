import torch
import torch.nn as nn
import torch.optim as optim
from poker_model import PokerTransformerModel
from poker_game import PlayerAction
from collections import deque
import random
import numpy as np
import os

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
        self.memory = deque(maxlen=10000)  # Buffer de replay

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

    def get_action(self, state, epsilon, valid_actions):
        """
        Sélectionne une action selon la politique epsilon-greedy.
        Ici, 'state' est une séquence de vecteurs (shape: [n, 116]).

        Retourne l'action choisie et une éventuelle pénalité si l'action était invalide.
        """
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"state doit être une liste ou un numpy array (reçu: {type(state)})")
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon doit être entre 0 et 1 (reçu: {epsilon})")
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
            PlayerAction.ALL_IN: 4
        }
        valid_indices = [action_map[a] for a in valid_actions]

        # Création d'un masque pour ne considérer que les actions valides
        valid_action_mask = torch.zeros((1, self.action_size), device=self.device)
        for idx in valid_indices:
            valid_action_mask[0, idx] = 1

        if np.random.random() < epsilon:
            chosen_index = np.random.choice(valid_indices)
            reverse_action_map = {v: k for k, v in action_map.items()}
            return reverse_action_map[chosen_index], valid_action_mask
        
        else:
            # Ajout d'une dimension batch : state est une liste de tensors, on utilise torch.stack pour obtenir la séquence sous forme d'un tensor 
            state_tensor = torch.stack(state).unsqueeze(0).to(self.device)
            self.model.eval()

            with torch.no_grad():
                # Le modèle retourne (action_probs, state_value)
                action_probs, _ = self.model(state_tensor)
            masked_probs = action_probs * valid_action_mask

            if masked_probs.sum().item() == 0:
                chosen_index = np.random.choice(valid_indices)

            else:
                masked_probs = masked_probs / masked_probs.sum()
                chosen_index = torch.argmax(masked_probs).item()

            if chosen_index not in valid_indices:
                # Forcer une action valide
                chosen_index = np.random.choice(valid_indices)

            reverse_action_map = {v: k for k, v in action_map.items()}
            return reverse_action_map[chosen_index], valid_action_mask

    def remember(self, state_seq, action, reward, next_state, done, valid_action_mask):
        """
        Stocke une transition dans la mémoire de replay
        :param state_seq: Séquence d'états
        :param action: Action effectuée
        :param reward: Récompense reçue
        :param next_state: État suivant
        :param done: True si l'épisode est terminé
        """
        action_map = {
            PlayerAction.FOLD: 0,
            PlayerAction.CHECK: 1,
            PlayerAction.CALL: 2,
            PlayerAction.RAISE: 3,
            PlayerAction.ALL_IN: 4,
            None: 0 #  L'action None est mappée au même indice que l'action fold (indice 0). L'action None est utilisée pour traiter le Showdown sans action, sans que cela n'ait d'impact négatif sur l'apprentissage global car elle n'est pas prise en compte dans le calcul des pertes.
        }
        numerical_action = action_map[action]
        self.memory.append((state_seq, numerical_action, reward, next_state, done, valid_action_mask))

    def train_model(self):
        """
        Entraîne le modèle sur un batch de transitions.
        Les états sont des séquences (shape: [n, 116]).
        """
        if len(self.memory) < 32:
            print('Not enough data to train :', len(self.memory))
            return {'loss': 0, 'entropy': 0, 'value_loss': 0, 'std': 0, 'learning_rate': self.learning_rate}

        try:
            batch = random.sample(self.memory, 32)
            state_sequences, actions, rewards, next_state_sequences, dones, valid_action_masks = zip(*batch)

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

            # Même chose pour les next_states
            padded_next_states = torch.zeros((len(next_state_sequences), max_seq_len, self.state_size), device=self.device)
            for i, next_state_sequence in enumerate(next_state_sequences):
                if next_state_sequence is not None:
                    if len(next_state_sequence) > max_seq_len:
                        ns_seq = next_state_sequence[-max_seq_len:]
                    else:
                        ns_seq = next_state_sequence
                    
                    # Convertir directement en tensor PyTorch
                    ns_seq_tensor = torch.stack(ns_seq).to(self.device)
                    
                    # Vérifier la forme de la séquence
                    if len(ns_seq_tensor.shape) == 1:
                        ns_seq_tensor = ns_seq_tensor.reshape(1, -1)
                    
                    # Remplir le tenseur padded_next_states
                    padded_next_states[i, :len(ns_seq_tensor)] = ns_seq_tensor

            # Conversion des autres données en tensors PyTorch
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            # Calculer les probabilités d'action et les valeurs d'état
            action_probs, state_values = self.model(padded_states)
            state_values = state_values.squeeze(-1)

            # Calcul des cibles TD et des avantages
            with torch.no_grad():
                _, next_state_values = self.model(padded_next_states)
                next_state_values = next_state_values.squeeze(-1)
            temporal_difference_targets = rewards_tensor + self.gamma * next_state_values * (1 - dones_tensor)
            advantages = temporal_difference_targets - state_values

            # Calcul des pertes
            selected_action_probs = action_probs[torch.arange(len(actions_tensor)), actions_tensor]
            policy_loss = -torch.mean(torch.log(selected_action_probs + 1e-10) * advantages.detach())
            value_loss = torch.mean((state_values - temporal_difference_targets.detach()) ** 2)
            entropy_loss = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))

            # Ajout d'un terme de régularisation pour les actions invalides utilisant le masque des actions valides 0 pour action invalide, 1 pour action valide
            invalid_action_penalty = torch.sum((1 - valid_action_masks_tensor) * action_probs) * 0.1

            total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss + invalid_action_penalty

            advantages_std = advantages.std().item()

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics = {
                'entropy_loss': entropy_loss.item(),
                'value_loss': value_loss.item(),
                'invalid_action_penalty': invalid_action_penalty.item(),
                'std': advantages_std,
                'learning_rate': self.learning_rate,
                'loss': total_loss.item()
            }
            return metrics

        except Exception as e:
            print(f"Erreur pendant l'entraînement: {str(e)}")
            raise e