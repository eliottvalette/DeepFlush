# poker_agents.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from poker_model import PokerTransformerActorModel, PokerTransformerCriticModel
from poker_small_model import PokerSmallModel
from poker_game_expresso import PlayerAction
from utils.config import DEBUG
import torch.nn.functional as F
from collections import deque
import random
import time
import os

class PokerAgent:
    """
    Wrapper de haut niveau qui couple un **Acteur** (politique π_θ) et un **Critique Duel**
    (Q_ϕ & V_ϕ).
    Caractéristiques principales
    ------------ 
      • *Boucle d'apprentissage*  
        1. L'acteur produit π_θ(a | s) et sélectionne les actions (ε-greedy).  
        2. Le critique produit Q(s,·) et V(s) → Cible TD  
           *td* = r + γ maxₐ′ Q(s′, a′).  
        3. Pertes  
           – **Acteur** : −log π_θ · Avantage  (A = Q−V)  − β H[π]  
           – **Critique**: MSE(Q(s,a), td)  
        4. Deux optimiseurs Adam indépendants mettent à jour θ et ϕ.
    """
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
        self.actor_model = PokerTransformerActorModel(input_dim=state_size, output_dim=action_size).to(device)
        self.critic_model = PokerTransformerCriticModel(input_dim=state_size, output_dim=action_size).to(device)
        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate * 0.1)
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
            checkpoint = torch.load(load_path)
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"Modèle chargé avec succès: {load_path}")
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
            raise ValueError(f"Aucune action valide trouvée pour l'état: {state}")

        # Création d'un masque pour ne considérer que les actions valides
        valid_action_mask = torch.zeros((1, self.action_size), device=self.device)
        for idx in valid_indices:
            valid_action_mask[0, idx] = 1

        # Implémentation epsilon-greedy
        if random.random() < epsilon:  # Avec une probabilité epsilon, choisir une action aléatoire
            chosen_index = random.choice(valid_indices)
            # Créer une distribution uniforme pour le rapport
            action_probs = torch.zeros((1, self.action_size), device=self.device)
            action_probs[0, valid_indices] = 1.0/len(valid_indices)
        elif len(self.memory) == 0:
            if random.random() < 0.25: # 25% de chance de choisir l'action avec la plus haute probabilité du vecteur cible
                chosen_index = np.argmax(target_vector)
                action_probs = torch.zeros((1, self.action_size), device=self.device)
                action_probs[0, chosen_index] = 1.0
                if DEBUG:
                    print(f"highest proba from target vector")
            elif random.random() < 0.5: # 25% de chance de choisir une action aléatoire parmi les actions valides
                chosen_index = random.choice(valid_indices)
                action_probs = torch.zeros((1, self.action_size), device=self.device)
                action_probs[0, chosen_index] = 1.0
                if DEBUG:
                    print(f"random choice from valid actions")
            else: # 50% de faire choisir le dernier agent saved (les agents avec des self.memory == 0 sont ceux pour lequels on a load_model = True mais qu'on a pas ré-entrainé par la suite)
                # Convertir les arrays numpy en tenseurs PyTorch
                state_tensors = [torch.from_numpy(s).float().to(self.device) for s in state]
                state_tensor = torch.stack(state_tensors).unsqueeze(0)
                
                self.actor_model.eval()
                with torch.no_grad():
                    action_probs = self.actor_model(state_tensor)
                    if round(action_probs.sum().item(), 2) != 1:
                        raise ValueError(f"Le modèle a prédit une probabilité de {round(action_probs.sum().item(), 2)} pour toutes les actions valides, état: {state}")
                masked_probs = action_probs * valid_action_mask
                
                if masked_probs.sum().item() == 0:
                    raise ValueError(f"Le modèle a prédit une probabilité de 0 pour toutes les actions valides, état: {state}")
                else:
                    masked_probs = masked_probs / masked_probs.sum()
                    chosen_index = torch.argmax(masked_probs).item()
                if DEBUG:
                    print(f"random choice from saved model")

        else:
            # Convertir les arrays numpy en tenseurs PyTorch
            state_tensors = [torch.from_numpy(s).float().to(self.device) for s in state]
            state_tensor = torch.stack(state_tensors).unsqueeze(0)
            
            self.actor_model.eval()
            with torch.no_grad():
                action_probs = self.actor_model(state_tensor)
                if round(action_probs.sum().item(), 2) != 1:
                    raise ValueError(f"Le modèle a prédit une probabilité de {round(action_probs.sum().item(), 2)} pour toutes les actions valides, état: {state}")
            masked_probs = action_probs * valid_action_mask
            
            if masked_probs.sum().item() == 0:
                raise ValueError(f"Le modèle a prédit une probabilité de 0 pour toutes les actions valides, état: {state}")
            else:
                masked_probs = masked_probs / masked_probs.sum()
                chosen_index = torch.argmax(masked_probs).item()

        reverse_action_map = {v: k for k, v in action_map.items()}

        return reverse_action_map[chosen_index], valid_action_mask, action_probs

    def remember(self, state_seq, action_index, valid_action_mask, reward, target_vector, done, next_state_seq):
        """
        Stocke une transition dans la mémoire de replay, cette transition sera utilisée pour l'entrainement du model
        """
        # Stocker la séquence d'états, l'indice d'action choisie, le masque d'action valide et la récompense finale
        self.memory.append((state_seq, action_index, valid_action_mask, reward, target_vector, done, next_state_seq))

    def train_model(self, batch_size=32):
        """
        Une étape d'optimisation sur un mini-batch.

        Workflow
        --------
            1.  Échantillonne `batch_size` transitions du buffer de replay choisi
                (court = on-policy, long = off-policy).  
            2.  Calcule
                    π_θ(a|s)                        # Réseau acteur
                    Q_ϕ(s, ·), V_ϕ(s)               # Réseau critique 
                    Q_target(s′, ·)                 # Réseau critique cible pour TD  
                    td_target = r + γ·maxₐ′ Q_target(s′, a′)  
                    advantage = Q(s,a) − V(s)  
            3.  Pertes  
                    critic_loss = Huber(Q(s,a), td_target)  
                    actor_loss  = −E[log π(a|s) · advantage] − β entropy  
            4.  Rétropropager et mettre à jour les deux optimiseurs.
            5.  Mettre à jour le réseau cible avec un lissage de Polyak.
        """
        if len(self.memory) < batch_size:
            if DEBUG:
                print('Pas assez de données pour entraîner:', len(self.memory))
            return {
                'reward_norm_mean': None,
                'invalid_action_loss': None,
                'value_loss': None,
                'policy_loss': None,
                'entropy': None,
                'total_loss': None
            }

        # Échantillonner les transitions et décompresser, y compris les target_vectors
        batch = random.sample(self.memory, batch_size)
        state_sequences, actions, valid_action_masks, rewards, target_vectors, dones, next_state_sequences = zip(*batch)

        # Convertir les masques d'action valides en tenseur
        valid_action_masks_tensor = torch.stack([mask for mask in valid_action_masks]).float().to(self.device)

        # Padder les séquences à une longueur maximale de 10
        max_seq_len = 10
        padded_states = torch.zeros((len(state_sequences), max_seq_len, self.state_size), device=self.device)
        for i, state_sequence in enumerate(state_sequences):
            seq = state_sequence[-max_seq_len:] if len(state_sequence) > max_seq_len else state_sequence
            seq_tensors = [torch.from_numpy(s).float() for s in seq]
            seq_tensor = torch.stack(seq_tensors)
            if seq_tensor.dim() == 1:
                seq_tensor = seq_tensor.unsqueeze(0)
            padded_states[i, :seq_tensor.size(0)] = seq_tensor
        
        padded_next_states = torch.zeros((len(next_state_sequences), max_seq_len, self.state_size), device=self.device)
        for i, next_state_sequence in enumerate(next_state_sequences):
            seq = next_state_sequence[-max_seq_len:] if len(next_state_sequence) > max_seq_len else next_state_sequence
            seq_tensors = [torch.from_numpy(s).float() for s in seq]
            seq_tensor = torch.stack(seq_tensors)
            if seq_tensor.dim() == 1:
                seq_tensor = seq_tensor.unsqueeze(0)
            padded_next_states[i, :seq_tensor.size(0)] = seq_tensor

        # Convertir les actions et récompenses en tenseurs
        actions_tensor = torch.tensor([a for a in actions], device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float, device=self.device)

        # Empiler les target_vectors en tenseur
        target_tensors = torch.stack([
            torch.from_numpy(tv).float() if isinstance(tv, np.ndarray) else tv
            for tv in target_vectors
        ]).to(self.device)

        # Passage en avant à travers le réseau
        action_probs = self.actor_model(padded_states)
        q_values, state_values = self.critic_model(padded_states)       # Q(s,*), V(s) => (batch_size, num_actions), (batch_size, 1)
        
        # Calcul des valeurs des états suivants en utilisant le réseau cible pour la stabilité
        with torch.no_grad():
            q_next, _ = self.critic_model(padded_next_states) # Q_target(s',*) => (batch_size, num_actions)
            next_state_values = q_next.max(dim=1).values      # max_a' Q_target(s',a') => (batch_size, 1)
        
        # Obtenir la Q-value pour l'action choisie
        chosen_action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) # Q(s,a) => (batch_size, 1)

        # Calculer la cible TD et l'avantage
        td_targets = rewards_tensor + self.gamma * next_state_values * (1 - dones_tensor)     # td_target = r + γ·maxₐ′ Q_target(s′, a′)
        advantages = chosen_action_q_values - state_values.squeeze(1).detach()                # A = Q - V Positif signifie que l'action est meilleure que prévu.

        if random.random() < 0.001:
            print(f'state_values, mean : {state_values.mean()}, max : {state_values.max()}, min : {state_values.min()}')
            print(f'rewards, mean : {rewards_tensor.mean()}, max : {rewards_tensor.max()}, min : {rewards_tensor.min()}')
            print(f'next_state_values, mean : {next_state_values.mean()}, max : {next_state_values.max()}, min : {next_state_values.min()}')
            print(f'advantages, mean : {advantages.mean()}, max : {advantages.max()}, min : {advantages.min()}')

        # Perte du critique: MSE entre les Q-values prédites et les cibles TD
        critic_loss = F.mse_loss(chosen_action_q_values, td_targets.detach())
        
        # Perte de l'état: MSE entre les valeurs d'état prédites et les récompenses normalisées
        state_value_loss = F.mse_loss(state_values.squeeze(), td_targets.detach())
        
        # Combinaison des pertes du critique
        total_critic_loss = critic_loss + self.value_loss_coeff * state_value_loss

        # Perte de l'acteur: utiliser l'avantage pour guider la politique
        # On veut maximiser log(π(a|s)) * avantage, donc minimiser le négatif
        log_probs = torch.log(action_probs.clamp(min=1e-8))
        # Récupérer les log-probs pour les actions choisies
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        # Politique guidée par l'avantage
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Perte de correspondance avec la politique cible MCCFR
        target_match_loss = -torch.sum(target_tensors * log_probs, dim=1).mean()
        
        # Bonus d'entropie: encourager l'exploration
        entropy = -torch.sum(action_probs * log_probs, dim=1).mean()
        
        # Perte d'action invalide: pénaliser la masse de probabilité sur les actions invalides
        invalid_probs = action_probs * (1 - valid_action_masks_tensor)
        invalid_action_loss = invalid_probs.sum(dim=1).mean()
        
        # Perte totale de l'acteur: combinaison pondérée des composantes
        total_actor_loss = (
            policy_loss * self.policy_loss_coeff
            + target_match_loss * 0.5  # Coefficient pour l'alignement avec la cible MCCFR
            + invalid_action_loss * self.invalid_action_loss_coeff
            - entropy * self.entropy_coeff  # Le négatif car on veut maximiser l'entropie
        )

        # Étape d'optimisation pour le critique
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Étape d'optimisation pour l'acteur
        self.optimizer.zero_grad()
        total_actor_loss.backward()
        self.optimizer.step()

        # Métriques pour le suivi
        metrics = {
            'reward_norm_mean': rewards_tensor.mean().item(),
            'critic_loss': critic_loss.item(),
            'state_value_loss': state_value_loss.item() * self.value_loss_coeff,
            'policy_loss': policy_loss.item() * self.policy_loss_coeff,
            'target_match_loss': target_match_loss.item() * 0.5,
            'invalid_action_loss': invalid_action_loss.item() * self.invalid_action_loss_coeff,
            'entropy': entropy.item() * self.entropy_coeff,
            'total_actor_loss': total_actor_loss.item(),
            'total_critic_loss': total_critic_loss.item()
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
        if hasattr(self, 'critic_optimizer'):
            self.critic_optimizer.zero_grad(set_to_none=True)
        
        # Cleanup device cache
        try:
            torch.cuda.empty_cache() 
        except:
            pass
        try:
            torch.mps.empty_cache()
        except:
            pass