# poker_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCriticModel, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )

        # Separate streams for actor and critic
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size),
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.action_size = action_size

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(module.bias)

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor: Predict action probabilities for all actions
        action_logits = self.actor_layers(shared_features)
        action_probs = F.softmax(action_logits, dim=1)

        # Critic: Predict state value
        state_value = self.critic_layers(shared_features).squeeze(-1)

        return action_probs, state_value
