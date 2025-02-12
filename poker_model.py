# poker_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=101):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len]

class PokerTransformerModel(nn.Module):
    def __init__(self, input_dim=116, output_dim=5, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        
        # Embedding dimension must be divisible by number of heads
        d_model = nhead * 8  # 32 if nhead=4
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Create positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Set batch_first=True in TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # Set this to True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.action_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        print('Shape de l input', x.shape)
        
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask for variable length sequences
        padding_mask = None  # We don't need padding mask since all sequences are valid
        
        # Pass through transformer
        transformer_out = self.transformer(x, mask=None, src_key_padding_mask=padding_mask)
        
        # Use the last sequence element for predictions
        last_hidden = transformer_out[:, -1]
        
        # Get action probabilities and state value
        action_probs = self.action_head(last_hidden)
        state_value = self.value_head(last_hidden)
        
        return action_probs, state_value
