# poker_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# -------- Hyperparamètres du modèle Transformer Hero --------

HERO_TRANSFORMER_INPUT_DIM = 116
HERO_TRANSFORMER_OUTPUT_DIM = 16

HERO_TRANSFORMER_NHEAD = 2
HERO_TRANSFORMER_NUM_LAYERS = 2
HERO_TRANSFORMER_DIM_FEEDFORWARD = 256
HERO_TRANSFORMER_D_MODEL = 32

HERO_TRANSFORMER_MAX_LEN = 10


# -------- Hyperparamètres du modèle Transformer History --------

HISTORY_TRANSFORMER_INPUT_DIM = 33
HISTORY_TRANSFORMER_OUTPUT_DIM = 16

HISTORY_TRANSFORMER_NHEAD = 2
HISTORY_TRANSFORMER_NUM_LAYERS = 2
HISTORY_TRANSFORMER_DIM_FEEDFORWARD = 256
HISTORY_TRANSFORMER_D_MODEL = 32

HISTORY_TRANSFORMER_MAX_LEN = 100

# -------- Hyperparamètres du modèle final --------

MODEL_DIM_FEEDFORWARD = 256
MODEL_D_MODEL = 32
MODEL_OUTPUT_DIM = 16


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

class PokerHeroTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Correction pour le RL : on se base sur HERO_TRANSFORMER_D_MODEL (actuellement 32)
        # Chaque vecteur d'état de dimension HERO_TRANSFORMER_INPUT_DIM sera projeté en un vecteur de dimension HERO_TRANSFORMER_D_MODEL.
        # Remarque : la variable locale "d_model = 64" a été supprimée car non utilisée.
        self.input_projection = nn.Linear(HERO_TRANSFORMER_INPUT_DIM, HERO_TRANSFORMER_D_MODEL)
        
        self.pos_encoder = PositionalEncoding(HERO_TRANSFORMER_D_MODEL, max_len=HERO_TRANSFORMER_MAX_LEN)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HERO_TRANSFORMER_D_MODEL,
            nhead=HERO_TRANSFORMER_NHEAD,
            dim_feedforward=HERO_TRANSFORMER_DIM_FEEDFORWARD,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=HERO_TRANSFORMER_NUM_LAYERS
        )
        
        # Les têtes internes d'action et de value ne seront pas utilisées dans le cadre RL,
        # puisque la politique et la valeur seront calculées dans le PokerModel.
        self.action_head = nn.Sequential(
            nn.Linear(HERO_TRANSFORMER_D_MODEL, HERO_TRANSFORMER_DIM_FEEDFORWARD),
            nn.GELU(),
            nn.LayerNorm(HERO_TRANSFORMER_DIM_FEEDFORWARD),
            nn.Linear(HERO_TRANSFORMER_DIM_FEEDFORWARD, HERO_TRANSFORMER_OUTPUT_DIM),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(HERO_TRANSFORMER_D_MODEL, HERO_TRANSFORMER_DIM_FEEDFORWARD),
            nn.GELU(),
            nn.LayerNorm(HERO_TRANSFORMER_DIM_FEEDFORWARD),
            nn.Linear(HERO_TRANSFORMER_DIM_FEEDFORWARD, 1)
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

    # Nouvelle méthode pour extraire directement la représentation du hero (le last_hidden)
    def extract_features(self, x):
        # Créer un masque de padding basé sur les valeurs nulles
        padding_mask = (x.sum(dim=-1) == 0)  # Shape: (batch_size, seq_len)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x, mask=None, src_key_padding_mask=padding_mask)
        last_hidden = transformer_out[:, -1]
        return last_hidden
    

class PokerHistoryTransformer(nn.Module):
    """
    Transformer dédié au traitement de l'historique des actions.
    
    Ce modèle prend en entrée les vecteurs tokenisés générés par get_tokenized_history.
    Par défaut, chaque token est de dimension 33.
    """
    def __init__(self, 
                 input_dim=HISTORY_TRANSFORMER_INPUT_DIM,                 # Dimension de chaque token (issus de get_tokenized_history)
                 d_model=HISTORY_TRANSFORMER_D_MODEL,                     # Dimension de l'espace latent
                 nhead=HISTORY_TRANSFORMER_NHEAD,                         # Nombre de "heads" dans le mécanisme d'attention
                 num_layers=HISTORY_TRANSFORMER_NUM_LAYERS,               # Nombre de couches dans l'encodeur Transformer
                 dim_feedforward=HISTORY_TRANSFORMER_DIM_FEEDFORWARD,     # Dimension du réseau feed-forward interne
                 history_output_dim=HISTORY_TRANSFORMER_OUTPUT_DIM,       # Dimension de la représentation finale (peut servir de feature ou pour une tâche spécifique)
                 max_len=HISTORY_TRANSFORMER_MAX_LEN):                    # Longueur maximale de la séquence
        
        super().__init__()
        # Projection linéaire pour passer de input_dim à d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        # Encodage positionnel
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        # Définition d'une couche d'encodeur Transformer (batch_first=True pour des tenseurs de forme (batch, seq, features))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Tête de sortie afin d'extraire une représentation fixe de l'historique
        self.history_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, history_output_dim)
        )

    def forward(self, x):
        """
        x: Tenseur de forme (batch_size, seq_len, input_dim)
           Ce tenseur peut être obtenu en transposant la sortie de get_tokenized_history
           (qui est initialement de forme (input_dim, seq_len)).
        """
        # On crée un masque de padding afin d'ignorer les tokens dont la somme des valeurs est nulle
        src_key_padding_mask = (x.abs().sum(dim=-1) == 0)  # forme: (batch_size, seq_len)
        # Projection linéaire
        x = self.input_projection(x)
        # Ajout de l'encodage positionnel
        x = self.pos_encoder(x)
        # Passage dans l'encodeur Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # Agrégation de la séquence.
        # Ici, nous utilisons le dernier token de la séquence (en supposant que le padding est ajouté en fin de séquence)
        out = x[:, -1, :]  # forme: (batch_size, d_model)
        # Passage par la tête de sortie
        out = self.history_head(out)
        return out
    
    # Exemple d'utilisation:
    # Si get_tokenized_history retourne un tenseur de forme (33, seq_len),
    # il faudra le transposer pour obtenir (batch_size, seq_len, 33)
    # par exemple:
    #
    # history_tensor = game.get_tokenized_history()  # shape: (33, seq_len)
    # history_tensor = history_tensor.transpose(0, 1).unsqueeze(0)  # shape: (1, seq_len, 33)
    # model = PokerHistoryTransformer()
    # output = model(history_tensor) 


class PokerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Instancier le transformer pour le state du hero
        self.state_transformer = PokerHeroTransformer()
        # Instancier le transformer pour l'historique des actions
        self.history_transformer = PokerHistoryTransformer()
        
        # Dimension combinée : 
        # - from state_transformer : HERO_TRANSFORMER_D_MODEL (la représentation extraite par extract_features)
        # - from history_transformer : HISTORY_TRANSFORMER_OUTPUT_DIM (par défaut 16)
        combined_dim = HERO_TRANSFORMER_D_MODEL + HISTORY_TRANSFORMER_OUTPUT_DIM
        
        # Tête pour la politique (les logits seront utilisés pour la distribution categorical)
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, MODEL_DIM_FEEDFORWARD),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM_FEEDFORWARD),
            nn.Linear(MODEL_DIM_FEEDFORWARD, HERO_TRANSFORMER_OUTPUT_DIM)  # Taille de sortie = nombre d'actions (ici 16)
        )
        
        # Tête pour la valeur (qui prédit un scalaire)
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, MODEL_DIM_FEEDFORWARD),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM_FEEDFORWARD),
            nn.Linear(MODEL_DIM_FEEDFORWARD, 1)
        )
        
    def forward(self, state_input, history_input):
        """
        state_input  : Tenseur pour le PokerHeroTransformer de forme (batch_size, seq_len, HERO_TRANSFORMER_INPUT_DIM).
        history_input: Tenseur pour le PokerHistoryTransformer de forme (batch_size, seq_len, input_dim)
                       (typiquement input_dim=33).
        """
        # Extraire la représentation du hero (state) sans passer par les têtes internes du PokerHeroTransformer
        state_features = self.state_transformer.extract_features(state_input)  # Shape: (batch_size, HERO_TRANSFORMER_D_MODEL)
        # Obtenir la représentation de l'historique des actions
        history_features = self.history_transformer(history_input)             # Shape: (batch_size, HISTORY_TRANSFORMER_OUTPUT_DIM)
        # Concaténer les deux représentations
        combined_features = torch.cat([state_features, history_features], dim=-1)  # Shape: (batch_size, combined_dim)
        
        # Calcul des sorties RL :
        # - policy_logits : logits de la politique (à passer à torch.distributions.Categorical par exemple)
        # - state_value   : valeur estimée de l'état
        policy_logits = self.policy_head(combined_features)
        state_value = self.value_head(combined_features)
        return policy_logits, state_value 