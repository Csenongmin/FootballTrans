import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MorphologySequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMMorphologyModel(nn.Module):
    def __init__(self, input_dim=46, hidden_dim=128, num_layers=2, num_classes=45, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        # out: [batch, seq_len, hidden_dim]

        logits = self.classifier(out)
        # logits: [batch, seq_len, num_classes]

        return logits
    
class TransformerMorphologyModel(nn.Module):
    def __init__(
            self,
            input_dim,
            num_classes,
            seq_len=75,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.2
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        #frame feature -> transformer embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        #learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1,seq_len, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]

        x = self.input_proj(x)
        # x: [batch, seq_len, d_model]

        x = x + self.pos_embedding[:, :x.size(1), :]
        # positional information 추가

        out = self.transformer_encoder(x)
        # out: [batch, seq_len, d_model]

        logits = self.classifier(out)
        # logits: [batch, seq_len, num_classes]

        return logits