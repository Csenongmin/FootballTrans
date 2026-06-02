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