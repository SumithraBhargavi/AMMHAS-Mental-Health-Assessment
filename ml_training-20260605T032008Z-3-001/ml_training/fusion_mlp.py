import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, input_dim=21, hidden1=64, hidden2=32, output_dim=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
