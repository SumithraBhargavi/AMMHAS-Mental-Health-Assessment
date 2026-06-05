import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class AudioGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, output_dim=7):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        """
        x: (B, T_audio, A)
        lengths: (B,)
        """
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.gru(packed)

        # h_n shape: (2, B, hidden_dim)
        h_forward = h_n[-2]
        h_backward = h_n[-1]

        h = torch.cat((h_forward, h_backward), dim=1)
        out = self.fc(h)

        return out
