import torch
import torch.nn as nn

class VisualGRU(nn.Module):
    def __init__(
        self,
        input_dim=713,
        hidden_dim=128,
        num_layers=1,
        output_dim=7,
        bidirectional=True
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Linear(gru_out_dim, output_dim)

    def forward(self, x, lengths):
        """
        x: (B, T, 713)
        lengths: (B,)
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        # Take last valid timestep for each sample
        idx = (lengths - 1).view(-1, 1, 1)
        idx = idx.expand(out.size(0), 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)

        return self.fc(last)
