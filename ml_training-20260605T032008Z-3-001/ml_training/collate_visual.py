import torch
import numpy as np

MAX_FRAMES = 200

def collate_visual_rnn(batch):
    visuals = []
    lengths = []
    labels = []

    for sample in batch:
        visual = sample["visual"]   # ✅ FIXED

        visual = np.nan_to_num(
            visual, nan=0.0, posinf=0.0, neginf=0.0
        )

        if visual.shape[0] > MAX_FRAMES:
            idx = np.linspace(0, visual.shape[0] - 1, MAX_FRAMES).astype(int)
            visual = visual[idx]

        visuals.append(torch.tensor(visual, dtype=torch.float32))
        lengths.append(visual.shape[0])
        labels.append(torch.tensor(sample["label"], dtype=torch.float32))

    visuals = torch.nn.utils.rnn.pad_sequence(
        visuals, batch_first=True
    )
    lengths = torch.tensor(lengths)
    labels = torch.stack(labels)

    return visuals, lengths, labels
