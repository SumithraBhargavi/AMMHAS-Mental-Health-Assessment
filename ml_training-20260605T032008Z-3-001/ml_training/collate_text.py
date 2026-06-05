# ml_training/collate_text.py

import torch
import numpy as np

def collate_text_rnn(batch):
    """
    Pads variable-length text sequences for RNN input.

    Returns:
      texts   : (B, T_max, 300)
      lengths : (B,)
      labels  : (B, 7)
    """
    texts = []
    labels = []

    for b in batch:
        t = b["text"]

        # ensure numpy array
        t = np.asarray(t, dtype="float32")

        # ensure 2D (T, 300)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)

        texts.append(t)
        labels.append(b["label"])

    lengths = torch.LongTensor([t.shape[0] for t in texts])

    T_max = lengths.max().item()
    D = texts[0].shape[1]
    B = len(texts)

    padded = np.zeros((B, T_max, D), dtype="float32")

    for i, t in enumerate(texts):
        padded[i, :t.shape[0], :] = t

    texts = torch.from_numpy(padded)
    labels = torch.from_numpy(np.stack(labels).astype("float32"))

    return texts, lengths, labels
