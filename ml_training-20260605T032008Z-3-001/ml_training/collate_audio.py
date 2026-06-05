import torch
import numpy as np

# -------- SPEED CONTROL --------
MAX_FRAMES = 200   # <<<< KEY LINE (can be 100, 200, 300)

def collate_audio_rnn(batch):
    audios = []
    lengths = []
    labels = []

    for sample in batch:
        audio = sample["audio"]  # shape: (T, F)

        # ---- LIMIT AUDIO LENGTH ----
        if audio.shape[0] > MAX_FRAMES:
            # Uniform sampling across full sequence
            idx = np.linspace(0, audio.shape[0] - 1, MAX_FRAMES).astype(int)
            audio = audio[idx]

        audios.append(torch.tensor(audio, dtype=torch.float32))
        lengths.append(audio.shape[0])
        labels.append(torch.tensor(sample["label"], dtype=torch.float32))

    audios = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True
    )
    lengths = torch.tensor(lengths)
    labels = torch.stack(labels)

    return audios, lengths, labels
