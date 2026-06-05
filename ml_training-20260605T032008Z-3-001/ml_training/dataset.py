import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MoseiNpyDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # =====================
        # TEXT
        # =====================
        text = np.load(row["text_wordvec_path"]).astype("float32")

        if text.ndim == 1:
            text = text.reshape(1, -1)
        elif text.ndim > 2:
            text = text.reshape(text.shape[0], -1)

        text = np.nan_to_num(text, nan=0.0, posinf=0.0, neginf=0.0)

        # =====================
        # AUDIO  (IMPORTANT FIX)
        # =====================
        audio = np.load(row["audio_covarep_path"]).astype("float32")

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim > 2:
            audio = audio.reshape(audio.shape[0], -1)

        # 🔑 THIS LINE FIXES EVERYTHING
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # =====================
        # LABELS
        # =====================
        raw_label = np.load(row["labels_path"]).astype("float32")

        if raw_label.ndim == 2:
            if raw_label.shape[0] == 7:
                label = raw_label.mean(axis=1)
            elif raw_label.shape[1] == 7:
                label = raw_label.mean(axis=0)
            else:
                raise ValueError(f"Unexpected label shape: {raw_label.shape}")
        else:
            label = raw_label

        label = np.nan_to_num(label, nan=0.0)

        return {
            "id": row["id"],
            "text": text,
            "audio": audio,
            "label": label
        }
