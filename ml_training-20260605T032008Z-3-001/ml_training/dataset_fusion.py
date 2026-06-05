import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    def __init__(self, text_preds, audio_preds, visual_preds, labels):
        """
        All inputs must be aligned lists
        """
        self.x = torch.cat(
            [text_preds, audio_preds, visual_preds], dim=1
        )
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
