import os
import numpy as np
from torch.utils.data import Dataset
from ml_training.dataset import MoseiNpyDataset


class MoseiVisualDataset(Dataset):
    def __init__(self, csv_path, visual_dir):
        self.visual_dir = visual_dir
        self.label_ds = MoseiNpyDataset(csv_path)

    def __len__(self):
        return len(self.label_ds)

    def __getitem__(self, idx):
        sample = self.label_ds[idx]
        sample_id = sample["id"]

        visual_path = os.path.join(self.visual_dir, sample_id + ".npy")
        visual = np.load(visual_path)

        return {
            "visual": visual,
            "label": sample["label"]
        }
