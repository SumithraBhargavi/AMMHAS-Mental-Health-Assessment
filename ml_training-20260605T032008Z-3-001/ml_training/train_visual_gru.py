import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from importlib import reload

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = "/content/drive/MyDrive/Software"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------- IMPORTS ----------------
import ml_training.dataset
import ml_training.collate_visual
import ml_training.visual_gru

reload(ml_training.dataset)
reload(ml_training.collate_visual)
reload(ml_training.visual_gru)

from ml_training.dataset_visual import MoseiVisualDataset
from ml_training.collate_visual import collate_visual_rnn
from ml_training.visual_gru import VisualGRU
# ---------------- PATHS ----------------
TRAIN_CSV = "/content/drive/MyDrive/Software/npy/train_metadata.csv"
VAL_CSV   = "/content/drive/MyDrive/Software/npy/val_metadata.csv"

SAVE_DIR  = "/content/drive/MyDrive/Software/models"
SAVE_PATH = os.path.join(SAVE_DIR, "visual_gru.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- SPEED CONTROL ----------------
MAX_TRAIN_SAMPLES = 500
MAX_VAL_SAMPLES   = 100
MAX_EPOCHS = 3
BATCH_SIZE = 4
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- DATA ----------------
VISUAL_DIR = "/content/drive/MyDrive/Software/npy/visual_openface"

train_ds_full = MoseiVisualDataset(
    csv_path=TRAIN_CSV,
    visual_dir=VISUAL_DIR
)

val_ds_full = MoseiVisualDataset(
    csv_path=VAL_CSV,
    visual_dir=VISUAL_DIR
)

train_ds = Subset(train_ds_full, range(MAX_TRAIN_SAMPLES))
val_ds   = Subset(val_ds_full,   range(MAX_VAL_SAMPLES))

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_visual_rnn
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_visual_rnn
)

# ---------------- MODEL ----------------
sample = train_ds_full[0]["visual"]
INPUT_DIM = sample.shape[1]

model = VisualGRU(input_dim=INPUT_DIM).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN ----------------
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0

    for visuals, lengths, labels in train_loader:
        visuals = visuals.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(visuals, lengths)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for visuals, lengths, labels in val_loader:
            visuals = visuals.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(visuals, lengths)
            loss = criterion(preds, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

# ---------------- SAVE ----------------
torch.save(model.state_dict(), SAVE_PATH)
print("\n✅ Visual GRU model saved at:")
print(SAVE_PATH)
