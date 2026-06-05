import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = "/content/drive/MyDrive/Software"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml_training.fusion_mlp import FusionMLP

# ---------------- PATHS ----------------
FUSION_DIR = "/content/drive/MyDrive/Software/npy/fusion"

TEXT_PRED_PATH   = os.path.join(FUSION_DIR, "text_preds.pt")
AUDIO_PRED_PATH  = os.path.join(FUSION_DIR, "audio_preds.pt")
VISUAL_PRED_PATH = os.path.join(FUSION_DIR, "visual_preds.pt")
LABEL_PATH       = os.path.join(FUSION_DIR, "labels.pt")

SAVE_PATH = "/content/drive/MyDrive/Software/models/fusion_mlp.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- LOAD DATA ----------------
text_preds   = torch.load(TEXT_PRED_PATH)
audio_preds  = torch.load(AUDIO_PRED_PATH)
visual_preds = torch.load(VISUAL_PRED_PATH)
labels       = torch.load(LABEL_PATH)

# ---------------- HARDEN DATA (MANDATORY) ----------------
text_preds   = torch.nan_to_num(text_preds, nan=0.0, posinf=0.0, neginf=0.0)
audio_preds  = torch.nan_to_num(audio_preds, nan=0.0, posinf=0.0, neginf=0.0)
visual_preds = torch.nan_to_num(visual_preds, nan=0.0, posinf=0.0, neginf=0.0)
labels       = torch.nan_to_num(labels, nan=0.0)

# ---------------- CONCAT FEATURES ----------------
X = torch.cat([text_preds, audio_preds, visual_preds], dim=1)
y = labels

# ---------------- NORMALIZE INPUT (CRITICAL FIX) ----------------
X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)

print("Fusion input shape:", X.shape)
print("Fusion label shape:", y.shape)
print("Any NaN in X:", torch.isnan(X).any().item())
print("Any NaN in y:", torch.isnan(y).any().item())

# ---------------- DATASET ----------------
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---------------- MODEL ----------------
model = FusionMLP(input_dim=21).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ---------------- TRAIN ----------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)

        loss = criterion(preds, yb)
        loss.backward()

        # 🔒 PREVENT GRADIENT EXPLOSION
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f}")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), SAVE_PATH)
print("\n✅ Fusion model saved at:")
print(SAVE_PATH)
