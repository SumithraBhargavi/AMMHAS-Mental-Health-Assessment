import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = "/content/drive/MyDrive/Software"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------- IMPORTS ----------------
from ml_training.dataset import MoseiNpyDataset
from ml_training.dataset_visual import MoseiVisualDataset

from ml_training.text_rnn import TextRNN
from ml_training.audio_gru import AudioGRU
from ml_training.visual_gru import VisualGRU

# ---------------- PATHS ----------------
CSV_PATH = "/content/drive/MyDrive/Software/npy/train_metadata.csv"

TEXT_MODEL_PATH   = "/content/drive/MyDrive/Software/models/text_rnn.pt"
AUDIO_MODEL_PATH  = "/content/drive/MyDrive/Software/models/audio_gru.pt"
VISUAL_MODEL_PATH = "/content/drive/MyDrive/Software/models/visual_gru.pt"

VISUAL_DIR = "/content/drive/MyDrive/Software/npy/visual_openface"

SAVE_DIR = "/content/drive/MyDrive/Software/npy/fusion"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
text_model = TextRNN(input_dim=300, hidden_dim=128, num_layers=1, output_dim=7)
audio_model = AudioGRU(input_dim=74)
visual_model = VisualGRU(input_dim=713)

text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=DEVICE))
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
visual_model.load_state_dict(torch.load(VISUAL_MODEL_PATH, map_location=DEVICE))

text_model.to(DEVICE).eval()
audio_model.to(DEVICE).eval()
visual_model.to(DEVICE).eval()

# ---------------- DATASETS ----------------
text_audio_ds = MoseiNpyDataset(CSV_PATH)
visual_ds = MoseiVisualDataset(CSV_PATH, VISUAL_DIR)

text_preds = []
audio_preds = []
visual_preds = []
labels = []

# ---------------- GENERATE PREDICTIONS ----------------
with torch.no_grad():
    for i in tqdm(range(len(text_audio_ds))):
        sample_ta = text_audio_ds[i]
        sample_v  = visual_ds[i]

        # ---- TEXT ----
        text = torch.tensor(sample_ta["text"]).unsqueeze(0).to(DEVICE)
        text_len = torch.tensor([text.shape[1]]).to(DEVICE)
        t_pred = text_model(text, text_len).cpu()

        # ---- AUDIO ----
        # ---- AUDIO (HARDENED) ----
        audio = torch.tensor(sample_ta["audio"], dtype=torch.float32)

        # 1️⃣ Remove NaN / Inf
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # 2️⃣ Clamp extreme values (very important)
        audio = torch.clamp(audio, min=-50.0, max=50.0)

        audio = audio.unsqueeze(0).to(DEVICE)
        audio_len = torch.tensor([audio.shape[1]]).to(DEVICE)

        a_pred = audio_model(audio, audio_len)

        # 3️⃣ Remove NaN / Inf from model output
        a_pred = torch.nan_to_num(a_pred, nan=0.0, posinf=0.0, neginf=0.0)

        a_pred = a_pred.cpu()
        if torch.isnan(a_pred).any():
          print(f"❌ NaN audio prediction at index {i}")
          continue

        # ---- VISUAL ----
        visual = torch.tensor(sample_v["visual"]).unsqueeze(0).to(DEVICE)
        visual_len = torch.tensor([visual.shape[1]]).to(DEVICE)
        v_pred = visual_model(visual, visual_len).cpu()

        text_preds.append(t_pred)
        audio_preds.append(a_pred)
        visual_preds.append(v_pred)
        labels.append(torch.tensor(sample_ta["label"]).unsqueeze(0))

# ---------------- SAVE ----------------
torch.save(torch.cat(text_preds),  os.path.join(SAVE_DIR, "text_preds.pt"))
torch.save(torch.cat(audio_preds), os.path.join(SAVE_DIR, "audio_preds.pt"))
torch.save(torch.cat(visual_preds),os.path.join(SAVE_DIR, "visual_preds.pt"))
torch.save(torch.cat(labels),      os.path.join(SAVE_DIR, "labels.pt"))

print("✅ Fusion training data saved")
