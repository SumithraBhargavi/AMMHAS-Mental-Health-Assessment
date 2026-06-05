import os
import sys
import torch
import numpy as np

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = "/content/drive/MyDrive/Software"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------- IMPORT MODELS ----------------
from ml_training.text_rnn import TextRNN
from ml_training.audio_gru import AudioGRU
from ml_training.visual_gru import VisualGRU
from ml_training.fusion_mlp import FusionMLP

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PATHS ----------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TEXT_MODEL_PATH   = os.path.join(MODEL_DIR, "text_rnn.pt")
AUDIO_MODEL_PATH  = os.path.join(MODEL_DIR, "audio_gru.pt")
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_gru.pt")
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_mlp.pt")
NORM_PATH         = os.path.join(MODEL_DIR, "fusion_norm.pt")

# ---------------- LOAD MODELS ----------------
text_model = TextRNN(input_dim=300, hidden_dim=128, num_layers=1, output_dim=7)
audio_model = AudioGRU(input_dim=74)
visual_model = VisualGRU(input_dim=713)
fusion_model = FusionMLP(input_dim=21)

text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=DEVICE))
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
visual_model.load_state_dict(torch.load(VISUAL_MODEL_PATH, map_location=DEVICE))
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))

text_model.to(DEVICE).eval()
audio_model.to(DEVICE).eval()
visual_model.to(DEVICE).eval()
fusion_model.to(DEVICE).eval()

# ---------------- LOAD NORMALIZATION ----------------
norm = torch.load(NORM_PATH)
fusion_mean = norm["mean"].to(DEVICE)
fusion_std  = norm["std"].to(DEVICE)

# ---------------- EMOTIONS ----------------
EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "neutral"
]

# ---------------- INFERENCE FUNCTION ----------------
@torch.no_grad()
def analyze(text_feat, audio_feat, visual_feat):
    """
    text_feat   : np.ndarray (T, 300)
    audio_feat  : np.ndarray (T, 74)
    visual_feat : np.ndarray (T, 713)
    """

    # ---- TEXT ----
    t = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_len = torch.tensor([t.shape[1]]).to(DEVICE)
    t_pred = text_model(t, t_len)

    # ---- AUDIO (HARDENED) ----
    a = torch.tensor(audio_feat, dtype=torch.float32)
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = a.unsqueeze(0).to(DEVICE)
    a_len = torch.tensor([a.shape[1]]).to(DEVICE)
    a_pred = audio_model(a, a_len)
    a_pred = torch.nan_to_num(a_pred, nan=0.0)

    # ---- VISUAL ----
    v = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    v_len = torch.tensor([v.shape[1]]).to(DEVICE)
    v_pred = visual_model(v, v_len)

    # ---- FUSION ----
    fused = torch.cat([t_pred, a_pred, v_pred], dim=1)
    fused = (fused - fusion_mean) / fusion_std
    fused = torch.nan_to_num(fused, nan=0.0)

    out = fusion_model(fused).squeeze(0).cpu().numpy()

    return dict(zip(EMOTIONS, out))
