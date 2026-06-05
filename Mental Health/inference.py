import torch
import numpy as np
import os
import sys

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
MODEL_DIR  = "/content/drive/MyDrive/Software/models"
FUSION_DIR = "/content/drive/MyDrive/Software/npy/fusion"

TEXT_MODEL_PATH   = os.path.join(MODEL_DIR, "text_rnn.pt")
AUDIO_MODEL_PATH  = os.path.join(MODEL_DIR, "audio_gru.pt")
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_gru.pt")
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_mlp.pt")

FUSION_MEAN_PATH = os.path.join(FUSION_DIR, "fusion_mean.pt")
FUSION_STD_PATH  = os.path.join(FUSION_DIR, "fusion_std.pt")

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
fusion_mean = torch.load(FUSION_MEAN_PATH).to(DEVICE)
fusion_std  = torch.load(FUSION_STD_PATH).to(DEVICE)

# ---------------- EMOTIONS ----------------
EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "neutral"
]

# ---------------- INFERENCE ----------------
@torch.no_grad()
def analyze_sample(text_feat, audio_feat, visual_feat):
    """
    text_feat   : (T, 300)
    audio_feat  : (T, 74)
    visual_feat : (T, 713)
    """

    # ---- TEXT ----
    t = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_len = torch.tensor([t.shape[1]]).to(DEVICE)
    t_pred = text_model(t, t_len)

    # ---- AUDIO (sanitize) ----
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

    # ---- FUSION (CRITICAL FIX) ----
    fusion_input = torch.cat([t_pred, a_pred, v_pred], dim=1)

    # 🔑 NORMALIZE USING TRAIN STATS
    fusion_input = (fusion_input - fusion_mean) / fusion_std

    fusion_output = fusion_model(fusion_input)

    scores = fusion_output.squeeze(0).cpu().numpy()
    return dict(zip(EMOTIONS, scores))


# ---------------- TEST ----------------
if __name__ == "__main__":
    text_feat   = np.random.randn(40, 300)
    audio_feat  = np.random.randn(120, 74)
    visual_feat = np.random.randn(100, 713)

    result = analyze_sample(text_feat, audio_feat, visual_feat)
# Test 1
    text_feat = np.random.randn(40, 300)
    audio_feat = np.random.randn(120, 74)
    visual_feat = np.random.randn(100, 713)

    print(analyze_sample(text_feat, audio_feat, visual_feat))

# Test 2 (different sizes)
    text_feat = np.random.randn(20, 300)
    audio_feat = np.random.randn(200, 74)
    visual_feat = np.random.randn(60, 713)

    print(analyze_sample(text_feat, audio_feat, visual_feat))

    print("\nInference output:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
