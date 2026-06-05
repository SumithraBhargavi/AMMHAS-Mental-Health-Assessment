import sys
sys.path.insert(0, "/content/drive/MyDrive/Software/mental_health_backend")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile, os

from backend.inference_engine import analyze  # fixed import

app = FastAPI(title="Multimodal Emotion Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fake_text_features(text):
    return np.random.randn(40, 300).astype(np.float32)

def fake_audio_features(path):
    return np.random.randn(80, 74).astype(np.float32)

def fake_visual_features(path):
    return np.random.randn(60, 713).astype(np.float32)

@app.post("/analyze")
async def analyze_endpoint(
    text: str = Form(...),
    audio: UploadFile = File(...),
    video: UploadFile = File(...)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, audio.filename)
        video_path = os.path.join(tmpdir, video.filename)

        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        with open(video_path, "wb") as f:
            f.write(await video.read())

        text_feat   = fake_text_features(text)
        audio_feat  = fake_audio_features(audio_path)
        visual_feat = fake_visual_features(video_path)

        emotions = analyze(text_feat, audio_feat, visual_feat)
        emotions = {k: float(v) for k, v in emotions.items()}
        dominant = max(emotions, key=emotions.get)

        return {"status": "success", "emotions": emotions, "dominant_emotion": dominant}

@app.get("/")
def health():
    return {"status": "API running"}